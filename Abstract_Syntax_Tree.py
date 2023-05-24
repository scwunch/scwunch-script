# import re
import Env
from Syntax import *


class Tokenizer:
    """
    lines: list[Line]
    ln, col: indices for line and column
    char, current_line: current character and line of reader
    in_string: ' or " or None
    """
    idx: int
    indent: int
    ln: int
    col: int
    char: str | None
    in_string: list[str]
    fn_lv: int
    tokens: list[Token]
    def __init__(self, script: str):
        self.script = script
        self.idx = -1
        self.ln, self.col = 0, 0
        self.char = None
        self.next_line()
        self.in_string = []
        self.fn_lv = 0
        self.tokens = self.read_tokens()

    def __repr__(self):
        head = f"Tokenizer State:\n\t" \
               f"head:({self.ln}, {self.col})\n\tcurrent_line:{self.ln}\n\tchar:{self.char}\n"
        lines = []
        ln = 0
        start = 0
        indent = ''
        for i, tok in enumerate(self.tokens):
            if tok.type == TokenType.NewLine:
                ln = tok.pos[0]
            if tok.type in (TokenType.NewLine, TokenType.BlockEnd):
                lines.append(f"{ln}. {indent}{' '.join(t.source_text.strip() for t in self.tokens[start:i])}")
                start = i+1
            if tok.type in (TokenType.BlockStart, TokenType.BlockEnd):
                indent = tok.source_text
        return head + "\n".join(lines)

    def read_tokens(self):
        tokens: list[Token] = []
        while self.char:
            start = self.idx
            pos = (self.ln, self.col)
            token_type = None
            if re.match(r'\d', self.char):
                text = self.read_number()
            elif self.char == "`":
                text = self.read_string_literal()
                token_type = TokenType.StringLiteral
            elif self.char in ('"', "'"):
                text = self.char
                if len(self.in_string) > self.fn_lv:
                    token_type = TokenType.StringEnd
                    self.in_string.pop()
                else:
                    token_type = TokenType.StringStart
                    self.in_string.append(text)
            elif re.match(r'\w', self.char):
                text = self.read_word()
            elif re.match(op_char_patt, self.char):
                text = self.read_operator()
            elif self.char in "{}":
                text = self.char
                if self.in_string:
                    if self.char == '{' and len(self.in_string) - self.fn_lv == 1:
                        self.fn_lv += 1
                    elif self.char == '}':
                        self.fn_lv -= 1
            elif self.char in '[](),':
                text = self.char
            elif self.char in "#\n":
                last_indent = self.indent
                self.next_line()
                if self.indent == last_indent:
                    text = '\n'
                    token_type = TokenType.NewLine
                else:
                    text = '\t' * self.indent
                    if self.indent > last_indent:
                        token_type = TokenType.BlockStart
                    else:
                        token_type = TokenType.BlockEnd
            elif self.char == '\\':
                while self.next_char() and self.char in ' \t':
                    pass
                if self.char != '\n':
                    raise Env.SyntaxErr(f"Expected newline after backslash at {pos}")
                self.next_line()
                continue
            else:
                raise Env.SyntaxErr(f'{pos} What kind of character is this? "{self.char}"')
            if text == 'debug':
                pass
            tokens.append(Token(text, pos, token_type))
            if len(self.in_string) > self.fn_lv:
                tokens.append(Token(self.read_string(self.in_string[-1]),  (self.ln, self.col),  TokenType.StringPart))
            while self.idx == start or self.char and self.char in ' \t':
                self.next_char()
        return tokens

    def next_char(self, count=1):
        self.idx += count
        if self.idx >= len(self.script):
            self.char = None
            return None
        self.col += count
        self.char = self.script[self.idx]
        return self.char

    def next_line(self):
        while self.char and self.char != '\n':
            self.next_char()
        self.ln += 1
        self.indent = 0
        while self.next_char():
            if self.char == '\t':
                self.indent += 1
            elif self.script[self.idx:self.idx+4] == '    ':
                self.next_char(3)
                self.indent += 1
            elif self.char in '#\n':
                return self.next_line()
            else:
                break
        self.col = 1

    def read_number(self):
        num_text = self.char
        while self.next_char():
            if self.char == '_':
                continue
            elif re.match(r'[\d.]', self.char):
                num_text += self.char
            else:
                break
        if self.char == 'd':
            num_text += 'd'
            self.next_char()
        return num_text

    def read_string(self, quote: str) -> str:
        str_text = ""
        str_start = self.idx
        while self.next_char():
            if self.char == "\\":
                str_text += self.char + (self.next_char() or "")
                if self.char is None:
                    raise NotImplemented("Should detect newline at line ", self.ln + 1)
                continue
            if self.char in ("{", quote):
                return str_text
            str_text += self.char
        raise Env.SyntaxErr(f"Unterminated string at {{ln={self.ln + 1}, ch={self.col + 1}}}: "
                            f"{self.script[str_start:self.idx]} "
                            f"{str_text}")
    def read_string_literal(self):
        str_start = self.idx
        backticks = 1
        while self.next_char() and self.char == "`":
            backticks += 1
        str_text = "`" * backticks
        while self.char:
            str_text += self.char
            if str_text.endswith("`" * backticks):
                self.next_char()
                return str_text
            self.next_char()
        raise Env.SyntaxErr(f"Unterminated string at {{ln={self.ln + 1}, ch={self.col + 1}}}: "
                            f"{self.script[str_start:self.idx]} "
                            f"{str_text}")

    def read_word(self):
        word = self.char
        while self.next_char() and re.match(r'\w', self.char):
            word += self.char
        return word

    def read_operator(self):
        op = self.char
        while self.next_char() and re.match(op_char_patt, self.char):
            op += self.char
        return op

    def peek(self, offset=1, count=1):
        index = self.col + offset
        if 0 <= index < len(self.script):  # current_line.text):
            return self.script[index:index + count]
        return None


class AST:
    def __init__(self, toks: Tokenizer):
        # super().__init__()
        self.tokens = toks.tokens
        self.idx = 0
        self.tok = self.tokens[0] if self.tokens else None
        self.block = self.read_block(0)

    def peek(self, count=1):
        i = self.idx + count
        if 0 <= i < len(self.tokens):
            return self.tokens[i]
        else:
            return None

    def seek(self, count=1):
        self.idx += count
        if self.idx >= len(self.tokens):
            self.tok = None
            return None
        self.tok = self.tokens[self.idx]
        return self.tok

    def read_block(self, indent: int) -> Block:
        executables: list[Statement] = []
        while self.tok:
            statement = self.read_statement(TokenType.NewLine, TokenType.BlockEnd)
            if statement.nodes:
                executables.append(statement)
            if self.tok is None:
                break
            elif self.tok.type == TokenType.NewLine:
                self.seek()
            elif self.tok.type == TokenType.BlockEnd:
                self.indent = len(self.tok.source_text)
                if self.indent < indent:
                    break
                else:
                    self.seek()

        return Block(executables)

    def read_statement(self, *end_of_statement: TokenType) -> Statement:
        nodes: list[Node] = []
        while self.tok:
            if self.tok.type in end_of_statement:
                return Statement(nodes)
            match self.tok.type:
                case TokenType.GroupEnd | TokenType.ListEnd | TokenType.FnEnd:
                    raise Env.SyntaxErr(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
                case TokenType.NewLine | TokenType.BlockEnd:
                    pass
                case TokenType.BlockStart:
                    indent = len(self.tok.source_text)
                    self.seek()
                    nodes.append(self.read_block(indent))
                    if self.indent == indent-1 and self.peek() and self.peek().source_text == 'else':
                        self.seek()
                    continue
                case TokenType.GroupStart:
                    if nodes and nodes[-1].type not in (TokenType.Operator, TokenType.Command, TokenType.Keyword):
                        nodes.append(Token('&'))
                    self.seek()
                    nodes.append(self.read_statement(TokenType.GroupEnd))
                case TokenType.ListStart:
                    if self.idx and self.peek(-1).type in \
                            (TokenType.Name, TokenType.GroupEnd, TokenType.ListEnd, TokenType.FnEnd):
                        nodes.append(Token('.', self.tok.pos))
                    self.seek()
                    nodes.append(List(self.read_list(TokenType.ListEnd)))
                case TokenType.FnStart:
                    self.seek()
                    nodes.append(FunctionLiteral(self.read_list(TokenType.FnEnd)))
                case TokenType.StringStart:
                    nodes.append(self.read_string())
                case TokenType.Comma:
                    self.tok.type = TokenType.Operator
                    nodes.append(self.tok)
                case _:
                    if self.tok.source_text == 'if' and not nodes:
                        self.tok.type = TokenType.Keyword
                    nodes.append(self.tok)
                    if self.tok.source_text == 'debug':
                        pass
            self.seek()
        raise Env.SyntaxErr('EOF reached.  Expected ', end_of_statement)

    def read_list(self, end: TokenType) -> list[Statement]:
        white_space_tokens = TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd
        items: list[Statement] = []
        while self.tok:
            while self.tok and self.tok.type in white_space_tokens:
                self.seek()
            if self.tok is None:
                break
            if self.tok.type == end:
                return items
            statement = self.read_statement(end, TokenType.Comma, *white_space_tokens)
            if statement.nodes:
                items.append(statement)
            else:
                continue
            if self.tok and self.tok.type == TokenType.Comma:
                self.seek()

        raise Env.SyntaxErr("Reached EOF: Expected " + repr(end))

    def read_string(self) -> StringNode:
        nodes = []
        line = self.tok.pos[0]
        while self.seek():
            match self.tok.type:
                case TokenType.StringPart:
                    nodes.append(self.tok)
                case TokenType.FnStart:
                    self.seek()
                    nodes.append(self.read_statement(TokenType.FnEnd))
                case TokenType.StringEnd:
                    return StringNode(nodes)
                case _:
                    raise Env.SyntaxErr(f"Found unexpected token in string on line {line}")
        raise Env.SyntaxErr(f"Unterminated string on line {line}")

    def __repr__(self):
        return '\n'.join((repr(expr) if expr else '') for expr in self.block.statements)


if __name__ == "__main__":
    script_path = "test_script.ss"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    print(repr(tokenizer))
    ast = AST(tokenizer)
    print(repr(ast))
