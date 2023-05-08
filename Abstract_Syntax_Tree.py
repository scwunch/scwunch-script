# import re
from Syntax import *


class Builder:
    def __init__(self):
        self.ln: int = 0
        self.col: int = 0
        self.lines: list[Line] = []
        self.current_line: Line | None = None

    def next_line(self):
        self.ln += 1
        if self.ln < len(self.lines):
            self.current_line = self.lines[self.ln]
            self.col = -1
            # self.read_char()
            # self.next_token()
        else:
            self.current_line = None
        return self.current_line


class Tokenizer(Builder):
    """
    lines: list[Line]
    ln, col: indices for line and column
    char, current_line: current character and line of reader
    """

    def __init__(self, script: str):
        super().__init__()

        # initialize lines
        script_lines = script.split("\n")
        for i, line in enumerate(script_lines):
            self.lines.append(Line(line, i+1))
        self.current_line: Line | None = self.lines[0]
        self.char = self.current_line.text[0]

        # add tokens to lines
        for line in self.lines:
            line.tokens = self.read_tokens()
            self.next_line()

    def __repr__(self):
        head = f"Tokenizer State:\n\t" \
               f"head:({self.ln}, {self.col})\n\tcurrent_line:{self.current_line}\n\tchar:{self.char}\n"
        return head + "\n".join(repr(line) for line in self.lines)

    def read_tokens(self):
        tokens: list[Token] = []
        while self.char:
            pos = (self.ln, self.col)
            if re.match(r'\d', self.char):
                text = self.read_number()
            elif self.char == '"' or self.char == "'" or self.char == "`":
                text = self.read_string()
            elif re.match(r'\w', self.char):
                text = self.read_word()
            elif re.match(op_char_patt, self.char):
                text = self.read_operator()
            elif re.match(r'[\[\]{}(),]', self.char):
                text = self.char
            elif self.char == '\\':
                text = self.char
            else:
                raise Exception("What kind of character is this?", self.char)
            # warnings
            if text == ':' and tokens and tokens[0].source_text in KeyWords:
                print(f"SYNTAX WARNING ({self.ln+1}): Pili does not use colons for control blocks like if and for.")
            tokens.append(Token(text, pos))
            while self.col == pos[1] or self.char and re.match(r'\s', self.char):
                self.next_char()
            # if self.char == '\\' and re.match(r'\\\s*', self.current_line.text[self.col:]):
            #     self.next_line()
        # self.tokens = tokens
        return tokens

    def next_char(self):
        self.col += 1
        if self.col >= len(self.current_line):
            self.char = None
        else:
            self.char = self.current_line.text[self.col]
        return self.char

    def next_line(self):
        if super().next_line():
            self.next_char()
            return self.current_line
        else:
            return None

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

    def read_string(self):
        quote = self.char
        str_text = quote
        while self.next_char() and self.char != quote:
            str_text += self.char
            if self.char == '\\':
                str_text += self.next_char()
        if self.char is None:
            raise Exception(f"Unterminated string at {{ln={self.ln+1}, ch={self.col+1}}}: ",
                            self.current_line.text[self.col:],
                            str_text)
        self.next_char()
        return str_text + quote

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
        if 0 <= index < len(self.lines):
            return self.current_line.text[index:index + count]
        return None


class AST(Builder):
    def __init__(self, toks: Tokenizer):
        super().__init__()
        self.lines = toks.lines
        self.current_line = self.lines[0]
        self.tok = self.current_line.tokens[0]
        self.block = self.read_block()

    def peek(self, count=1):
        i = self.col + count
        if 0 <= i < len(self.current_line.tokens):
            return self.current_line.tokens[i]
        else:
            return None

    def seek(self, count=1):
        self.col += count
        if self.col >= len(self.current_line.tokens):
            self.tok = None
            return None
        self.tok = self.current_line.tokens[self.col]
        return self.tok

    def next_line(self):
        if super().next_line():
            self.seek()
            return self.current_line
        else:
            return None

    def read_block(self) -> Block:
        indent = self.current_line.indent
        executables: list[Statement] = []
        first_line = self.current_line  # just for error checking
        while self.current_line is not None:
            if not self.current_line.tokens:
                self.next_line()
                continue
            if self.current_line.indent < indent:
                break
            else:
                executables.append(self.read_statement())

            # check to make sure we successfully advanced at least one line
            if first_line == self.current_line:
                raise Exception("read statement but failed to advance to next line")
            else:
                first_line = self.current_line

        block = Block(executables)
        return block

    def read_statement(self, *end_of_statement: TokenType) -> Statement:
        nodes: list[Node] = []
        while self.tok:
            if self.tok.type in end_of_statement:
                # self.seek()
                return Statement(nodes)
            elif self.tok.type in (TokenType.GroupEnd, TokenType.ListEnd, TokenType.FnEnd):
                raise Exception(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
            elif self.tok.type == TokenType.GroupStart:
                self.seek()
                nodes.append(self.read_statement(TokenType.GroupEnd))
            elif self.tok.type == TokenType.ListStart:
                if self.col and self.peek(-1).type in \
                        (TokenType.Name, TokenType.GroupEnd, TokenType.ListEnd, TokenType.FnEnd):
                    nodes.append(Token('.', self.tok.pos))
                self.seek()
                nodes.append(List(self.read_list(TokenType.ListEnd)))
            elif self.tok.type == TokenType.FnStart:
                self.seek()
                nodes.append(FunctionLiteral(self.read_list(TokenType.FnEnd)))
            elif self.tok.type == TokenType.Name:
                # or `,` or `]` too?
                # if self.peek() and self.peek().source_text.startswith(':') \
                #         and (self.peek(-1) is None or self.peek(-1).type != TokenType.Operator):
                #     # nodes.append(Token('&name'))
                #     # self.tok.type = TokenType.PatternName
                #     pass
                # elif self.peek(-1) and self.peek(-1).source_text == '.':
                #     self.tok.type = TokenType.PatternName
                nodes.append(self.tok)
            elif self.tok.source_text == '-' and (not self.peek(-1) or self.peek(-1).type == TokenType.Operator):
                nodes.append(self.tok)
            elif self.tok.source_text == 'if' and not nodes:
                self.tok.type = TokenType.Keyword
                nodes.append(self.tok)
            # elif self.tok.source_text == ':' and self.peek():
            #     nodes.append(self.tok)
            #     self.seek()
            #     return_node: list[Node] = [Token('return')]
            #     nodes.append(Block([Statement(return_node + self.read_statement().nodes)]))
            #       I thought I could be smart and automatically transform a `name: inline-function`
            #       pattern to a block ... but it doesn't work because there are some examples where
            #       blocks don't work, like in parameter expressions.
            elif self.tok.type == TokenType.Backslash:
                if not super().next_line():
                    raise SyntaxError("Expected statement after backslash at: ", self.ln, self.col)
            else:
                nodes.append(self.tok)
            self.seek()
        else:
            if end_of_statement:
                raise Exception("End of line, expected: " + repr(end_of_statement))

        indent = self.current_line.indent
        if self.next_line():
            if self.current_line.indent > indent:
                sub_block = self.read_block()
                nodes.append(sub_block)
                if self.tok and self.tok.source_text == 'else':
                    # self.seek()
                    nodes += self.read_statement().nodes

        executable = Statement(nodes)
        return executable

    def read_list(self, end: TokenType) -> list[Statement]:
        if self.tok.type == end:
            return []
        items: list[Statement] = []
        while self.tok:
            items.append(self.read_statement(end, TokenType.Comma))
            if self.tok.type == TokenType.Comma:
                self.seek()
            elif self.tok.type == end:
                break
            else:
                raise Exception(f"Unexpected token {repr(self.tok)}; expected {end}")

        return items

    def __repr__(self):
        return '\n'.join((repr(expr) if expr else '') for expr in self.block.statements)
        # (print(line) for line in self.block.statements)


if __name__ == "__main__":
    script_path = "test_script.ss"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    print(repr(tokenizer))
    ast = AST(tokenizer)
    print(repr(ast))
