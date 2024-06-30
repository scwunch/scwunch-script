import Env
from operators import Op
from Syntax import *

print(f"loading module: {__name__} ...")

class Tokenizer:
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
        self.ln, self.col = 1, 0
        self.char = '\n'
        self.next_line()
        self.in_string = []
        self.fn_lv = 0
        self.tokens = self.read_tokens()

    def __repr__(self):
        head = f"Tokenizer State:\n\t" \
               f"head:({self.ln}, {self.col})\n\tcurrent_line:{self.ln}\n\tchar:{self.char}\n"
        lines = []
        start = 0
        indent = ''
        for i, tok in enumerate(self.tokens):
            if tok.type in (TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd):
                ln = self.tokens[start].pos[0]
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
                token_type = TokenType.Number
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
            elif self.char in '[](),;':
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
        if self.char == '\n':
            self.ln += 1
            self.col = 0
        return self.char

    def next_line(self):
        while self.char != '\n':
            self.next_char()
        # self.ln += 1
        self.indent = 0
        while self.next_char():
            if self.char == '\t':
                self.indent += 1
            elif self.char in '#\n':
                return self.next_line()
            elif self.script[self.idx:self.idx+4] == '    ':
                self.next_char(3)
                self.indent += 1
            elif self.char == ' ':
                continue
            else:
                break
        # self.col = 1

    def read_number(self):
        num_text = self.char
        while self.next_char():
            if self.char == '_':
                continue
            elif re.match(r'[\d.]', self.char):
                num_text += self.char
            else:
                break
        if self.char in 'fbtqphsond':
            num_text += self.char
            self.next_char()
        return num_text

    def read_string(self, quote: str) -> str:
        str_text = ""
        str_start = self.idx
        while self.next_char():
            if self.char == "\\":
                str_text += self.char + (self.next_char() or "")
                if self.char is None:
                    raise NotImplementedError("Should detect newline at line ", self.ln + 1)
                continue
            if self.char == '\n':
                break
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
        # try to get the longest operator (up to three characters)
        chars = self.peek(0, 3)
        if chars in Op:
            self.next_char(3)
            return chars
        if chars[:2] in Op and chars[1:] != "==":
            # second condition is to disambiguate these two cases: +== and *==
            self.next_char(2)
            return chars[:2]
        self.next_char()
        return chars[0]

    def peek(self, offset=1, count=1):
        i = self.idx + offset
        return self.script[i:i + count]


class AST:
    def __init__(self, toks: Tokenizer):
        self.tokens = toks.tokens
        self.idx = 0
        self.tok = self.tokens[0] if self.tokens else None
        self.indent = 0
        try:
            self.block = self.read_block(0)
        except Exception as e:
            self.block = None
            print(f"AST failed at {self.tok.pos}: {self.tok}")
            raise e

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
        statements: list[Node] = []
        table_names: set[str] = set()
        trait_names: set[str] = set()
        func_names: set[str] = set()
        while self.tok:
            stmt_nodes = self.read_statement(TokenType.NewLine, TokenType.BlockEnd)
            if stmt_nodes:
                stmt = mathological(stmt_nodes)
                statements.append(stmt)
                match stmt:
                    case TableExpr(table_name=tbl):
                        if tbl in table_names:
                            raise SyntaxErr(f'Line {stmt.line}: Duplicate table name "{tbl}"')
                        table_names.add(tbl)
                    case TraitExpr(trait_name=trait):
                        if trait in trait_names:
                            raise SyntaxErr(f'Line {stmt.line}: Duplicate trait name "{trait}"')
                        trait_names.add(trait)
                    case FunctionExpr(fn_name=fn):
                        if fn in func_names:
                            raise SyntaxErr(f'Line {stmt.line}: Duplicate function name "{fn}"')
                        func_names.add(fn)

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

        return Block(statements, table_names, trait_names, func_names)

    def read_statement(self, *end_of_statement: TokenType) -> list[Node]:
        nodes: list[Node] = []
        while self.tok and self.tok.type not in end_of_statement:
            match self.tok.type:
                case TokenType.GroupEnd | TokenType.ListEnd | TokenType.FnEnd:
                    raise Env.SyntaxErr(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
                case TokenType.NewLine | TokenType.BlockEnd:
                    pass
                case TokenType.BlockStart:
                    indent = len(self.tok.source_text)
                    self.seek()
                    nodes.append(self.read_block(indent))
                    # this is a slight opportunity for optimization: ExprWtithBlock repeatedly slices the nodes in order
                    # to make the block ladder, but it would be more efficient to do it here so no slicing is necessary.
                    # if self.indent == indent - 1 and self.peek() and self.peek().type == TokenType.Else:
                    #     nodes.append(self.seek())
                    #     self.seek()
                    #     nodes.extend(self.read_statement(*end_of_statement))
                    if self.indent == indent-1 and self.peek() and self.peek().type == TokenType.Else:
                        self.seek()
                    continue
                case TokenType.GroupStart:
                    if nodes and nodes[-1].type not in (TokenType.Operator, TokenType.Command, TokenType.Keyword):
                        nodes.append(Token('&', self.tok.pos))
                    self.seek()
                    group = self.read_statement(TokenType.GroupEnd)
                    if group:
                        nodes.append(mathological(group))
                    else:
                        nodes.append(ListNode([], ListType.Tuple))
                case TokenType.ListStart:
                    if self.idx and self.peek(-1).type in \
                            (TokenType.Name, TokenType.GroupEnd, TokenType.ListEnd, TokenType.FnEnd):
                        nodes.append(Token('.', self.tok.pos))
                        list_type = ListType.Args
                    else:
                        list_type = ListType.List
                    self.seek()
                    items, params = self.read_list(TokenType.ListEnd)
                    ls = ListNode(items, list_type if params is None else params)
                    if self.peek().source_text == ':' and self.peek(2).type == TokenType.BlockStart:
                        ls.list_type = ListType.Params
                    nodes.append(ls)
                case TokenType.FnStart:
                    self.seek()
                    nodes.append(ListNode(self.read_list(TokenType.FnEnd)[0], ListType.Function))
                case TokenType.StringStart:
                    nodes.append(self.read_string())
                case TokenType.Command:
                    command = self.tok.source_text
                    if command == 'debug':
                        # Context.debug = True
                        pass
                    Cmd = expressions.get(command, CommandWithExpr)
                    pos = self.tok.pos
                    self.seek()
                    expr = self.read_statement(*end_of_statement)
                    nodes.append(Cmd(command, expr, pos[0], ''))
                    return nodes
                case TokenType.Comma | TokenType.Semicolon:
                    self.tok.type = TokenType.Operator
                    nodes.append(self.tok)
                case _ if self.tok.source_text == 'if' and nodes:
                    self.seek()
                    cond_nodes = self.read_statement(TokenType.Else, *end_of_statement)
                    if self.tok.type == TokenType.Else:
                        self.seek()
                        alt_nodes = self.read_statement(*end_of_statement)
                    else:
                        alt_nodes = []
                    return [IfElse(*(map(mathological, (nodes, cond_nodes, alt_nodes))))]
                case TokenType.Keyword:
                    key = self.tok.source_text
                    line = self.tok.pos[0]
                    Cmd = expressions[key]
                    self.seek()
                    header = self.read_statement(TokenType.BlockStart, *end_of_statement)
                    if self.tok.type != TokenType.BlockStart:
                        raise SyntaxErr(f"Line {line}: missing block after {key} statement.")
                    if header[-1].source_text == ':':
                        raise SyntaxErr(f"Line {line}: "
                                        f"Pili does not use colons for control blocks like if and for.")
                    # thought: maybe I need to pass this logic to a looping function that creates a ladder of blocks
                    # because this will only catch one else block, not else if ...
                    blk_node = self.read_statement(*end_of_statement)
                    # alt_nodes = self.read_block_ladder(*end_of_statement)
                    nodes.append(Cmd(header, blk_node, line, ''))
                    return nodes
                case _ if self.tok.source_text == ':':
                    nodes.append(self.tok)
                    if self.seek() and self.tok.type == TokenType.BlockStart:
                        continue
                    line = self.tok.pos[0]
                    expr_nodes = self.read_statement(*end_of_statement)
                    if not expr_nodes:
                        raise SyntaxErr(f"Line {nodes[-1].line}: missing block or expression after ':' operator.")
                    # determine if `nodes` represents a set of parameters, a value/tuple, or is ambiguous
                    match nodes:
                        case [ListNode(list_type=ListType.Params) as ls_node, Token()] \
                             | [*_, Token(source_text='.'), ListNode() as ls_node, Token()]:
                            nodes.append(Block([CommandWithExpr('return', expr_nodes, line, '')]))
                            ls_node.list_type = ListType.Params
                        # case [ListNode(), Token()]:
                        #     raise Env.SyntaxErr(f"Line {line}: Ambiguous statement.  Please specify whether this is an"
                        #                         f"option function definition or a key-value pair.")
                        case _:
                            nodes.extend(expr_nodes)
                    return nodes
                case _:
                    nodes.append(self.tok)
            self.seek()
        if self.tok is None:
            raise Env.SyntaxErr('EOF reached.  Expected ', end_of_statement)

        return nodes

    def read_list(self, end: TokenType) -> tuple[list[Node], list[Node] | None]:
        named_params: list[Node] | None = None  # this also acts as an indicator for list type
        white_space_tokens = TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd
        items: list[Node] = []
        current = items
        while self.tok:
            while self.tok and self.tok.type in white_space_tokens:
                self.seek()
            if self.tok is None:
                break
            if self.tok.type == end:
                return items, named_params
            statement = self.read_statement(end, TokenType.Comma, TokenType.Semicolon, *white_space_tokens)
            if statement:
                current.append(mathological(statement))
                if named_params is None and isinstance(current[-1], BindExpr):
                    named_params = []
            else:
                continue
            if self.tok:
                if self.tok.type == TokenType.Comma:
                    self.seek()
                elif self.tok.type == TokenType.Semicolon:
                    if named_params is None:
                        named_params = []
                    current = named_params
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
                    nodes.append(mathological(self.read_statement(TokenType.FnEnd)))
                case TokenType.StringEnd:
                    return StringNode(nodes)
                case _:
                    raise Env.SyntaxErr(f"Found unexpected token in string on line {line}")
        raise Env.SyntaxErr(f"Unterminated string on line {line}")

    def read_block_ladder(self, *end_of_statement):
        pass

    def __repr__(self):
        return '\n'.join(f"{expr.line}. {expr}" for expr in self.block.statements)





if __name__ == "__main__":
    script_path = "syntax_demo.pili"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    # print(repr(tokenizer))
    print('\nAST:')
    ast = AST(tokenizer)
    print(repr(ast))
