import Env
from operators import Op
from Syntax import *
from itertools import islice

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
        self.ln, self.col = 0, 0
        self.char = '\n'
        self.next_line()
        self.in_string = []
        self.fn_lv = 0
        self.tokens = self.read_tokens()

    def __str__(self):
        head = f"Tokenizer State:\n\t" \
               f"head:({self.ln}, {self.col})\n\tcurrent_line:{self.ln}\n\tchar:{self.char}\n"
        lines = []
        start = 0
        indent = ''
        for i, tok in enumerate(self.tokens):
            if tok.type in (TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd):
                ln = self.tokens[start].pos.ln
                lines.append(f"{ln}. {indent}{' '.join(t.text for t in self.tokens[start:i])}")
                start = i+1

            if tok.type in (TokenType.BlockStart, TokenType.BlockEnd):
                indent = tok.text
        return head + "\n".join(lines)

    def read_tokens(self):
        tokens: list[Token] = []
        text: str
        token_type = None
        while self.char:
            start = self.idx
            pos = (self.ln, self.col)
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
                if text == ':':
                    if len(tokens) > 2 and tokens[-1].type == TokenType.Operator and tokens[-2].text == '.':
                        # This allows operators to be used as function names like this: `.>:`
                        tokens[-1].type = TokenType.Name
                    if (''.join(islice(
                            (t.text for t in reversed(tokens) if not re.match(r'\s', t.text)),
                            3)) == "]*,"):
                        # if the last three non-whitespace tokens are `,*]` then
                        # special case syntax to designate arbitrary kwargs
                        tokens[-2].type = TokenType.Name
                token_type = TokenType.Operator
            elif self.char in "{}":
                text = self.char
                if self.in_string:
                    if self.char == '{' and len(self.in_string) - self.fn_lv == 1:
                        self.fn_lv += 1
                    elif self.char == '}':
                        self.fn_lv -= 1
            elif self.char in '[](),;':
                text = self.char
            elif self.char == '#' and self.peek(1, 5) == 'debug':
                text = '#debug'
                self.next_char(6)
            elif self.char in "#\n":
                last_indent = self.indent
                self.next_line()
                if self.indent == last_indent:
                    text = '\n'
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

            if text == '#debug':
                pass
            if self.idx == start:
                self.next_char()
            end = self.idx
            tokens.append(Token(text, token_type, pos, start, end))
            token_type = None
            if len(self.in_string) > self.fn_lv:
                pos = self.ln, self.col
                start = self.idx  # ie, the index after the last token, which was a quote or brace
                text = self.read_string(self.in_string[-1])
                tokens.append(Token(text, TokenType.StringPart, pos, start, self.idx))
            while self.char and self.char in ' \t':
                self.next_char()
        tokens.append(Token('', TokenType.BlockEnd, (self.ln, self.col)))
        tokens.append(Token('', TokenType.EOF, (self.ln, self.col)))
        return tokens

    def next_char(self, count=1):
        """ if count>1, assumes there is no newline character in that range """
        if self.char == '\n':
            self.ln += 1
            self.col = 0
        self.idx += count
        self.col += count
        self.char = self.script[self.idx:self.idx+1]
        return self.char

    def next_line(self):
        while self.char != '\n':
            self.next_char()
        # self.ln += 1
        self.indent = 0
        while self.next_char():
            if self.char == '\t':
                self.indent += 1
            elif self.char == '#' and self.peek(1, 5) == 'debug':
                break
            elif self.char in '#\n':
                return self.next_line()
            elif self.peek(0, 4) == '    ':
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
            elif re.match(r'\d|\.', self.char):
                num_text += self.char
            else:
                break
        if self.char in 'btqphsond':
            num_text += self.char
            self.next_char()
        if self.char == 'f':
            num_text += self.char
            self.next_char()
        return num_text

    def read_string(self, quote: str) -> str:
        # str_text = ""
        str_start = self.idx
        while self.char:
            if self.char == '\n':
                break
            if self.char in ("{", quote):
                return self.script[str_start:self.idx]
            if self.char == "\\":
                self.next_char()
            self.next_char()
        raise Env.SyntaxErr(f"Unterminated string at {{ln={self.ln}, ch={self.col}}}: "
                            f"{self.script[str_start:self.idx]} ")
                            # f"{str_text}")

    def read_string_literal(self):
        backticks = 1
        while self.next_char() == "`":
            backticks += 1
        str_start = self.idx
        # str_text = "`" * backticks
        while self.char:
            if self.peek(0, backticks) == "`" * backticks:
                s = slice(str_start, self.idx)
                self.next_char(backticks)
                return self.script[s]
            # if str_text.endswith("`" * backticks):
            #     self.next_char()
            #     return str_text
            self.next_char()
        raise Env.SyntaxErr(f"Unterminated string at {{ln={self.ln + 1}, ch={self.col + 1}}}: "
                            f"{self.script[str_start:self.idx]} ")
                            # f"{str_text}")

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


# class CST:
#     """
#     Takes a list of tokens from a Tokenizer object and returns a Concrete Syntax Tree.
#     The root Node is Block self.block.  Each node has an operation, and a sequence or struct of nodes.
#     The CST simply transforms a flat list of tokens into a hierarchical tree structure with the highest level operators
#     and syntax.  Ie:
#         lines
#         ;, =, :, and ,
#         ... if ... else ... statements
#         commands
#         control blocks (if, while, etc)
#         ArrayNodes (Strings, lists, tuples, functions, etc)
#     The AST then has the following primary purposes:
#         insert implicit operators (ie function call, bind, and sub-pattern intersection)
#         hierarchicalize remaining lists of nodes into operator expressions
#         disambiguate certain cases (eg ParamsNode vs ArgsNode, Tuple vs FieldMatcher, key: value vs params: block, etc)
#     """
#     tokens: list[Token]
#     idx: int
#     tok: Token | None
#     indent: int
#     block: Block | None
#     def __init__(self, toks: Tokenizer):
#         self.tokens = toks.tokens
#         self.idx = 0
#         self.seek(0)
#         self.indent = 0
#         try:
#             self.block = self.read_block(0)
#         except Exception as e:
#             self.block = None
#             if self.tok:
#                 print(f"CST failed at {self.tok.pos}: {self.tok}")
#             else:
#                 print("CST failed at end.")
#             raise e
#
#     def peek(self, offset=1):
#         i = self.idx + offset
#         if 0 <= i < len(self.tokens):
#             return self.tokens[i]
#         else:
#             return None
#
#     def seek(self, offset=1):
#         self.idx += offset
#         if self.idx >= len(self.tokens):
#             self.tok = None
#             return None
#         self.tok = self.tokens[self.idx]
#         return self.tok
#
#     def read_block(self, indent: int) -> Block:
#         statements: list[Node] = []
#         table_names: set[str] = set()
#         trait_names: set[str] = set()
#         func_names: set[str] = set()
#         while self.tok:
#             stmt = self.read_statement(TokenType.NewLine, TokenType.BlockEnd)
#             statements.append(stmt)
#             if stmt.type is ExprType.Command:
#                 match stmt.nodes:
#                     case [Token(text='function'), Concrete(nodes=[Token(text=fn)])]:
#                         if fn in func_names:
#                             raise SyntaxErr(f'Line {stmt.line}: Duplicate function name "{fn}"')
#                         func_names.add(fn)
#                     case [Token(text='table'), Concrete(nodes=[Token(text=tbl)])]:
#                         if tbl in table_names:
#                             raise SyntaxErr(f'Line {stmt.line}: Duplicate table name "{tbl}"')
#                         table_names.add(tbl)
#                     case [Token(text='trait'), Concrete(nodes=[Token(text=trait)])]:
#                         if trait in trait_names:
#                             raise SyntaxErr(f'Line {stmt.line}: Duplicate trait name "{trait}"')
#                         trait_names.add(trait)
#             if self.tok is None:
#                 break
#             elif self.tok.type == TokenType.NewLine:
#                 self.seek()
#             elif self.tok.type == TokenType.BlockEnd:
#                 self.indent = len(self.tok.text)
#                 if self.indent < indent:
#                     break
#                 else:
#                     self.seek()
#         return Block(list(map(mathological, statements)), table_names, trait_names, func_names)
#
#     def read_statement(self, *end_of_statement: TokenType) -> Concrete:
#         nodes: list[Node] = []
#         while self.tok and self.tok.type not in end_of_statement:
#             match self.tok.type:
#                 case TokenType.RightParen | TokenType.RightBracket | TokenType.RightBrace:
#                     raise Env.SyntaxErr(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
#                 case TokenType.NewLine | TokenType.BlockEnd:
#                     pass
#                 case TokenType.BlockStart:
#                     if TokenType.RightParen in end_of_statement:
#                         self.seek()
#                         continue
#                     indent = len(self.tok.text)
#                     self.seek()
#                     nodes.append(self.read_block(indent))
#                     # this is a slight opportunity for optimization: ExprWithBlock repeatedly slices the nodes in order
#                     # to make the block ladder, but it would be more efficient to do it here so no slicing is necessary.
#                     # if self.indent == indent - 1 and self.peek() and self.peek().type == TokenType.Else:
#                     #     nodes.append(self.seek())
#                     #     self.seek()
#                     #     nodes.extend(self.read_statement(*end_of_statement))
#                     if self.indent == indent - 1 and self.peek() and self.peek().type == TokenType.Else:
#                         self.seek()
#                     continue
#                 case TokenType.Command:
#                     cmd_tok = self.tok
#                     # nodes.append(Concrete([cmd_tok, self.read_statement(*end_of_statement)],
#                     #                       ExprType.Command))
#                     command = self.tok.text
#                     Cmd = EXPRMAP.get(command, CommandWithExpr)
#                     if issubclass(Cmd, NamedExpr):
#                         field_name = [self.seek().text]
#                     else:
#                         field_name = []
#                     self.seek()
#                     expr = self.read_statement(*end_of_statement)
#                     # expr_nodes: list[Node] = [self.seek()] if Cmd in (SlotExpr, FormulaExpr, SetterExpr) else []
#                     # self.seek()
#                     # expr_nodes.extend(self.read_statement(*end_of_statement))
#                     pos = cmd_tok.pos + self.peek(-1).pos
#                     nodes.append(Cmd(command, *field_name, expr, pos))
#                     return Concrete(nodes)
#                 case TokenType.LeftParen if not nodes or nodes[-1].type == TokenType.Operator:
#                     # unary-state => normal grouping or tuple
#                     group_nodes = [self.tok]
#                     self.seek()
#                     group_nodes.append(self.read_statement(TokenType.RightParen))
#                     group_nodes.append(self.tok)
#                     nodes.append(Concrete(group_nodes, ExprType.Group))
#                 case TokenType.LeftParen | TokenType.LeftBracket | TokenType.LeftBrace:
#                     # each node of this Concrete is one of: bracket, semicolon, comma-separated-list
#                     # groups a list like this:  [ 1+1, 2+2; 3+3, 4+4 ]
#                     # => ('[' (('1' '+' '1') ('2' '+' '2')) ';' (('3' + '3') ('4' + '4')) ']')
#                     end_type, list_type = {
#                         TokenType.LeftParen:   (TokenType.RightParen, ExprType.Parens),
#                         TokenType.LeftBracket: (TokenType.RightBracket, ExprType.Brackets),
#                         TokenType.LeftBrace:   (TokenType.RightBrace, ExprType.Braces)   }\
#                         [self.tok.type]
#                     list_nodes = [self.tok]
#                     while self.tok.type != end_type:
#                         self.seek()
#                         items: list[Concrete] = self.read_list_items(end_type)
#                         list_nodes.append(Concrete(items))
#                         list_nodes.append(self.tok)
#                     nodes.append(Concrete(list_nodes, list_type))
#                 case TokenType.StringStart:
#                     nodes.append(self.read_string())
#                 # case _ if self.tok.text and self.tok.text[-1] in ":=;":
#                 #     left = Concrete(nodes)
#                 #     op_node = self.tok
#                 #     self.seek()
#                 #     right = self.read_statement(*end_of_statement)
#                 #     return Concrete([left, op_node, right], ExprType.Assignment)
#                 case TokenType.Keyword:
#                     key = self.tok.text
#                     pos = self.tok.pos
#                     Cmd = EXPRMAP[key]
#                     key_nodes = [self.tok]
#                     self.seek()
#                     header = self.read_statement(TokenType.BlockStart, *end_of_statement)
#                     if self.tok.type != TokenType.BlockStart:
#                         raise SyntaxErr(pos, "missing block after {key} statement.")
#                     if header[-1].source_text == ':':
#                         raise SyntaxErr(pos, f"Pili does not use colons for control blocks like if and for.")
#                     # thought: maybe I need to pass this logic to a looping function that creates a ladder of blocks
#                     # because this will only catch one else block, not else if ...
#                     blk_node = self.read_statement(*end_of_statement)
#                     # alt_nodes = self.read_block_ladder(*end_of_statement)
#                     pos = Position(pos.pos, pos.start_index, self.tok.pos.start_index)
#                     nodes.append(Cmd(header, blk_node, pos))
#                     return Concrete(nodes)
#                 case _ if self.tok.text == 'if' and nodes:
#                     # first, make sure to separate lhs of assignment, or other expression separated by ;
#                     for i, node in enumerate(nodes):
#                         if node.text[-1] in ":=;":
#                             left = Concrete(nodes[i+1:])
#                             nodes = nodes[:i+1]
#                             break
#                     else:
#                         left = Concrete(nodes)
#                         nodes = []
#                     self.seek()
#                     cond = self.read_statement(TokenType.Else, *end_of_statement)
#                     if self.tok.type == TokenType.Else:
#                         self.seek()
#                         alt = self.read_statement(*end_of_statement)
#                     else:
#                         alt = Concrete([])
#                     pos = Concrete([left, cond, alt]).pos
#                     nodes.append(IfElse(left, cond, alt, pos))
#                 case TokenType.Comma:
#                     self.tok.type = TokenType.Operator
#                     nodes.append(self.tok)
#                 case TokenType.Debug:
#                     pass  # #debug
#                 case _:
#                     nodes.append(self.tok)
#             self.seek()
#         if not self.tok:
#             if TokenType.NewLine in end_of_statement:
#                 return Concrete(nodes)
#             raise Env.SyntaxErr('EOF reached.  Expected ', end_of_statement)
#         return Concrete(nodes)
#
#     def read_string(self):
#         nodes = []
#         pos = self.tok.pos
#         while self.seek():
#             match self.tok.type:
#                 case TokenType.StringPart:
#                     nodes.append(self.tok)
#                 case TokenType.LeftBrace:
#                     self.seek()
#                     nodes.append(mathological(self.read_statement(TokenType.RightBrace)))
#                 case TokenType.StringEnd:
#                     return StringNode(nodes, Position(pos.pos, pos.start_index, self.tok.pos.stop_index))
#                 case _:
#                     raise Env.SyntaxErr(f"Found unexpected token in string @ {self.tok.pos}")
#         raise Env.SyntaxErr(f"Unterminated string on line @ {self.tok.pos}")
#
#     def read_list_items(self, end: TokenType, whitespace_delimits=False) -> list[Concrete]:
#         whitespace = TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd
#         delimiters = whitespace if whitespace_delimits else ()
#         delimiters += (TokenType.Comma, TokenType.Semicolon)
#         items = []
#         while self.tok:
#             if self.tok.type == end:
#                 return items
#             statement = self.read_statement(end, *delimiters)
#             if statement:
#                 items.append(statement)
#             if self.tok.type in (end, TokenType.Semicolon):
#                 return items
#             self.seek()
#
#     def read_list(self, end: TokenType, ignore_whitespace=False) -> tuple[list[Node], list[Node] | None]:
#         named_params: list[Node] | None = None  # this only gets set if read_list detects parameters
#         whitespace = TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd
#         delimiters = () if ignore_whitespace else whitespace
#         delimiters += (TokenType.Comma, TokenType.Semicolon)
#         items: list[Node] = []
#         current = items
#         while self.tok:
#             while self.tok and self.tok.type in whitespace:
#                 self.seek()
#             if self.tok is None:
#                 break
#             if self.tok.type == end:
#                 return items, named_params
#             statement = self.read_statement(end, *delimiters)
#             if statement:
#                 current.append(mathological(statement))
#                 # if named_params is None and isinstance(current[-1], BindExpr):
#                 #     named_params = []
#             else:
#                 continue
#             if self.tok:
#                 if self.tok.type == TokenType.Comma:
#                     self.seek()
#                 elif self.tok.type == TokenType.Semicolon:
#                     if named_params is not None:
#                         raise SyntaxErr(f"Line {self.tok.line}: Only one semicolon allowed in params list.")
#                     # if named_params is None:
#                     #     named_params = []
#                     current = named_params = []
#                     self.seek()
#         raise Env.SyntaxErr("Reached EOF: Expected " + repr(end))


class OldAST:
    def __init__(self, toks: Tokenizer):
        self.tokens = toks.tokens
        self.idx = 0
        self.tok = self.tokens[0] if self.tokens else None
        self.indent = 0
        try:
            self.block = self.read_block(0)
        except Exception as e:
            self.block = None
            if self.tok:
                print(f"AST failed at {self.tok.pos}: {self.tok}")
            else:
                print("AST failed at end.")
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
                case TokenType.RightParen | TokenType.RightBracket | TokenType.RightBrace:
                    raise Env.SyntaxErr(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
                case TokenType.NewLine | TokenType.BlockEnd:
                    pass
                case TokenType.BlockStart:
                    if TokenType.RightParen in end_of_statement:
                        self.seek()
                        continue
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
                case TokenType.LeftParen:
                    if nodes and nodes[-1].type not in (TokenType.Operator, TokenType.Command, TokenType.Keyword):
                        nodes.append(Token('&', self.tok.pos))
                        self.seek()
                        items, params = self.read_list(TokenType.RightParen, ignore_whitespace=True)
                        if params:
                            raise SyntaxErr(f"Line {self.tok.line}: "
                                            f"Sorry, can't handle semicolons (;) in tuples/groups yet 🙁")
                        nodes.append(ListNode(items, ListType.FieldMatcher))
                    else:
                        self.seek()
                        group = self.read_statement(TokenType.RightParen)
                        if group:
                            nodes.append(mathological(group))
                        else:
                            nodes.append(ListNode([], ListType.Tuple))
                case TokenType.LeftBracket:
                    if self.idx and self.peek(-1).type in \
                            (TokenType.Name, TokenType.RightParen, TokenType.RightBracket, TokenType.RightBrace,
                             TokenType.StringEnd, TokenType.StringLiteral, TokenType.Number, TokenType.Singleton):
                        nodes.append(Token('.', self.tok.pos))
                        # TODO: eventually I would really like to replace this with another operator, or no operator
                        #  at all and catch it in mathological instead
                        list_type = ListType.Args
                    else:
                        list_type = ListType.List
                    self.seek()
                    items, params = self.read_list(TokenType.RightBracket)
                    # conditions for interpreting as params:
                    if params is not None or self.peek().source_text in (':', '=>'):
                        ls = ParamsNode(items, params or [])
                        if nodes and (last := nodes[-1]).type == TokenType.Operator \
                                 and last.source_text != '.' and self.peek().source_text != '=>':  # noqa
                            # >[params]: ...
                            last.type = TokenType.Name
                            nodes.append(Token('.', last.pos))
                    else:
                        ls = ListNode(items, list_type)
                    nodes.append(ls)
                case TokenType.LeftBrace:
                    self.seek()
                    nodes.append(ListNode(self.read_list(TokenType.RightBrace)[0], ListType.Function))
                case TokenType.StringStart:
                    nodes.append(self.read_string())
                case TokenType.Command:
                    command = self.tok.source_text
                    if command == 'debug':
                        # Context.debug = True
                        pass
                    Cmd = EXPRMAP.get(command, CommandWithExpr)
                    pos = self.tok.pos
                    expr_nodes: list[Node] = [self.seek()] if Cmd in (SlotExpr, FormulaExpr, SetterExpr) else []
                    self.seek()
                    expr_nodes.extend(self.read_statement(*end_of_statement))
                    nodes.append(Cmd(command, expr_nodes, pos[0], ''))
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
                    Cmd = EXPRMAP[key]
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
                case _ if self.tok.source_text == ':':  # Token(source_text=':'):
                    nodes.append(self.tok)
                    if self.seek() and self.tok.type == TokenType.BlockStart:
                        continue
                    line = self.tok.pos[0]
                    expr_nodes = self.read_statement(*end_of_statement)
                    if not expr_nodes:
                        raise SyntaxErr(f"Line {nodes[-1].line}: missing block or expression after ':' operator.")
                    # determine if `nodes` represents a set of parameters, a value/tuple, or is ambiguous
                    if len(nodes) > 1 and isinstance(nodes[-2], ParamsNode):
                        nodes.append(Block([CommandWithExpr('return', expr_nodes, line, '')]))
                    else:
                        nodes.extend(expr_nodes)
                    return nodes
                case TokenType.Debug:
                    pass  # #debug
                case _:
                    nodes.append(self.tok)
            self.seek()
        if self.tok is None:
            raise Env.SyntaxErr('EOF reached.  Expected ', end_of_statement)
        return nodes

    def read_list(self, end: TokenType, ignore_whitespace=False) -> tuple[list[Node], list[Node] | None]:
        named_params: list[Node] | None = None  # this only gets set if read_list detects parameters
        whitespace = TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd
        delimiters = () if ignore_whitespace else whitespace
        delimiters += (TokenType.Comma, TokenType.Semicolon)
        items: list[Node] = []
        current = items
        while self.tok:
            while self.tok and self.tok.type in whitespace:
                self.seek()
            if self.tok is None:
                break
            if self.tok.type == end:
                return items, named_params
            statement = self.read_statement(end, *delimiters)
            if statement:
                current.append(mathological(statement))
                # if named_params is None and isinstance(current[-1], BindExpr):
                #     named_params = []
            else:
                continue
            if self.tok:
                if self.tok.type == TokenType.Comma:
                    self.seek()
                elif self.tok.type == TokenType.Semicolon:
                    if named_params is not None:
                        raise SyntaxErr(f"Line {self.tok.line}: Only one semicolon allowed in params list.")
                    # if named_params is None:
                    #     named_params = []
                    current = named_params = []
                    self.seek()
        raise Env.SyntaxErr("Reached EOF: Expected " + repr(end))

    def read_string(self) -> StringNode:
        nodes = []
        line = self.tok.pos[0]
        while self.seek():
            match self.tok.type:
                case TokenType.StringPart:
                    nodes.append(self.tok)
                case TokenType.LeftBrace:
                    self.seek()
                    nodes.append(mathological(self.read_statement(TokenType.RightBrace)))
                case TokenType.StringEnd:
                    return StringNode(nodes)
                case _:
                    raise Env.SyntaxErr(f"Found unexpected token in string on line {line}")
        raise Env.SyntaxErr(f"Unterminated string on line {line}")

    def read_block_ladder(self, *end_of_statement):
        pass

    def __repr__(self):
        return '\n'.join(f"{expr.line}. {expr}" for expr in self.block.statements)


class AST:
    tokens: list[Token]
    idx: int
    tok: Token | None
    indent: int
    block: Block | None
    trait_name: str | None = None
    in_trait_block: str | None
    trait_blocks: list[str]

    @property
    def in_trait_block(self):
        return self.trait_blocks[-1] if self.trait_blocks else None

    def __init__(self, toks: Tokenizer):
        self.tokens = toks.tokens
        self.idx = 0
        self.seek(0)
        self.indent = 0
        self.trait_blocks = []
        try:
            self.block = self.read_block(0)
        except Exception as e:
            self.block = None
            if self.tok:
                print(f"CST failed at {self.tok.pos}: {self.tok}")
            else:
                print("CST failed at end.")
            raise e

    def peek(self, offset=1):
        i = self.idx + offset
        if 0 <= i < len(self.tokens):
            return self.tokens[i]
        else:
            return None

    def seek(self, offset=1):
        self.idx += offset
        if self.idx >= len(self.tokens):
            self.tok = None
            return None
        self.tok = self.tokens[self.idx]
        return self.tok

    def read_block(self, indent: int) -> Block:
        self.trait_blocks.append(self.trait_name)
        self.trait_name = None
        statements: list[Node] = []
        table_names: set[str] = set()
        trait_names: set[str] = set()
        func_names: set[str] = set()
        pos = self.tok.pos
        while self.tok:
            stmt = self.read_expression(TokenType.NewLine, TokenType.BlockEnd, TokenType.EOF)
            statements.append(stmt)
            match stmt:
                case TableExpr(table_name=tbl):
                    if tbl in table_names:
                        raise SyntaxErr(f'Line {stmt.line}: Duplicate table name "{tbl}"')
                    table_names.add(tbl)
                case TraitExpr(fn_name=trait):
                    if trait in trait_names:
                        raise SyntaxErr(f'Line {stmt.line}: Duplicate trait name "{trait}"')
                    trait_names.add(trait)
                case FunctionExpr(fn_name=fn):
                    if fn in func_names:
                        raise SyntaxErr(f'Line {stmt.line}: Duplicate function name "{fn}"')
                    func_names.add(fn)
            if self.tok.type == TokenType.EOF:
                break
            elif self.tok.type == TokenType.NewLine:
                self.seek()
            elif self.tok.type == TokenType.BlockEnd:
                self.indent = len(self.tok.text)
                if self.indent < indent:
                    break
                else:
                    self.seek()
        self.trait_blocks.pop()
        pos += self.tok.pos
        return Block(statements, table_names, trait_names, func_names, pos)

    def read_expression(self, *end_of_statement: TokenType) -> Node:
        ops: list[list[Operator | str | Position]] = []
        # each element of the operator stack is a list of [Operator, str fixity, Position]
        # fixity is one of: prefix, binop, postfix
        terms: list[Node] = []
        unary_state = True

        def reduce(prec: int = -1):
            nonlocal terms, ops
            while ops:
                op, fixity, op_pos = ops[-1]
                op_prec = getattr(op, fixity)
                assert op_prec is not None
                if op_prec and (op_prec > prec or op.associativity == 'left' and op_prec == prec):
                    ops.pop()
                    t1 = terms.pop()
                    t0 = terms[-1] if terms else None
                    if isinstance(t0, OpExpr) and t0.op == op and op.chainable:
                        t0.terms += t1,
                    elif fixity == 'binop':
                        if not terms:
                            raise OperatorErr(f"Line {op_pos.ln}: binary operator {op} missing right-hand-side.")
                        terms.pop()
                        pos = t0.pos + t1.pos
                        terms.append(OpExpr(op, t0, t1, pos=pos))
                    elif fixity == 'postfix' and isinstance(t1, BindExpr):
                        t1.quantifier = op.text
                        t1.pos += op_pos
                        terms.append(t1)
                    else:
                        pos = op_pos + t1.pos
                        terms.append(OpExpr(op, t1, pos=pos))
                else:
                    return

        def loop_nodes():
            while self.tok.type not in end_of_statement:
                if unary_state:
                    yield self.read_operand(end_of_statement)
                else:
                    # some tokens will yield both an operator and also queue up an operand
                    yield from self.read_operator()
            # yield None

        nodes = loop_nodes()

        for node in nodes:
            # In the unary state, we're expecting either a unary operator or an operand or grouping parenthesis (or others).
            # otherwise, we are in binary state expecting binary operators, or close parenthesis (or open paren).
            if unary_state:
                match node:
                    case Token(type=TokenType.Operator, text=op_text, pos=pos):
                        try:
                            op = Op[op_text]
                        except KeyError:
                            raise OperatorErr(f"Line {pos.ln}: unrecognized operator: {op_text}")
                        if op.prefix:
                            ops.append([op, 'prefix', pos])
                        elif op.binop and ops and ops[-1][0].postfix:
                            ops[-1][1] = 'postfix'
                            ops.append([op, 'binop', pos])
                        elif ops:
                            raise OperatorErr(f"Line {pos.ln}: expected term or prefix operator after {ops[-1][0]}.  "
                                              f"Instead got {op}.")
                        else:
                            raise OperatorErr(f"Line {pos.ln}: expected term or prefix operator.  Got {op}")
                    case Node():
                        terms.append(node)
                        unary_state = False
                    case _:
                        raise AssertionError
            else:
                match node:
                    case Token(text='if', pos=pos):
                        reduce(2)
                        self.seek()
                        cond = self.read_expression(TokenType.Else)
                        assert self.tok.type is TokenType.Else
                        self.seek()
                        alt = self.read_expression(*end_of_statement)
                        # I guess the term stack could be 1 here, or it could be more with = on the ops stack
                        terms.append(IfElse(terms.pop(), cond, alt))
                    case Token(TokenType.Name, text=name, pos=pos):
                        reduce(3)
                        pos = terms[-1].pos + pos
                        terms.append(BindExpr(terms.pop(), name, pos=pos))
                    case Token(text=op_text, pos=pos):
                        # assert node.type in {TokenType.Operator, TokenType.LeftBracket, TokenType.LeftParen, TokenType.Comma, TokenType.Semicolon}
                        try:
                            op = Op[op_text]
                        except KeyError:
                            raise OperatorErr(f"Line {pos.ln}: '{op_text}' is not an operator.")
                        if op.binop:
                            reduce(op.binop)
                            ops.append([op, 'binop', pos])
                            unary_state = True
                        elif op.postfix:
                            reduce(op.postfix)
                            ops.append([op, 'postfix', pos])
                        else:
                            raise OperatorErr(f"Line {pos.line}: Prefix {op} used as binary/postfix operator.")
                    case None:
                        raise AssertionError  # end of statement
        if unary_state:
            # expression still looking for another operand
            if ops and ops[-1][1] == 'binop':
                op, fixity, op_pos = ops.pop()
                # try reinterpreting the last binop as a postfix
                if op.postfix:
                    reduce(op.postfix)
                    ops.append([op, 'postfix', op_pos])
                else:
                    raise OperatorErr(f"Line {op_pos.ln}: expected operand after {ops[-1]}")
            elif len(terms) == 0:
                return EmptyExpr(Position(self.tok.pos.pos))
            else:
                raise OperatorErr(f"Line {terms[-1].line}: expected operand.")
        reduce()
        # if len(terms) == 0 == len(ops):
        #     return EmptyExpr(Position(self.tok.pos.pos))
        assert len(terms) == 1 and len(ops) == 0
        expr = terms[0]
        return expr

    def read_operand(self, end_of_statement: tuple[TokenType, ...]) -> Node | None:
        if self.tok.type in end_of_statement:
            return None
        while self.tok.type in {TokenType.NewLine, TokenType.BlockEnd}:
            self.seek()
        pos = self.tok.pos
        match self.tok.type:
            case TokenType.Name | TokenType.StringLiteral | TokenType.Number | TokenType.Singleton:
                tok = self.tok
                self.seek()
                return tok
            case TokenType.RightParen | TokenType.RightBracket | TokenType.RightBrace:
                raise Env.SyntaxErr(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
            case TokenType.BlockStart:
                if TokenType.NewLine in end_of_statement:
                    indent = len(self.tok.text)
                    self.seek()
                    blk_node = self.read_block(indent)
                    return blk_node
                else:
                    self.seek()
                    return self.read_operand(end_of_statement)
                # if TokenType.RightParen in end_of_statement:
                #     raise SyntaxErr('I guess this should have been avoided somehow.  '
                #                     'Maybe skipping over tokens like this at the beginning of this function.')
                #     self.seek()
                # indent = len(self.tok.text)
                # self.seek()
                # return self.read_block(indent)
                # if self.indent == indent - 1 and self.peek() and self.peek().type == TokenType.Else:
                #     raise SyntaxErr("Umm, should I treat the elses as operators?")
                #     self.seek()
            case TokenType.Command:
                command = self.tok.text
                Cmd = EXPRMAP.get(command, CommandWithExpr)
                if issubclass(Cmd, NamedExpr):
                    field_name = self.seek().text,
                else:
                    field_name = ()
                self.seek()
                expr = self.read_expression(*end_of_statement)
                # expr_nodes: list[Node] = [self.seek()] if Cmd in (SlotExpr, FormulaExpr, SetterExpr) else []
                # self.seek()
                # expr_nodes.extend(self.read_statement(*end_of_statement))
                pos += expr.pos
                return Cmd(command, *field_name, expr, pos)
            case TokenType.LeftParen:
                self.seek()
                node = self.read_expression(TokenType.RightParen)
                assert self.tok.type == TokenType.RightParen
                node.pos = pos + self.tok.pos
                self.seek()
                return node
            case TokenType.LeftBracket:
                nodes, named_params = self.read_list(TokenType.RightBracket, semicolon_behaviour="split", pos=pos)
                pos += self.tok.pos
                if self.peek() and self.peek().source_text in (':', '=>'):
                    node = ParamsNode(nodes, named_params, pos)
                elif named_params:
                    raise SyntaxErr(f"Line {pos.ln}: semicolon not allowed in list literals.")
                else:
                    node = ListLiteral(nodes, pos)
                self.seek()
                return node
            case TokenType.LeftBrace:
                nodes = self.read_list(TokenType.RightBrace, whitespace_delimits=True,
                                       semicolon_behaviour="comma", pos=pos)
                pos += self.tok.pos
                node = FunctionLiteral(nodes, pos)
                self.seek()
                return node
            case TokenType.StringStart:
                str_node = self.read_string()
                self.seek()
                return str_node
            case TokenType.Keyword:
                command = self.tok.text
                Cmd = EXPRMAP[command]
                self.seek()
                if command in ('table', 'trait'):
                    self.trait_name = self.tok.text
                header = self.read_expression(TokenType.BlockStart, *end_of_statement)
                if self.tok.type != TokenType.BlockStart:
                    raise SyntaxErr(f"Line {pos.ln}: missing block after {Cmd} statement.")
                if self.peek(-1).source_text == ':':
                    raise SyntaxErr(pos, f"Pili does not use colons for control blocks like if and for.")
                # thought: maybe I need to pass this logic to a looping function that creates a ladder of blocks
                # because this will only catch one else block, not else if ...
                # alt_nodes = self.read_block_ladder(*end_of_statement)
                blk_nodes = self.read_block_ladder()
                pos += self.tok.pos
                return Cmd(header, blk_nodes, pos)
            case TokenType.Debug:
                self.seek()
                return self.read_operand(end_of_statement)
            case TokenType.Operator if self.peek() and self.peek().text == '[':
                self.tok.type = TokenType.Name
                tok = self.tok
                self.seek()
                return tok
            case _ if ''.join(t.text for t in self.tokens[self.idx-1:self.idx+2]) == ',*]':
                self.tok.type = TokenType.Name
                tok = self.tok
                self.seek()
                return tok
            case _:
                tok = self.tok
                self.seek()
                return tok
                # raise OperatorErr(f"Line {self.tok.line}: Expected operand but got {self.tok}")

    def read_operator(self) -> Token:
        while self.tok.type in {TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd}:
            self.seek()
        match self.tok.type:
            case TokenType.Operator | TokenType.Comma | TokenType.Semicolon | TokenType.Else | TokenType.Name:
                yield self.tok
                self.seek()
            case TokenType.LeftParen:
                pos = self.tok.pos
                yield Token('&', TokenType.Operator, pos.pos)
                # field matcher node
                nodes = self.read_list(TokenType.RightParen, semicolon_behaviour="disallow", pos=pos)
                pos += self.tok.pos
                yield FieldMatcherNode(nodes, pos)
                self.seek()
            case TokenType.LeftBracket:
                yield self.tok
                # args or params node
                pos = self.tok.pos
                nodes, named_params = self.read_list(TokenType.RightBracket, semicolon_behaviour="split", pos=pos)
                # if self.tok.type is TokenType.Semicolon:
                #     self.seek()
                #     param_nodes = self.read_list(TokenType.RightBracket, semicolon_behaviour="end", pos=pos)
                #     if self.tok.type is TokenType.Semicolon:
                #         raise SyntaxErr(f"Line {self.tok.line}: only zero or one semicolon allowed in params/args list.")
                # else:
                #     param_nodes = []
                pos += self.tok.pos
                if self.peek().source_text in (':', '=>'):
                    if self.in_trait_block and self.peek().text == ':':
                        nodes.insert(0, BindExpr(Token(self.in_trait_block, TokenType.Name, pos.pos),
                                                 'self', pos=pos.pos))
                    yield ParamsNode(nodes, named_params, pos)
                elif named_params:
                    raise SyntaxErr(f"Line {pos.ln}: semicolon not allowed in argument list")
                else:
                    yield ArgsNode(nodes, pos)
                self.seek()
            # switched this out in favor of BindExpr again, it's a little easier to deal with
            # case TokenType.Name:
            #     yield Token('@', TokenType.Operator, self.tok.pos.pos)
            #     yield self.tok
            #     self.seek()
            case _ if self.tok.text == 'if':
                yield self.tok
            case TokenType.Debug:
                self.seek()
                yield from self.read_operator()
            # case _ if self.tok.type in end_of_statement:
            #     yield None
            case _:
                raise OperatorErr(f"Line {self.tok.line}: Expected operator but got {self.tok}")

    def read_list(self, end: TokenType, whitespace_delimits=False, semicolon_behaviour: str = 'comma',
                  pos: Position = None) -> list[Node] | tuple[list[Node], list[Node]]:
        whitespace = TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd
        delimiters = whitespace if whitespace_delimits else ()
        delimiters += (TokenType.Comma, TokenType.Semicolon)
        current = items = []
        named_params = []
        while self.seek().type != end:
            statement = self.read_expression(end, *delimiters)
            if statement:
                current.append(statement)
            if self.tok.type == end:
                break
            if self.tok.type == TokenType.Semicolon:
                if semicolon_behaviour == 'end':
                    return items
                if semicolon_behaviour == 'disallow':
                    raise SyntaxErr(f"Line {self.tok.line}: semicolons not allowed in this context.")
                if semicolon_behaviour == 'split':
                    if named_params:
                        raise SyntaxErr(f"Line {self.tok.line}: only zero or one semicolon allowed in params list.")
                    current = named_params
        assert self.tok.type == end
        if semicolon_behaviour == 'split':
            return items, named_params
        return items
        # raise SyntaxErr(f"Line: {pos.ln}: Reached end of file.  Expected {end}")

    def read_string(self):
        nodes = []
        pos = self.tok.pos
        while self.seek():
            match self.tok.type:
                case TokenType.StringPart:
                    nodes.append(self.tok)
                case TokenType.LeftBrace:
                    self.seek()
                    nodes.append(self.read_expression(TokenType.RightBrace))
                case TokenType.StringEnd:
                    return StringNode(nodes, Position(pos.pos, pos.start_index, self.tok.pos.stop_index))
                case _:
                    raise SyntaxErr(f"Found unexpected token in string @ {self.tok.pos}")
        raise SyntaxErr(f"Unterminated string starting @ {pos}")

    def read_block_ladder(self) -> list[Node]:
        assert self.tok.type is TokenType.BlockStart
        nodes = []
        while 1:
            blk_node = self.read_operand((TokenType.NewLine,))
            nodes.append(blk_node)
            assert self.tok.type == TokenType.BlockEnd
            if self.peek().type != TokenType.Else:
                break
            if self.seek().text == 'else':
                self.seek()
                assert self.tok.type is TokenType.BlockStart
                continue
            if self.tok.text == 'elif':
                self.seek()
                expr = self.read_expression(TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd)
                nodes.append(expr)
                if self.tok.type != TokenType.BlockStart:
                    break
        return nodes



if __name__ == "__main__":
    script_path = "syntax_demo.pili"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    # print(repr(tokenizer))
    print('\nAST:')
    ast = AST(tokenizer)
    print(repr(ast))
