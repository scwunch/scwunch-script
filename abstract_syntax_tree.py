from state import Op
from utils import SyntaxErr, OperatorErr
from syntax import Node, TokenType, Token, Position, Operator
from lexer import Tokenizer
from interpreter import Block, TableExpr, TraitExpr, FunctionExpr, OpExpr, EmptyExpr, ParamsNode, BindExpr, IfElse, \
    EXPRMAP, CommandWithExpr, NamedExpr, ListLiteral, FunctionLiteral, FieldMatcherNode, ArgsNode, StringNode

print(f'loading {__name__}.py')

class AST:
    tokens: list[Token]
    idx: int
    tok: Token | None
    indent: int
    block: Block | None
    function_name: str | None = None
    in_function_body: str | None
    function_blocks: list[str]

    @property
    def in_function_body(self):
        return self.function_blocks[-1] if self.function_blocks else None

    def in_group_or_function_block(self, end_of_statement: tuple):
        return (self.in_function_body
                or TokenType.RightParen in end_of_statement
                or TokenType.RightBracket in end_of_statement
                or TokenType.RightBrace in end_of_statement)

    def __init__(self, toks: Tokenizer):
        self.tokens = toks.tokens
        self.idx = 0
        self.seek(0)
        self.indent = 0
        self.function_blocks = []
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
        self.function_blocks.append(self.function_name)
        self.function_name = None
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
        self.function_blocks.pop()
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
                            raise OperatorErr(f"Line {op_pos.ln}: binary operator '{op}' missing right-hand-side.")
                        terms.pop()
                        pos = t0.pos + t1.pos
                        if op.text == ':':
                            # try to insert self param in dot methods
                            match t0:
                                case OpExpr('.', [EmptyExpr(), method], pos=t0pos):
                                    if not self.in_function_body:
                                        raise SyntaxErr(f'Line {pos.ln}: dot methods not allowed '
                                                        f'outside of table/trait/function blocks.')
                                    match method:
                                        case OpExpr('[', [loc, ParamsNode(pos=pp) as params]) as t0:
                                            t0.pos = t0pos
                                            params.nodes.insert(0, BindExpr(Token(self.in_function_body, TokenType.Name, pp.pos),
                                                                            'self', pos=pp.pos))
                                        case fn_node:
                                            ppos = Position((t0pos.stop_index,)*2)
                                            t0 = OpExpr(Op['['], fn_node,
                                                        ParamsNode([BindExpr(Token(self.in_function_body, TokenType.Name, ppos.pos),
                                                                             'self', pos=ppos)],
                                                                   []), pos=t0pos)
                        terms.append(OpExpr(op, t0, t1, pos=pos))
                    elif fixity == 'postfix' and op.text[0] in '?+*' and isinstance(t1, BindExpr):
                        t1.quantifier = op.text
                        t1.pos += op_pos
                        terms.append(t1)
                    else:
                        pos = op_pos + t1.pos
                        if fixity == 'prefix':
                            op_terms = (EmptyExpr(pos.pos), t1)
                        elif fixity == 'postfix':
                            op_terms = (t1, EmptyExpr((pos.ln, pos.ch + pos.stop_index - pos.start_index)))
                        else:
                            raise AssertionError(fixity)
                        terms.append(OpExpr(op, *op_terms, pos=pos))
                else:
                    return

        def loop_nodes():
            while self.tok.type not in end_of_statement:
                if unary_state:
                    yield self.read_operand(end_of_statement)
                else:
                    # some tokens will yield both an operator and also queue up an operand
                    yield from self.read_operator()

        nodes = loop_nodes()

        for node in nodes:
            # In the unary state, we're expecting either: unary operator or an operand (any valid expression)
            # otherwise, we are in binary state expecting binary operators, or end of statement
            if unary_state:
                match node:
                    case Token(type=TokenType.Operator, text=op_text, pos=pos):
                        try:
                            op = Op[op_text]
                        except KeyError:
                            raise OperatorErr(f"Line {pos.ln}: unrecognized operator: '{op_text}'")
                        if op.prefix:
                            ops.append([op, 'prefix', pos])
                        elif op.binop and ops and ops[-1][0].postfix:
                            ops[-1][1] = 'postfix'
                            ops.append([op, 'binop', pos])
                        elif ops:
                            raise OperatorErr(f"Line {pos.ln}: expected term or prefix operator after '{ops[-1][0]}'.  "
                                              f"Instead got '{op}'.")
                        else:
                            raise OperatorErr(f"Line {pos.ln}: expected term or prefix operator.  Got '{op}'")
                    case Node():
                        terms.append(node)
                        unary_state = False
                    case _:
                        raise AssertionError
            else:
                match node:
                    case Token(text='if'):
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
                        # assert node.type in {TokenType.Operator, TokenType.LeftBracket, TokenType.LeftParen,
                        #                      TokenType.Comma, TokenType.Semicolon}
                        if (self.tok.text == ':' and self.peek().type != TokenType.BlockStart
                                and not self.in_group_or_function_block(end_of_statement)):
                            raise SyntaxErr(f'Line {self.tok.line}: invalid syntax outside function block.\n\t'
                                            f'If you meant to define a function option, the body must be on an indented'
                                            f' block.\n\tIf you meant to assign to a key, use "=" instead of ":"')
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
                            raise OperatorErr(f"Line {pos.line}: Prefix '{op}' used as binary/postfix operator.")
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
                    raise OperatorErr(f"Line {op_pos.ln}: expected operand after '{op}' @ {op_pos}")
            elif len(terms) == 0:
                return EmptyExpr(self.tok.pos.pos)
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
                raise SyntaxErr(f'Unexpected {repr(self.tok)} found at {self.tok.pos}!')
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
                return Cmd(command, *field_name, expr, pos)  # noqa
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
                if self.peek(2) and (self.peek().text == '=>' or
                                     self.peek().text == ':' and self.peek(2).type == TokenType.BlockStart):
                    node = ParamsNode(nodes, named_params, pos)
                elif named_params:
                    raise SyntaxErr(f"Line {pos.ln}: semicolon not allowed in list literals or argument list.")
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
                if command in ('table', 'trait', 'function'):
                    self.function_name = self.tok.text
                header = self.read_expression(TokenType.BlockStart, *end_of_statement)
                if self.tok.type != TokenType.BlockStart:
                    raise SyntaxErr(f"Line {pos.ln}: missing block after {Cmd} statement.")
                if self.peek(-1).source_text == ':':
                    raise SyntaxErr(pos, f"Pili does not use colons for control blocks like if and for.")
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
                # special syntax for designating arbitrary named args in parameter set
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
                nodes = self.read_list(TokenType.RightParen, semicolon_behaviour="disallow", pos=pos)
                pos += self.tok.pos
                yield FieldMatcherNode(nodes, pos)
                self.seek()
            case TokenType.LeftBracket:
                yield self.tok
                # args or params node
                pos = self.tok.pos
                nodes, named_params = self.read_list(TokenType.RightBracket, semicolon_behaviour="split", pos=pos)
                pos += self.tok.pos
                if self.peek(2) and (self.peek().text == '=>' or
                                     self.peek().text == ':' and self.peek(2).type == TokenType.BlockStart):
                    yield ParamsNode(nodes, named_params, pos)
                elif named_params:
                    raise SyntaxErr(f"Line {pos.ln}: semicolon not allowed in argument list")
                else:
                    yield ArgsNode(nodes, pos)
                self.seek()
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
        # I thought about introducing interval syntax like (1, 4] to mean 2, 3, 4 and [1, 4) to mean 1, 2, 3
        # ... but I changed my mind
        # if end in {TokenType.RightBracket, TokenType.RightParen}:
        #     delimiters += (TokenType.RightBracket, TokenType.RightParen)
        current = items = []
        named_params = []
        while self.seek().type != end:
            statement = self.read_expression(end, *delimiters)
            if statement:
                current.append(statement)
            if self.tok.type == end:
                break
            # *interval syntax*
            # if self.tok.type in {TokenType.RightBracket, TokenType.RightParen}:
            #     match items, named_params:
            #         case [OpExpr('>>', [start, end])], []:
            #             return IntervalExpr(start, end, end != TokenType.RightBracket, end == TokenType.RightBracket)
            #         case [OpExpr('>>', [start, end]), step_node], []:
            #             return IntervalExpr(start, end, end != TokenType.RightBracket, end == TokenType.RightBracket, step=step_node)
            #     raise SyntaxErr(f'Line {self.tok.line}: mismatching brackets/parens')
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
