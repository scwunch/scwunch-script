import tables
from stub import *
import math
import importlib
import re
from enum import Enum, EnumMeta
from Env import *
from operators import Operator
from tables import *
from patterns import Inst, virtualmachine, VM

print(f"loading module: {__name__} ...")

op_char_patt = r'[:.<>?/~!@$%^&*+=|-]'  # Note: , (comma) and ; (semicolon) are excluded because of special treatment

def contains(cls, item):
    if isinstance(item, cls):
        return item.name in cls._member_map_
    else:
        return item in cls._value2member_map_


EnumMeta.__contains__ = contains


class TokenType(Enum):
    Unknown = '?'
    StringLiteral = 'string'
    StringStart = 'string_start'
    StringPart = 'string_part'
    StringEnd = 'string_end'
    Number = 'number'
    Singleton = 'singleton'
    Operator = 'operator'
    Command = 'command'  # return, break, continue, ...
    Keyword = 'keyword'  # in, with, ...
    Else = 'else'
    Name = 'name'
    GroupStart = '('
    GroupEnd = ')'
    ListStart = '['
    ListEnd = ']'
    FnStart = "{"
    FnEnd = "}"
    Comma = ','
    Semicolon = ';'
    NewLine = '\n'
    BlockStart = '\t'
    BlockEnd = '\t\n'
    Debug = "#debug"


class Commands(Enum):
    Print = 'print'
    # If = 'if'
    # Else = 'else'
    # For = 'for'
    # While = 'while'
    Local = 'local'
    Var = 'var'
    Return = 'return'
    Break = 'break'
    Continue = 'continue'
    Exit = 'exit'
    Debug = 'debug'
    Else = 'else'
    Import = 'import'
    Inherit = 'inherit'
    Label = 'label'
    Function = 'function'
    Table = 'table'
    # Slice = 'slice'
    Trait = 'trait'
    Slot = 'slot'
    Formula = 'formula'
    Opt = 'opt'
    Setter = 'setter'

class OperatorWord(Enum):
    In = 'in'
    And = 'and'
    Or = 'or'
    Is = 'is'
    Not = 'not'
    Of = 'of'
    # If = 'if'
    Has = 'has'
    # Else = 'else'

class Singletons(Enum):
    blank = 'blank'
    true = 'true'
    false = 'false'
    inf = 'inf'

class KeyWords(Enum):
    # In = 'in'
    # And = 'and'
    # Or = 'or'
    # Is = 'is'
    # Not = 'not'
    # Of = 'of'
    If = 'if'
    # Else = 'else'  # I made else into a command because it's easier to parse that way
    For = 'for'
    While = 'while'
    Try = 'try'
    Except = 'except'

class Stmt(Enum):
    Empty = 'EmptyExpr'
    Cmd = 'Command'
    Expr = 'Expression'
    IfElse = 'IfElse'
    Asgmt = 'Assignment'


def token_mapper(item: str) -> TokenType:
    return TokenType._value2member_map_.get(item, TokenType.Unknown)
def command_mapper(item: str) -> Commands:
    return Commands._value2member_map_.get(item)
def singleton_mapper(item: str) -> Singletons:
    return Singletons._value2member_map_.get(item, None)
singletons = {'blank': None, 'false': False, 'true': True, 'inf': math.inf}
def keyword_mapper(item: str) -> KeyWords:
    return KeyWords._value2member_map_.get(item, None)


class Node:
    pos: tuple[int, int]
    type = TokenType.Unknown
    source_text: str

    def getline(self):
        return getattr(self, '_line', self.pos[0])

    def setline(self, line: int):
        self._line = line

    def evaluate(self):
        return eval_node(self)

    line = property(getline, setline)


class Token(Node):
    def __init__(self, text: str, pos: tuple[int, int] = (-1, -1), type: TokenType = None):
        self.pos = pos[0], pos[1]
        self.type = type
        self.source_text = text

        if not type:
            if text in ('else', 'elif'):
                self.type = TokenType.Else
            elif re.fullmatch(r'-?\d+(\.\d*)?d?', text):
                self.type = TokenType.Number
            elif re.match(op_char_patt, text) or text in OperatorWord:
                self.type = TokenType.Operator
            elif text in Commands:
                self.type = TokenType.Command
            elif text.lower() in Singletons:
                self.source_text = text.lower()
                self.type = TokenType.Singleton
            elif text in KeyWords:
                self.type = TokenType.Keyword
            elif re.fullmatch(r'\w+', text):
                self.type = TokenType.Name
            elif text.startswith('\t'):
                self.type = TokenType.BlockStart
            else:
                self.type = token_mapper(text)

    def __str__(self):
        return self.source_text

    def __repr__(self):
        return f"<{self.source_text}:{self.type.name}>"


class ArrayNode(Node):
    nodes: list[Node]
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        for n in nodes:
            if not isinstance(n, Node):
                pass
        for n in nodes:
            if n.pos != (0, 0) and n.pos != (-1, -1):
                self.pos = n.pos
                break
        else:
            self.pos = (-1, -1)

    @property
    def source_text(self):
        return ' '.join(n.source_text for n in self.nodes)


# class Statement(ArrayNode):
#     stmt_type: Stmt
#     data: dict[str]
#     def __init__(self, type_: Stmt = None, pos=(-1, -1), **data):
#         self.pos = pos
#         self.stmt_type = type_
#         assert type_
#         self.data = data
#
#     def expressionize(self):  # -> Expression:
#         match self.stmt_type:
#             case Stmt.Empty:
#                 return EmptyExpr(self.line, '')
#             case Stmt.Cmd:
#                 Cmd = expressions.get(self.data['command'], CommandWithExpr)
#                 return Cmd(key_word, self.data['nodes'], line, src)
#             case Stmt.Expr:
#                 pass
#             case Stmt.IfElse:
#                 pass
#             case Stmt.Asgmt:
#                 pass
#         raise NotImplementedError(f'Unrecognized/Not implemented statement type: {self}')
#
#     match nodes:
#         case []:
#             return EmptyExpr()
#         case [Token(type=TokenType.Command, source_text=key_word), *other_nodes]:
#             return expressions.get(key_word, CommandWithExpr)(key_word, other_nodes, line, src)
#             # return Command(cmd, other_nodes, line, src)
#         case [node]:
#             return node  # SingleNode(node, line, src)
#         case [Token(type=TokenType.Keyword, source_text=key_word), *_]:
#             return expressions[key_word](nodes, line, src)
#         case _:
#             return mathological(nodes, line, src)
#             # return Mathological(nodes, line, src)

class StringNode(ArrayNode):
    def __repr__(self):
        res = ''.join(str(n) if n.type == TokenType.StringPart else f"{{{repr(n)}}}" for n in self.nodes)
        return f'"{res}"'

class Expression(Node):
    def __init__(self, line, source):
        self.pos = (line, -1)
        self.line = line
        self.source_text = source

    def evaluate(self):
        raise NotImplementedError()

    # def __repr__(self):
    #     if self.line:
    #         return f"Line {self.line}: {self.source_text}"
    #     return self.source_text

class Block(ArrayNode):
    """
    a container for executables (statements, e
    representing the lines of code to put into a function
    """
    empty = frozenset()
    statements: list[Node]
    table_names: set[str]
    trait_names: set[str]
    function_names: set[str]

    def __init__(self, nodes: list[Node],
                 table_names: set[str] = empty, trait_names: set[str] = empty, func_names: set[str] = empty):
        super().__init__(nodes)
        self.statements = nodes
        self.table_names = table_names
        self.trait_names = trait_names
        self. function_names = func_names

    def evaluate(self):
        # raise NotImplementedError("Use Block.execute instead.")
        self.execute()
        return BuiltIns['blank']

    def execute(self):
        line = Context.line
        for expr in self.statements:
            Context.line = expr.line
            expr.evaluate()
            if Context.env.return_value:
                break
            if Context.break_loop or Context.continue_:
                break
        Context.line = line

    def __repr__(self):
        if not self.statements:
            return 'Block[empty]'
        elif len(self.statements) == 1:
            return f"Block[{repr(self.statements[0])}"
        else:
            return f"Block[{len(self.statements)} statements]"


class ListType(Enum):
    List = '[list]'
    Tuple = '(tuple)'
    Function = "{function}"
    Args = '[args]'
    Params = '[params]'

class ListNode(ArrayNode):
    """
    :param list_type: [list], (tuple), {function}, [args]
    """
    items: list[Node]
    list_type: ListType
    def __init__(self, items: list[Node], list_type: ListType):
        self.items = items
        self.list_type = list_type
        super().__init__(items)

    def evaluate(self):
        items = (n.evaluate() for n in self.nodes)
        match self.list_type:
            case ListType.List:
                return py_value(list(items))
            case ListType.Tuple:
                return py_value(tuple(items))
            case ListType.Args:
                named_args = {}
                flags = set()
                def generate_args(nodes: list[Node]):
                    for node in nodes:
                        match node:
                            case OpExpr(op=Operator(text='='),
                                        terms=(Token(type=TokenType.Name, source_text=name), val_node)):
                                named_args[name] = val_node.evaluate()
                            case OpExpr(op=Operator(text='!'),
                                        terms=(Token(type=TokenType.Name, source_text=name),)):
                                flags.add(name)
                            case _:
                                yield node.evaluate()
                return Args(*generate_args(self.nodes), flags=flags, named_arguments=named_args)
            case ListType.Function:
                print(f'WARNING: Line {self.line}: Function literal not yet implemented.  Producing empty function.')
                return Function()
                raise NotImplementedError(f'Line {self.line}: Function literal not yet implemented.')
        raise NotImplementedError(f'Line {self.line}: ListNode<{self.list_type}> not yet implemented.')

    def __repr__(self):
        res = map(str, self.nodes)
        return f"{self.list_type.name}[{', '.join(res)}]"

class ParamsNode(ArrayNode):
    list_type = ListType.Params
    named_params: list[Node]
    any_kwargs: bool = False
    def __init__(self, items: list[Node], named_params: list[Node]):
        self.items = items
        if named_params and named_params[-1].source_text == '*':
            named_params.pop()
            self.any_kwargs = True
        self.named_params = named_params
        # for params in (items, named_params):
        #     for i in range(len(params)):
        #         match params[i]:
        #             case Token(type=TokenType.Name, source_text=name, pos=pos):
        #                 params[i] = BindExpr(Token("any", pos), name)
        #             case OpExpr(op=Operator(text="="),
        #                         terms=(Token(type=TokenType.Name, source_text=name, pos=pos), default)) as pexpr:
        #                 pexpr.terms = (BindExpr(Token("any", pos), name), default)

        super().__init__(items)

    def evaluate(self):
        # items = (n.evaluate() for n in self.nodes)
        # TODO: enable param flags and named parameters
        def gen_params(nodes) -> tuple[str, Parameter]:
            for node in nodes:
                match node:
                    case BindExpr(name=name):
                        yield name, node.evaluate()
                    case Token(type=TokenType.Name, source_text=name):
                        yield name, Parameter(AnyMatcher(), name)
                    case OpExpr(op=Operator(text="="),
                                terms=(BindExpr(name=name) | Token(type=TokenType.Name, source_text=name) as p_node,
                                       default)):
                        param: Parameter
                        if isinstance(p_node, Token):
                            param = Parameter(AnyMatcher(), name)
                        else:
                            param = p_node.evaluate()
                        param.default = default.evaluate()
                        yield name, param
                    case _:
                        param = node.evaluate()
                        if not isinstance(param, Parameter):
                            param = Parameter(param)
                            # raise SyntaxErr(f"Line {Context.line}: Can't parse parameter {node}")
                        yield param.binding, param

        # named_params = {}
        # for node in self.named_params:
        #     match node:
        #         case BindExpr(name=name):
        #             named_params[name] = node.evaluate()
        #         case OpExpr(op=Operator(text="="), terms=(BindExpr(name=name) as p_node, default)):
        #             param: Parameter = p_node.evaluate()
        #             param.default = default.evaluate()
        #             named_params[name] = param
        #         case _:
        #             raise SyntaxErr(f"Line {Context.line}: Can't parse parameter {node}")

        return ParamSet(*(p[1] for p in gen_params(self.items)),
                        named_params=dict(gen_params(self.named_params)),
                        kwargs=self.any_kwargs)

    def __repr__(self):
        res = map(str, self.nodes)
        return f"Params[{', '.join(res)}]"

def expressionize(nodes: list[Node]):
    return mathological(nodes)
    src = " ".join(n.source_text for n in nodes)
    line = next((n.pos[0] for n in nodes if n.pos not in [(0, 0), (-1, -1)]),
                -1)

    match nodes:
        case []:
            return EmptyExpr()
        case [Token(type=TokenType.Command, source_text=key_word), *other_nodes]:
            return expressions.get(key_word, CommandWithExpr)(key_word, other_nodes, line, src)
            # return Command(cmd, other_nodes, line, src)
        case [node]:
            return node  # SingleNode(node, line, src)
        case [Token(type=TokenType.Keyword, source_text=key_word), *_]:
            return expressions[key_word](nodes, line, src)
        case _:
            return mathological(nodes, line, src)
            # return Mathological(nodes, line, src)


Context.make_expr = expressionize

class OpExpr(Expression):
    op: Operator
    terms: tuple[Node, ...]
    def __init__(self, op: Operator, *terms: Node, line: int = -1, src: str = ''):
        assert terms
        self.op = op
        self.terms = terms
        super().__init__(line, src)

    def evaluate(self):
        args: Args = self.op.eval_args(*self.terms)
        return self.op.fn.call(args)

    def __repr__(self):
        return f"{self.op}({', '.join(map(str, self.terms))})"


class BindExpr(Expression):
    node: Node
    name: str
    quantifier: str
    def __init__(self, node: Node, name: str, quantifier: str = ''):
        self.node = node
        self.name = name
        self.quantifier = quantifier
        super().__init__(node.line, f'{node.source_text} {name}{quantifier}')

    def __repr__(self):
        return f"Bind({self.node} {self.name}{self.quantifier})"

    def evaluate(self):
        return Parameter(patternize(self.node.evaluate()), self.name, self.quantifier)

def mathological(nodes: list[Node]) -> Node:
    match nodes:
        case []:
            return EmptyExpr(-1, "")
        case [node]:
            return node
        case Node() as node:
            raise SyntaxErr(f"Line {node.line}: {node} is not a list of nodes, it's just a node.  Should I handle this gracefully?")

    src = " ".join(n.source_text for n in nodes)
    line = next((n.pos[0] for n in nodes if n.pos not in [(0, 0), (-1, -1)]),
                -1)
    ops: list[list[Operator | str]] = []
    # second (str) element of each list in ops represents the fixity: prefix, binop, postfix
    terms: list[Node] = []
    unary_state = True

    def reduce(prec: int = -1):
        nonlocal terms, ops
        while ops:
            op, fixity = ops[-1]
            op_prec = getattr(op, fixity)
            assert op_prec is not None
            if op_prec and (op_prec > prec or op.associativity == 'left' and op_prec == prec):
                ops.pop()
                t1 = terms.pop()
                t_last = terms[-1] if terms else None
                if isinstance(t_last, OpExpr) and t_last.op == op and op.chainable:
                    t_last.terms += t1,
                elif fixity == 'binop':
                    terms.append(OpExpr(op, terms.pop(), t1, line=line))
                elif fixity == 'postfix' and isinstance(t1, BindExpr):
                    t1.quantifier = op.text
                    terms.append(t1)
                else:
                    terms.append(OpExpr(op, t1, line=line))
            else:
                return

    def loop_nodes():
        yield from nodes
        yield None

    for node in loop_nodes():
        # In the unary state, we're expecting either a unary operator or an operand or grouping parenthesis (or others).
        # otherwise, we are in binary state expecting binary operators, or close parenthesis (or open paren).
        if unary_state:
            match node:
                case Token(type=TokenType.Operator, source_text=op_text):
                    try:
                        op = Op[op_text]
                    except KeyError:
                        raise OperatorErr(f"Line {line}: unrecognized operator: {op_text}")
                    if op.prefix:
                        ops.append([op, 'prefix'])
                    elif op.binop and ops and ops[-1][0].postfix:
                        ops[-1][1] = 'postfix'
                        ops.append([op, 'binop'])
                    else:
                        raise OperatorErr(f"Line {line}: expected term or prefix operator after {ops[-1][0]}.  "
                                          f"Instead got {op}.")
                case Node():
                    terms.append(node)
                    unary_state = False
                case _:
                    if ops:
                        op, fixity = ops.pop()
                        if op.postfix:
                            reduce(op.postfix)
                            ops.append([op, 'postfix'])
                        else:
                            raise OperatorErr(f"Line {line}: expected operand after {ops[-1]}")
                    else:
                        raise OperatorErr(f"Line")
                    # if ops and ops[-1][0].postfix:
                    #     ops[-1][1] = 'postfix'
                    # elif ops:
                    #     raise OperatorErr(f"Line {line}: expected operand after {ops[-1]}")
                    # else:
                    #     raise OperatorErr(f"Line")
        else:
            match node:
                case Token(type=TokenType.Operator, source_text=op_text):
                    op = Op[op_text]
                    if op.binop:
                        reduce(op.binop)
                        ops.append([op, 'binop'])
                        unary_state = True
                    elif op.postfix:
                        reduce(op.postfix)
                        ops.append([op, 'postfix'])
                    else:
                        raise OperatorErr(f"Line {line}: Prefix {op} used as binary/postfix operator.")
                case Token(type=TokenType.Name, source_text=name):
                    reduce(3)
                    terms.append(BindExpr(terms.pop(), name))
                case None:
                    pass
                case _:
                    raise OperatorErr(f"Line {line}: Expected operator.  Got {node}")
    reduce()
    assert len(terms) == 1 and len(ops) == 0
    expr = terms[0]
    expr.line, expr.source_text = line, src
    return expr


class Mathological(Expression):
    op_idx: int
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)
        self.nodes = nodes
        op_idx = None
        min_precedence = math.inf
        if nodes[0].type == TokenType.Operator:
            pre_op = Op[nodes[0].source_text]
            if pre_op.prefix:
                min_precedence = pre_op.prefix
                op_idx = 0
        if nodes[-1].type == TokenType.Operator:
            post_op = Op[nodes[-1].source_text]
            if post_op.postfix and post_op.postfix < min_precedence:
                min_precedence = post_op.postfix
                op_idx = len(nodes) - 1

        def get_op(idx: int) -> Operator | None:
            if nodes[idx].type != TokenType.Operator:
                return None
            return Op[nodes[idx].source_text]
        prev_op = get_op(0)
        for i in range(1, len(nodes) - 1):
            op = get_op(i)
            if op and op.binop and op.binop < min_precedence + (op.associativity == 'left'):
                if op.prefix:
                    prev_op = get_op(i-1)
                    if prev_op and prev_op.binop and not prev_op.postfix:
                        continue
                if op.postfix:
                    next_op = get_op(i+1)
                    if next_op and next_op.binop and not next_op.prefix:
                        continue
                op_idx = i
                min_precedence = op.binop

        if op_idx is None:
            raise OperatorErr(f'Line {self.line}: No operator found in expression: {nodes}')
        self.op = Op[nodes[op_idx].source_text]
        self.lhs = nodes[:op_idx]
        self.rhs = nodes[op_idx+1:]

    def evaluate(self):
        # if self.op.static:
        #     return self.op.static(self.lhs, self.rhs)
        args = self.op.eval_args(self.lhs, self.rhs)
        return self.op.fn.call(*args)


class IfElse(Expression):
    do: Node
    condition: Node
    alt: Node
    def __new__(cls, do: Node, cond: Node, alt: Node):
        match do:
            case OpExpr(op=Operator(text=op_text)) as asgt if op_text == ':' or op_text.endswith('='):
                asgt.terms = asgt.terms[0], IfElse(asgt.terms[1], cond, alt)
                return asgt
            case _:
                return super().__new__(cls)

    def __init__(self, do: Node, cond: Node, alt: Node):
        # i = next((-1 - i for i, node in enumerate(reversed(other_nodes))
        #           if node.type == TokenType.Else),
        #          None)
        self.do = do
        self.condition = cond
        self.alt = alt
        super().__init__(do.line, f"{do.source_text} if {cond.source_text} else {alt.source_text}")

    def evaluate(self):
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.do.evaluate()
        else:
            return self.alt.evaluate()

    def __repr__(self):
        return f"IfElse({self.do} if {self.condition} else {self.alt})"

class ExprWithBlock(Expression):
    header: Node
    block: Block
    alt: Expression | Block | None = None

    # def __init__(self, nodes: list[Node], line: int | None, source: str):
    #     super().__init__(line, source)
    #     try:
    #         i = next(i for i, node in enumerate(nodes) if isinstance(node, Block))
    #     except StopIteration:
    #         raise SyntaxErr(f"Line {self.line}: missing block after {nodes[0].source_text} statement.")
    #     if nodes[i - 1].source_text == ':':
    #         raise SyntaxErr(f"Line ({self.line}): "
    #                         f"Pili does not use colons for control blocks like if and for.")
    #     self.header = nodes[1:i]
    #     self.block = nodes[i]  # noqa  nodes[i]: Block
    #     match nodes[i+1:]:
    #         case []:
    #             self.alt = []
    #         case [Token(source_text='else'), *other_nodes]:
    #             self.alt = other_nodes
    #         case _:
    #             raise SyntaxErr(f"Line {nodes[i+1].pos[0]}: "
    #                             f"Expected else followed by statement or block.  Got {nodes[i+1:]}")

    def __init__(self, header_nodes: list[Node], blk_nodes: list[Node], line: int, source: str):
        super().__init__(line, source)
        self.header = mathological(header_nodes)
        assert blk_nodes
        match blk_nodes:
            case [Block() as blk]:
                self.block = blk
            case [Block() as blk, Token(source_text='else'), Block() as alt]:
                self.block = blk
                self.alt = alt
            case [Block() as blk, Token(source_text='elif'), *alt_expr_nodes]:
                self.block = blk
                try:
                    i = next(i for i, node in enumerate(alt_expr_nodes) if isinstance(node, Block))
                except StopIteration:
                    raise SyntaxErr(f"Line {blk_nodes[1].line}: missing block after control statement.")
                self.alt = Conditional(alt_expr_nodes[:i], alt_expr_nodes[i:], blk_nodes[1].line, '')
            case _:
                raise NotImplementedError(f"Line {blk_nodes[0].line}: process else statement...")

    def __repr__(self):
        alt = ' else ' + repr(self.alt) if self.alt else ''
        return f"{self.__class__.__name__}({self.header} => {self.block}{alt})"


class Conditional(ExprWithBlock):
    condition: Node
    consequent: Block

    def evaluate(self):
        condition = self.header.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.block.evaluate()
        # elif isinstance(self.alt, CodeBlock):
        #     return self.alt.execute()
        elif self.alt:
            return self.alt.evaluate()
        else:
            return BuiltIns['blank']

class ForLoop(ExprWithBlock):
    var: Node
    iterable: Node
    def __init__(self, header_nodes: list[Node], blk_nodes: list[Node], line: int, source: str):
        super().__init__([], blk_nodes, line, source)
        for i, node in enumerate(header_nodes):
            if node.source_text == 'in':
                self.var = mathological(header_nodes[:i])
                self.iterable = mathological(header_nodes[i+1:])
                break
        else:
            raise SyntaxErr(f"Line {self.line}: For loop expression expected 'in' keyword.")

    def __repr__(self):
        res = super().__repr__()
        return res.replace(repr(EmptyExpr(-1, '')),
                           f"{str(self.var)}, {str(self.iterable)}")

    def evaluate(self):
        iterator = BuiltIns['iter'].call(self.iterable.evaluate())
        match self.var:
            case Token(type=TokenType.Name, source_text=var_name):
                pass
            case _:
                raise NotImplementedError

        for val in iterator:
            Context.env.assign(var_name, val)
            # Context.env.locals[var_name] = val
            self.block.execute()
            if Context.break_loop:
                Context.break_loop -= 1
                break
            elif Context.continue_:
                Context.continue_ -= 1
                if Context.continue_:
                    break
            elif Context.env.return_value:
                break
        return BuiltIns['blank']

class WhileLoop(ExprWithBlock):
    def evaluate(self):
        result = py_value(None)
        for i in range(6 ** 6):
            if Context.break_loop:
                Context.break_loop -= 1
                return py_value(None)
            condition_value = self.header.evaluate()
            if BuiltIns['bool'].call(condition_value).value:
                result = self.block.evaluate()
            else:
                return result
            if Context.break_loop:
                Context.break_loop -= 1
                return py_value(None)
        raise RuntimeErr(f"Line {self.line or Context.line}: Loop exceeded limit of 46656 executions.")


class Command(Expression):
    command: str
    def __init__(self, cmd: str, line: int | None, source: str):
        super().__init__(line, source)
        self.command = cmd


class CommandWithExpr(Command):
    expr: Expression
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        self.expr = expressionize(nodes)

    def evaluate(self):
        match self.command:
            case 'exit':
                print('Exiting now')
                exit()
            case 'debug':
                Context.debug = True
                result = self.expr.evaluate()
                return result
            case 'return':
                result = self.expr.evaluate()
                Context.env.return_value = result
                return result
            case 'print':
                # print('!@>', BuiltIns['string'].call(self.expr.evaluate()).value)
                print(BuiltIns['str'].call(self.expr.evaluate()).value)
                return BuiltIns['blank']
            case 'break':
                result = self.expr.evaluate()
                match result.value:
                    case None:
                        levels = 1
                    case int() as levels:
                        pass
                    case _:
                        raise RuntimeErr(f"Line {Context.line}: "
                                         f"break expression should evaluate to non-negative integer.  Found {result}.")
                Context.break_loop += levels
            case 'continue':
                result = self.expr.evaluate()
                match result.value:
                    case None:
                        levels = 1
                    case int() as levels:
                        pass
                    case _:
                        raise RuntimeErr(f"Line {Context.line}: "
                                         f"break expression should evaluate to non-negative integer.  Found {result}.")
                Context.continue_ += levels
            case 'import':
                module_name, _, var_name = self.expr.source_text.partition(' as ')
                mod = importlib.import_module(module_name)
                globals()[var_name or module_name] = mod
                # Context.env.assign_option(var_name or module_name, piliize(a))
                Context.env.locals[var_name or module_name] = py_value(mod)
                return py_value(mod)
            # case 'inherit':
            #     result = self.expr.evaluate()
            #     types = result.value if result.instanceof(BuiltIns['tuple']) else (result,)
            #     Context.env.mro += types
            #     return py_value(Context.env.mro)
            case 'label':
                Context.env.name = BuiltIns['str'].call(self.expr.evaluate()).value
            case _:
                raise SyntaxErr(f"Line {Context.line}: Unhandled command {self.command}")

    def __repr__(self):
        return f"Cmd:{self.command}({self.expr})"


# this lambda needs to be defined in this module so that it has access to the imports processed by importlib
opt: Option = BuiltIns['python'].op_list[0]
opt.fn = lambda code: py_value(eval(code.value))

class FunctionExpr(Command):
    fn_name: str
    body: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), Block() as blk]:
                self.fn_name = name
                self.body = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Function syntax should be `function <fn_name> <block>`.")

    def evaluate(self):
        fn = Context.env.locals[self.fn_name]
        assert isinstance(fn, Function)
        # fn = Function(name=self.fn_name)
        Context.env.assign(self.fn_name, fn)
        if self.body is not None:
            Closure(self.body).execute(fn=fn)
        return fn

    def __repr__(self):
        return f'Function({self.fn_name}, {self.body})'

class TraitExpr(Command):
    trait_name: str
    body: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), Block() as blk]:
                self.trait_name = name
                self.body = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Trait syntax should be `trait <trait_name> <block>`.")

    def evaluate(self):
        # trait = Trait(name=self.trait_name)
        trait = Context.env.locals[self.trait_name]
        assert isinstance(trait, Trait)
        Context.env.assign(self.trait_name, trait)
        if self.body is not None:
            Closure(self.body).execute(fn=trait)
        return trait

    def __repr__(self):
        return f'Trait({self.trait_name}, {self.body})'

class TableExpr(Command):
    table_name: str
    traits: list[str]
    body: Block = None
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), *trait_nodes, Block() as blk]:
                self.body = blk
            case [Token(type=TokenType.Name, source_text=name), *trait_nodes]:
                pass
            case _:
                raise SyntaxErr(f"Line {self.line}: Table syntax should be `table <table_name> (@<trait>)* <block>`.")
        self.table_name = name
        self.traits = []
        for i in range(0, len(trait_nodes), 2):
            match trait_nodes[i:i+2]:
                case [Token(source_text='@'), Token(type=TokenType.Name, source_text=name)]:
                    self.traits.append(name)
                case _:
                    raise SyntaxErr(f"Line {self.line}: "
                                    f"Table traits must be listed like `@trait_name` separated only by whitespace.")

    def evaluate(self):
        # table = ListTable(name=self.table_name)
        # Context.env.assign(self.table_name, table)
        table = Context.deref(self.table_name)
        if not isinstance(table, Table):
            table = ListTable()
        traits = tuple(Context.deref(tname) for tname in self.traits)
        table.traits += traits
        for trait, name in zip(table.traits, self.traits):
            if not isinstance(trait, Trait):
                raise TypeErr(f"Line: {Context.line}: expected trait, but '{name}' is {repr(trait)}")
        if self.body is not None:
            Closure(self.body).execute(fn=table)
        table.integrate_traits()
        return table

    def __repr__(self):
        return f'Table({self.table_name} with {", ".join(map(str, self.traits))}, {self.body})'

class SlotExpr(Command):
    slot_name: str
    slot_type: Expression
    default: None | Expression | Block | str = None
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), *other_nodes]:
                self.slot_name = name
            case _:
                raise SyntaxErr(f"Line {self.line}: Slot is missing name.")
        for i, node in enumerate(other_nodes):
            if node.source_text == '=':
                self.slot_type = expressionize(other_nodes[:i])
                self.default = expressionize(other_nodes[i+1:])
                break
            if node.source_text == ':':
                self.slot_type = expressionize(other_nodes[:i])
                if i+2 != len(other_nodes):
                    raise SyntaxErr(f"Line {self.line}: Slot is missing default block after `:`.")
                self.default = other_nodes[i+1]
                break
            if node.source_text == '?' and i + 1 == len(other_nodes):
                self.default = expressionize([])
        else:
            self.slot_type = expressionize(other_nodes)

    def evaluate(self):
        slot_type = patternize(self.slot_type.evaluate())
        # match patternize(self.slot_type.evaluate()):
        #     case ParamSet(parameters=(Parameter(matcher=slot_type),)):
        #         pass
        #     case _:
        #         raise TypeErr(f"Line {Context.line}: Invalid type: {self.slot_type.evaluate()}.  "
        #                       f"Expected value, table, trait, or single-parameter pattern.")
        match self.default:
            case None:
                default = None
            case Block() as blk:
                default = Closure(blk)
            case Node():
                default = self.default.evaluate()
                # default = Function({Parameter(AnyMatcher(), 'self'): default_value})
            case _:
                raise ValueError("Unexpected default")
        if default is not None:
            default = Function({Parameter(AnyMatcher(), 'self'): default})
        slot = Slot(self.slot_name, slot_type, default)
        match Context.env.fn:
            case Trait() as trait:
                trait.fields.append(slot)
            case Table(traits=(Trait() as trait, *_)):
                trait.fields.append(slot)
            case Function() as fn:
                fn.update_field(slot)
            case _:
                raise AssertionError

        return BuiltIns['blank']

    def __repr__(self):
        default = f", {self.default}" or ''
        return f"Slot({self.slot_name}, {self.slot_type}{default})"

class FormulaExpr(Command):
    formula_name: str
    formula_type: Expression
    block: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), *patt_nodes, Token(source_text=':'), Block() as blk]:
                self.formula_name = name
                self.formula_type = expressionize(patt_nodes)
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Formula syntax is: `formula <name> <type_expr>?: <block>`."
                                f"Eg, `formula count int: len[self.items]`")

    def evaluate(self):
        formula_type = patternize(self.formula_type.evaluate())
        # match patternize(self.formula_type.evaluate()):
        #     case ParamSet(parameters=(Parameter(pattern=formula_type), )):
        #         pass
        #     case _:
        #         raise TypeErr(f"Line {Context.line}: Invalid type: {self.formula_type.evaluate()}.  "
        #                       f"Expected value, table, trait, or single-parameter pattern.")
        patt = ParamSet(Parameter(Context.env.fn, binding='self'))
        formula_fn = Function({patt: Closure(self.block)})
        formula = Formula(self.formula_name, formula_type, formula_fn)
        match Context.env.fn:
            case Trait() as trait:
                trait.fields.append(formula)
            case Table(traits=(Trait() as trait, *_)):
                trait.fields.append(formula)
            case Function() as fn:
                fn.update_field(formula)
            case _:
                raise AssertionError
        return BuiltIns['blank']

    def __repr__(self):
        return f"Formula({self.formula_name}, {self.formula_type}, {self.block})"

class SetterExpr(Command):
    field_name: str
    param_nodes: Expression
    block: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), Token(source_text='.'),
                  ListNode(items=[Expression() as stmt]) as param_list,
                  Token(source_text=':'), Block() as blk]:
                self.field_name = name
                # self.param_nodes = stmt.nodes
                self.params = param_list
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Setter syntax is: `setter <name>[<value parameter>]: <block>`."
                                f"Eg, `setter description[str desc]: self._description = trim[desc]`")

    def evaluate(self):
        # fn = Function({ParamSet(Parameter(AnyMatcher(), 'self'), make_param(self.param_nodes)):
        #                    CodeBlock(self.block)})
        params: ParamSet = self.params.evaluate()
        params.parameters = (Parameter(AnyMatcher(), 'self'),) + params.parameters
        assert len(params) == 2
        fn = Function({params: Closure(self.block)})
        setter = Setter(self.field_name, fn)
        match Context.env.fn:
            case Trait() as trait:
                trait.fields.append(setter)
            case Table(traits=(Trait() as trait, *_)) as table:
                trait.fields.append(setter)
            case Function() as fn:
                fn.update_field(setter)
            case _:
                raise AssertionError

        return BuiltIns['blank']

    def __repr__(self):
        return f"Setter({self.field_name}, {self.params}, {self.block})"

class OptExpr(Command):
    params: list[Expression]
    return_type: Expression | None
    block: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [ListNode(nodes=params), *patt_nodes, Token(source_text=':'), Block() as blk]:
                self.params = params
                self.return_type = expressionize(patt_nodes)
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Opt syntax is: `opt [param1, param2, ...] <type_expr>?: <block>`."
                                f"Eg, `opt [int i, int j] str | blank : ...`")

    def evaluate(self):
        match patternize(self.return_type.evaluate()):
            case ParamSet(parameters=(Parameter(pattern=Matcher() as return_type))):
                pass
            case Matcher() as return_type:
                pass
            case _:
                raise TypeErr(f"Line {Context.line}: Invalid return type: {self.return_type.evaluate()}.  "
                              f"Expected value, table, trait, or single-parameter pattern.")
        # pattern = ParamSet(*map(make_param, self.params))
        pattern = ParamSet(*(p.evaluate() for p in self.params))
        match Context.env.fn:
            case Trait() as trait:
                pass
            case Table(traits=(Trait() as trait, *_)):
                pass
            case Function():
                raise TypeErr(f"To add options to functions, omit the Opt keyword.")
            case _:
                raise AssertionError
        trait.assign_option(pattern, Closure(self.block))
        return BuiltIns['blank']

    def __repr__(self):
        ret_type = f', {self.return_type}' if self.return_type else ''
        return f"Opt({self.params}{ret_type}, {self.block})"

class Declaration(Command):
    var_name: str
    # context_expr: None | Expression = None
    value_expr: None | Expression = None

    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name)]:
                self.var_name = name
            case [Token(type=TokenType.Name, source_text=name), Token(source_text='='), *other_nodes]:
                self.var_name = name
                if other_nodes:
                    self.value_expr = expressionize(other_nodes)
                else:
                    raise SyntaxErr(f"Line {self.line}: Missing right-hand-side expression for value")
            case _:
                raise SyntaxErr(f"Line {self.line}: invalid syntax for declaration.  "
                                f"Expected `[local|var] var_name [= <expression>]`")
            # case _:
            #     for i, node in enumerate(nodes):
            #         if node.type == TokenType.Operator and node.source_text == '=':
            #             self.value_expr = expressionize(nodes[i + 1:])
            #     else:
            #         i = len(nodes)
            #         # HOLD ON A SECOND... what does it mean to set `local foo.bar = "something"`
            #         # or `local (foo or bar).prop = "something"`
            #         # or `local foo(arg1, 2).prop = "something"`
            #         # ? should I just raise an error in this case?
            #     name_tok = nodes[i - 1]
            #     if name_tok.type != TokenType.Name:
            #         raise SyntaxErr(f"Line {self.line}: Expected name token before = in local expression.")
            #     self.var_name = name_tok.source_text
            #     context_nodes = nodes[:i - 1]
            #     if not context_nodes or context_nodes.pop().source_text != '.':
            #         raise SyntaxErr(f"Line {self.line}: Invalid left-hand-side for assignment.")
            #     self.context_expr = expressionize(context_nodes)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.var_name}={self.value_expr})"

class LocalExpr(Declaration):
    def evaluate(self):
        value = self.value_expr.evaluate() if self.value_expr else None
        # if self.context_expr:
        #     if value is None:
        #         raise SyntaxErr(f"Line {self.line}: I guess I should have caught this error in the parsing phase...")
        #     rec = self.context_expr.evaluate()
        #     rec: Record
        #     rec.set(self.var_name, value)
        #     # also raise an error here
        # else:
        Context.env.locals[self.var_name] = value
        return value or BuiltIns['blank']

class VarExpr(Declaration):
    def evaluate(self):
        value = self.value_expr.evaluate() if self.value_expr else BuiltIns['blank']
        Context.env.vars[self.var_name] = value
        return value


expressions = {
    'if': Conditional,
    'for': ForLoop,
    'while': WhileLoop,
    'trait': TraitExpr,
    'table': TableExpr,
    'function': FunctionExpr,
    'slot': SlotExpr,
    'formula': FormulaExpr,
    'opt': OptExpr,
    'setter': SetterExpr,
    'local': LocalExpr,
    'var': VarExpr,
}

# def piliize(val):
#     if isinstance(val, list | tuple):
#         # gen = (py_value(v) for v in val)
#         records = map(py_value, val)
#         if isinstance(val, tuple):
#             return py_value(tuple(records))
#         if isinstance(val, list):
#             return List(list(records))
#     return py_value(val)

def py_eval(code):
    return piliize(eval(code.value))

class EmptyExpr(Expression):
    def evaluate(self):
        return BuiltIns['blank']

    def __repr__(self):
        return "EmptyExpr()"

class SingleNode(Expression):
    def __init__(self, node: Node, line: int | None, source: str):
        super().__init__(line, source)
        self.node = node
    def evaluate(self):
        return eval_node(self.node)


def eval_node(node: Node) -> Record:
    match node:
        case Expression() as statement:
            return expressionize(statement).evaluate()
        case Token() as tok:
            return eval_token(tok)
        case Block() as block:
            return Closure(block).execute(fn=Function())
        case ListNode(list_type=list_type, items=items):
            match list_type:
                case ListType.List:
                    return py_value(list(map(eval_node, items)))
                case ListType.Args:
                    return eval_args_list(items)
                # case ListType.Params:
                #     return ParamSet(*map(make_param, items))
                case ListType.Tuple:
                    return py_value(tuple(map(eval_node, items)))
                case ListType.Function:
                    fn = Function()
                    Closure(Block(items)).execute(fn=fn)
                    return fn
                case _:
                    raise NotImplementedError
        case StringNode(nodes=nodes):
            return py_value(''.join(map(eval_string_part, nodes)))
    raise ValueError(f'Could not evaluate node {node} at line: {node.pos}')

def eval_string_part(node: Node) -> str:
    if node.type == TokenType.StringPart:
        return eval_string(node.source_text)
    return BuiltIns['str'].call(node.evaluate()).value
    # if isinstance(node, Expression):
    #     return BuiltIns['str'].call(node.evaluate()).value
    # raise ValueError('invalid string part')


def precook_args(op: Operator, lhs: Node, rhs: Node) -> list[Record]:
    if op.binop and lhs and rhs:
        args = [lhs.evaluate(), rhs.evaluate()]
        # args = [expressionize(lhs).evaluate(), expressionize(rhs).evaluate()]
    elif op.prefix and rhs or op.postfix and lhs:
        args = [(rhs or lhs).evaluate()]
        # args = [expressionize(rhs or lhs).evaluate()]
    else:
        raise ArithmeticError("Mismatch between operator type and operand positions.")
    return args


# Operator.eval_args = precook_args
# I think this has been replaced by a simpler function: Operator.eval_args


def eval_token(tok: Token) -> Record:
    s = tok.source_text
    match tok.type:
        case TokenType.Singleton:
            return py_value(singletons[s])
        case TokenType.Number:
            return py_value(read_number(s, Context.settings['base']))
        case TokenType.StringLiteral:
            return py_value(s.strip("`"))
        case TokenType.Name:
            if s == 'self':
                return Context.deref(s, Context.env.caller)
            return Context.deref(s)
        case _:
            raise Exception("Could not evaluate token", tok)


def eval_string(text: str):
    value = ""
    for i in range(len(text)):
        if i and text[i-1] == '\\':
            continue
        if text[i] == '\\':
            match text[i+1]:
                case 'n':
                    value += '\n'
                case 'r':
                    value += '\r'
                case 't':
                    value += '\t'
                case 'b':
                    value += '\b'
                case 'f':
                    value += '\f'
                case c:
                    # assume char is one of: ', ", {, \
                    value += c
        else:
            value += text[i]
    return value


def eval_args_list(statements: list[Expression]) -> Args:
    pos_args = []
    named_args = {}
    flags = set()
    for stmt in statements:
        match stmt.nodes:
            case [Token(type=TokenType.Name, source_text=name), Token(source_text='='), *expr_nodes]:
                named_args[name] = expressionize(expr_nodes).evaluate()
            case [Token(source_text='!'), Token(type=TokenType.Name, source_text=name)]:
                flags.add(name)
            case nodes:
                pos_args.append(expressionize(nodes).evaluate())

    return Args(*pos_args, flags=flags, **named_args)

def get_iterable_DEPRECATED(val: Record):
    match val:
        case PyValue(value=tuple() | list() | set() | frozenset() as iterable):
            return iterable
        case Table(records=iterable):
            return iterable
        case PyValue(value=str() as string):
            return (py_value(c) for c in string)
    return None


def make_value_param(param_nodes: list[Node]) -> Parameter:
    name = None
    match param_nodes:
        case []:
            raise SyntaxErr(f"Expected function parameter on line {Context.line}.")
        case [Token(type=TokenType.Name, source_text=name)]:
            value = py_value(name)
        case [Token(type=TokenType.Name, source_text=name)]:
            value = py_value(name)
        case _:
            value = expressionize(param_nodes).evaluate()
    return Parameter(ValueMatcher(value), name)


def make_param(param_nodes: list[Node] | Expression) -> Parameter:
    if isinstance(param_nodes, Expression):
        param_nodes = param_nodes.nodes
    if not param_nodes:
        raise SyntaxErr(f"Expected function parameter on line {Context.line}; no nodes found.")
    quantifier = ""
    match param_nodes:
        case [*_, Token(type=TokenType.Operator, source_text=op)] if op in ('?', '+', '*', '??', '+?', '*?'):
                quantifier = op
                param_nodes = param_nodes[:-1]
        case [Token(type=TokenType.Name, source_text=dot_name)]:
            return Parameter(dot_name)
        case []:
            raise SyntaxErr(f"Expected function parameter on line {Context.line}; no nodes found.")
    name = None
    match param_nodes:
        case [node]:
            pattern_nodes = param_nodes
            # return Parameter(patternize(eval_node(node)), quantifier=quantifier)
        case [*pattern_nodes, Token(type=TokenType.Name, source_text=name)]:
            last_op = Op.get(param_nodes[-2].source_text, None)
            if last_op and last_op.binop:
                if last_op.postfix:
                    print("WARNING: ambiguous pattern")
                else:
                    pattern_nodes = param_nodes
        case [*pattern_nodes]:
            pass
        case _:
            raise SyntaxErr(f"Expected function parameter on line {Context.line}; found only quantifier {quantifier}")
    try:
        expr_val = expressionize(pattern_nodes).evaluate()
    except NoMatchingOptionError as e:
        print(f"Line {Context.line}: Warning: did you try to assign a bare function name without defining a type?")
        raise e
    param = patternize(expr_val).parameters[0]
    param.name = name
    param.quantifier = quantifier
    return param


def split(nodes: list[Node], splitter: TokenType) -> list[list[Node]]:
    start = 0
    groups: list[list[Node]] = []
    for i, node in enumerate(nodes):
        if isinstance(node, Token) and node.type == splitter:
            groups.append(nodes[start:i])
            start = i+1
    if start < len(nodes):
        groups.append(nodes[start:])
    return groups


# def read_option_OLD(nodes: list[Node], is_value=False) -> Option:
#     dot_option = nodes[0].source_text == '.'
#     match nodes:
#         # .[].[]
#         case[Token(source_text='.'), ListNode() as opt, Token(source_text='.'), ListNode() as param_list]:
#             fn_nodes = [Token('pili'), Token('.'), opt]
#             param_list = [item.nodes for item in param_list.nodes]
#         # .fn_nodes.[]
#         case [Token(source_text='.'), *fn_nodes, Token(source_text='.'), ListNode() as param_list]:
#             param_list = [item.nodes for item in param_list.nodes]
#         case [Token(source_text='.'), *fn_nodes]:
#             param_list = []
#         case [*fn_nodes, ListNode() as param_list]:
#             if fn_nodes and fn_nodes[-1].source_text == '.':
#                 fn_nodes.pop()
#             param_list = [item.nodes for item in param_list.nodes]
#         case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name) as name_tok]:
#             param_list = [[name_tok]]
#         case _:
#             param_list = split(nodes, TokenType.Comma)
#             fn_nodes = False
#     if fn_nodes:
#         try:
#             if len(fn_nodes) == 1:
#                 name = fn_nodes[0].source_text
#                 fn_val = Context.deref(name, False)
#             else:
#                 fn_val = expressionize(fn_nodes).evaluate()
#                 if fn_val.type is BuiltIns['str']:
#                     fn_val = Context.deref(fn_val.value)
#         except NoMatchingOptionError:
#             if dot_option:
#                 # raise NoMatchingOptionError(f"Line {Context.line}: "
#                 #                             f"dot option {' '.join((map(str, fn_nodes)))} not found.")
#                 # make new function in the root scope
#                 temp_env = Context.env
#                 Context.env = Context._env[0]
#             opt = read_option(fn_nodes, True)
#             if opt.is_null():
#                 fn_val = Function()
#                 opt.value = fn_val
#             else:
#                 fn_val = opt.value
#             if dot_option:
#                 Context.env = temp_env
#             # how many levels deep should this go?
#             # This will recurse infinitely, potentially creating many function
#         # if fn_val.type != BasicType.Function:
#         #     raise RuntimeErr(f"Line {Context.line}: "
#         #                      f"Cannot add option to {fn_val.type.value} {' '.join((map(str, fn_nodes)))}")
#         fn = fn_val
#         definite_env = True
#     else:
#         fn = Context.env
#         definite_env = not is_value
#     params = map(make_param if not is_value else make_value_param, param_list)
#     if dot_option:
#         patt = ParamSet(Parameter(TableMatcher(Context.env)), *params)
#     else:
#         patt = ParamSet(*params)
#     # try:
#     #     # option = fn.select(patt, walk_prototype_chain=False, ascend_env=not definite_env)
#     #     option = fn.select_by_pattern(patt, walk_prototype_chain=False, ascend_env=not definite_env)
#     #     """critical design decision here: I want to have walk_prototype_chain=False so I don't assign variables from the prototype..."""
#     # except NoMatchingOptionError:
#     #     option = fn.add_option(patt)
#     option = fn.aselect_by_pattern(patt, ascend_env=True) or fn.add_option(patt)
#     option.dot_option = dot_option
#     return option

def read_option2(nodes: list[Node]) -> tuple[Function, ParamSet, bool]:
    nodes = nodes[:]
    dot_option = nodes[0].source_text == '.'
    if dot_option:
        if len(nodes) > 2 and not isinstance(nodes[1], ListNode):
            nodes.pop(0)
        if not isinstance(nodes[-1], ListNode):
            nodes.append(Token('.'))
            nodes.append(ListNode([], ListType.Params))
    # if len(nodes) > 1 and nodes[-2].source_text != '.':
    #     option_node = List([Statement(nodes)])
    #     nodes = []
    # else:
    option_node = nodes.pop()
    # nodes.pop() if len(nodes) else None
    if len(nodes):
        penultimate = nodes.pop()
        if penultimate.source_text != '.':
            # read the whole lhs as a single parameter
            option_node = ListNode([Expression([*nodes, penultimate, option_node])], ListType.Params)
            nodes = []
    match option_node:
        case ListNode(nodes=param_list):
            param_list = [item.nodes for item in param_list]
        case Token(type=TokenType.Name) as name_tok:
            param_list = [[name_tok]]
        case _:
            # raise SyntaxErr(f"Line {Context.line}: Cannot read option {option_node}")
            param_list = [[option_node]]
    # if len(nodes):
    #     dot = nodes.pop()
    #     if dot.source_text != '.':
    #         raise SyntaxErr(f"Line {Context.line}: Couldn't read option following {dot}; expected '.' or '['")
    arg_list = name = None
    match nodes:
        # case [*context_nodes, Token(source_text='.'), List(nodes=arg_list)]:
        #     pass
        case [*context_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
            pass
        case [Token(type=TokenType.Name, source_text=name)]:
            context_nodes = False
        case _:
            context_nodes = nodes
    if context_nodes:
        context_fn = expressionize(context_nodes).evaluate()
        ascend = False
    else:
        context_fn = Context.env
        ascend = True

    if name:  # and not dot_option:
        try:
            fn = context_fn.deref(name, ascend)
        except NoMatchingOptionError:
            fn = Function(name=name)
            if dot_option:
                # Context.root.add_option(name, fn)
                raise NotImplementedError
            else:
                context_fn.assign_option(name, fn)
    else:
        fn = context_fn


    # if arg_list:
    #     args = [expressionize(arg).evaluate() for arg in arg_list]
    #     try:
    #         opt, bindings = context_fn.select_and_bind(args, ascend)
    #     except NoMatchingOptionError:
    #         opt = bindings = None
    #     if opt:
    #         fn = opt.resolve(args, context_fn, bindings)
    #     else:
    #         fn = context_fn.add_option(patt, resolution).value
    # elif name:
    #     try:
    #         fn = context_fn.deref(name, ascend)
    #     except NoMatchingOptionError:
    #         fn = context_fn.add_option(name, Function(name=name)).value
    # else:
    #     fn = context_fn

    params = map(make_param, param_list)
    if dot_option:
        # patt = ParamSet(Parameter(TableMatcher(Context.env)), *params)
        raise NotImplementedError
    else:
        patt = ParamSet(*params)
    return fn, patt, dot_option


def read_option(nodes: list[Node]) -> tuple[Function, ParamSet, bool]:
    nodes = nodes[:]
    dot_option = nodes[0].source_text == '.'
    if dot_option:
        if len(nodes) > 2 and not isinstance(nodes[1], ListNode):
            # why don't we unconditionally pop the dot?
            nodes.pop(0)
        if not isinstance(nodes[-1], ListNode):
            # append implicit empty-param-set after dot-option
            nodes.append(Token('.'))
            nodes.append(ListNode([], ListType.Params))

    option_node = nodes.pop()
    if len(nodes):
        penultimate = nodes.pop()
        if penultimate.source_text != '.':
            # read the whole lhs as a single parameter
            option_node = ListNode([Expression([*nodes, penultimate, option_node])], ListType.Params)
            nodes = []

    param_list: list[list[Node]]
    match option_node:
        case ListNode(nodes=param_list_nodes):
            param_list = [item.nodes for item in param_list_nodes]
        case Token(type=TokenType.Name) as name_tok:
            # not sure if I want to allow this anymore
            param_list = [[name_tok]]
        case _:
            # raise SyntaxErr(f"Line {Context.line}: Cannot read option {option_node}")
            param_list = [[option_node]]

    # at this point, we should have three options: nodes is empty, nodes is one name, or nodes is an expression
    # which should evaluate to the function where we assign the option
    match nodes:
        case []:
            context_fn: Function = Context.env.fn
        case [Token(type=TokenType.Name, source_text=name)]:
            context_fn = Context.deref(name, None)
            if context_fn is None:
                context_fn = Function(name=name)
                if dot_option:
                    Context.root.locals[name] = context_fn
                else:
                    Context.env.locals[name] = context_fn
        case _:
            context_fn = expressionize(nodes).evaluate()

    params = map(make_param, param_list)
    if dot_option:
        match Context.env.fn:
            case Trait() | Table() as t:
                matcher = TableMatcher(t)
            case Function() as f:
                matcher = ValueMatcher(f)
            case _:
                raise RuntimeErr(f"Line {Context.line}: "
                         f"Dot options can only be declared in a scope which is a function, table, or trait.")
        patt = ParamSet(Parameter(matcher, name='self'), *params)
    else:
        patt = ParamSet(*params)
    return context_fn, patt, dot_option


def bytecode(node: Node) -> tuple[Inst, Inst]:
    match node:
        case Token(type=TokenType.Name, source_text=name):
            tail = Inst()
            head = Inst().match(AnyMatcher(name), tail)
            return head, tail
        case OpExpr(op=Operator(text='@'), terms=[term, Token(type=TokenType.Name, source_text=name)]):
            head, tail = bytecode(term)
            Inst().save()
            return head, tail
        case _:
            pass

    return Inst(), Inst()

def make_pattern(param_nodes: list[OpExpr]) -> Pattern:
    params: list[Inst] = []
    for param in param_nodes:
        head, tail = bytecode(param)
        match param.op.text:
            case '?':
                params.append(Inst().split(head, tail), tail)
            case '+':
                t = Inst()
                tail.Split(head, t)
                return head, t
            case '*':
                t = Inst()
                h = Inst().Split(head, t)
                tail.Jump(h)
                return h, t
            case '??':
                return Inst().Split(tail, head), tail
            case '+?':
                t = Inst()
                tail.Split(t, head)
                return head, t
            case '*?':
                t = Inst()
                h = Inst().Split(t, head)
                tail.Jump(h)
                return h, t


if __name__ == "__main__":
    pass