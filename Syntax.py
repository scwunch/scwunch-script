import tables
from stub import *
import math
import importlib
import re
from enum import Enum, EnumMeta
from Env import *
from operators import Operator
from tables import *

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
    StringLiteral = '`string`'
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
    LeftParen = '('
    RightParen = ')'
    LeftBracket = '['
    RightBracket = ']'
    LeftBrace = "{"
    RightBrace = "}"
    Comma = ','
    Semicolon = ';'
    NewLine = '\n'
    BlockStart = '\t'
    BlockEnd = '\t\n'
    Debug = "#debug"
    EOF = "end of file"


TokenType.map = {
        'else': TokenType.Else,
        'elif': TokenType.Else,
        '(': TokenType.LeftParen,
        ')': TokenType.RightParen,
        '[': TokenType.LeftBracket,
        ']': TokenType.RightBracket,
        '{': TokenType.LeftBrace,
        '}': TokenType.RightBrace,
        ',': TokenType.Comma,
        ';': TokenType.Semicolon,
        '\n': TokenType.NewLine,
        '#debug': TokenType.Debug,
    }


class Commands(Enum):
    Print = 'print'
    # If = 'if'
    # Else = 'else'
    # For = 'for'
    # While = 'while'
    # Local = 'local'
    # Var = 'var'
    Return = 'return'
    Break = 'break'
    Continue = 'continue'
    Exit = 'exit'
    Debug = 'debug'
    # Else = 'else'
    Import = 'import'
    # Inherit = 'inherit'
    Label = 'label'
    # Function = 'function'
    # Table = 'table'
    # Trait = 'trait'
    Slot = 'slot'
    Formula = 'formula'
    Opt = 'opt'
    Setter = 'setter'


for cmd in Commands:
    TokenType.map[cmd.value] = TokenType.Command

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
    Var = 'var'
    Local = 'local'


for op in OperatorWord:
    TokenType.map[op.value] = TokenType.Operator

class Singletons(Enum):
    blank = 'blank'
    true = 'true'
    false = 'false'
    inf = 'inf'


for single in Singletons:
    TokenType.map[single.value] = TokenType.Singleton
SINGLETONS = {'blank': None, 'false': False, 'true': True, 'inf': math.inf}

class KeyWords(Enum):
    If = 'if'
    For = 'for'
    While = 'while'
    Try = 'try'
    Except = 'except'
    Function = 'function'
    Table = 'table'
    Trait = 'trait'


for key in KeyWords:
    TokenType.map[key.value] = TokenType.Keyword

# class Stmt(Enum):
#     Empty = 'EmptyExpr'
#     Cmd = 'Command'
#     Expr = 'Expression'
#     IfElse = 'IfElse'
#     Asgmt = 'Assignment'


# def token_mapper(item: str) -> TokenType:
#     return TokenType._value2member_map_.get(item, TokenType.Unknown)
# def command_mapper(item: str) -> Commands:
#     return Commands._value2member_map_.get(item)
# def singleton_mapper(item: str) -> Singletons:
#     return Singletons._value2member_map_.get(item, None)
# def keyword_mapper(item: str) -> KeyWords:
#     return KeyWords._value2member_map_.get(item, None)

class Position:
    ln: int
    ch: int | None
    start_index: int = None
    stop_index: int = None
    def __init__(self, pos: tuple[int, int | None], start: int = None, end: int = None):
        self.ln, self.ch = pos
        if start is not None:
            self.start_index = start
            self.stop_index = end

    def __str__(self):
        if self.ch is None:
            return str(self.ln)
        return str((self.ln, self.ch))

    def __repr__(self):
        return f"Position{tuple(self.__dict__.values())}"

    # def __getitem__(self, item):
    #     if item == 0:
    #         return self.ln
    #     if item == 1:
    #         return self.ch
    #     raise IndexError

    def __add__(self, other):
        if other.stop_index is None:
            print(f"SyntaxWarning {self.pos}: Missing stop index of added position.")
            return Position(self.pos)
        return Position(self.pos, self.start_index, other.stop_index)

    @property
    def pos(self):
        return self.ln, self.ch

    def slice(self):
        if self.start_index is None:
            return slice(0, 0)
        else:
            return slice(self.start_index, self.stop_index)


class Node:
    # pos: tuple[int, int | None] | None
    type = TokenType.Unknown
    pos: Position = None
    source_text: str
    # source_slice: slice = None

    def get_line(self):
        return getattr(self, '_line', self.pos and self.pos.ln)

    def set_line(self, line: int):
        raise DeprecationWarning("line property of nodes should be read-only")
        self._line = line

    def evaluate(self):
        raise NotImplementedError

    def abstract(self, unary=True):
        if unary:
            return self
        else:
            raise NotImplementedError(f"Cannot abstract {self} as operator")

    line = property(get_line, set_line)

    @property
    def tokens(self):
        return Context.tokens[self.token_slice]

    @property
    def source_text(self):
        # if self.source_slice:
        #     return Context.source_code[self.source_slice]
        return Context.source_code[self.pos.slice()]

    def eval_pattern(self) -> Pattern:
        match self.evaluate():
            case Pattern() as patt:
                return patt
            case Matcher() as matcher:
                return Parameter(matcher)
            case Record() as rec:
                return Parameter(ValueMatcher(rec))
        raise NotImplementedError(f"Line {self.line}: failed to evaluate pattern of {self}")


def big_mathological(nodes: list[Node]) -> Node:
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

def mathological(nodes: list[Node]) -> Node:
    if isinstance(nodes, list):
        nodes = Concrete(nodes)
    match nodes.nodes:
        case []:
            return EmptyExpr()
        case [Concrete() as c]:
            return mathological(c)
        case [Node() as node]:
            return node
    expr = Expressionizer(nodes)
    return expr.node

class Token(Node):
    text: str
    this_is_a_dumb_hack = None  # purely for making a stupid shortcut in the matching portion of eval_colon_args
    __match_args__ = ('this_is_a_dumb_hack',)
    def __init__(self, text: str,
                 type: TokenType = None,
                 pos: tuple[int, int | None] = None,
                 start: int = None, stop: int = None):
        self.text = text
        if pos:
            self.pos = Position(pos, start, stop)
        if type is None:
            self.type = TokenType.map.get(text, TokenType.Name)
        else:
            self.type = type
        # if start is not None:
        #     self.source_slice = slice(start, stop)

    def evaluate(self):
        s = self.source_text
        match self.type:
            case TokenType.Singleton:
                return py_value(SINGLETONS[s])
            case TokenType.Number:
                return py_value(read_number(s, Context.settings['base']))
            case TokenType.StringLiteral:
                return py_value(s.strip("`"))
            case TokenType.Name:
                if s == 'self':
                    return Context.deref(s, Context.env.caller)
                return Context.deref(s)
        raise NotImplementedError(f"Line {self.line}: Could not evaluate token", self)

    def eval_pattern(self):
        if self.type == TokenType.Name:
            return Parameter(AnyMatcher(), self.source_text)
        return patternize(self.evaluate())

    def __str__(self):
        return self.text or str(self.type)

    def __repr__(self):
        data = [repr(self.source_text or self.text)]
        if self.pos:
            data.extend(map(str, self.pos.__dict__.values()))
        return f"Token<{self.type.name}>({', '.join(data)})"

# class ExprType(Enum):
#     Group = '(group)'
#     Parens = '(parens)'
#     Brackets = '[brackets]'
#     Braces = '{braces}'
#     Block = 'block'
#     Command = 'command'
#     Assignment = ':='
#     Multi = "multi;expr"
#     IfElse = 'something if this else that'
#     Default = 'mathological'  # operator expression
#
#
# class Expression(Node):
#     type: ExprType
#
# class Concrete(Expression):
#     """ concrete/proto expressions are just containers created by the CST.
#     They must be transformed into AST nodes before evaluation. """
#     nodes: list[Node]
#     type: ExprType
#     def __init__(self, nodes: list[Node], type: TokenType | ExprType = ExprType.Default):
#         for node in nodes:
#             if isinstance(node, list):
#                 pass
#         self.nodes: list[Token | Concrete] = nodes
#         self.type = type
#         if nodes:
#             for n in nodes:
#                 if n.pos is not None:
#                     self.pos = n.pos
#                     break
#             # start = next((node.pos.start_index for node in nodes if node.pos and node.pos.start_index), None)
#             # if start is not None:
#             # stop = next((node.pos.stop_index for node in reversed(nodes) if node.pos), None)
#             # self.pos.stop_index = stop
#             if self.pos:
#                 for s in reversed(nodes):
#                     if s is n:
#                         break
#                     if s.pos is None or s.pos.stop_index is None:
#                         continue
#                     self.pos = Position(self.pos.pos, self.pos.start_index, s.pos.stop_index)
#                     break
#
#     def __len__(self):
#         return len(self.nodes)
#
#     def __getitem__(self, item):
#         return self.nodes[item]
#
#     def __repr__(self):
#         if self.type is ExprType.Default:
#             return f"Concrete{tuple(self.nodes)}"
#         return f"Concrete<{self.type.name}>{tuple(self.nodes)}"
#
#     def evaluate(self):
#         raise NotImplementedError("Proto expressions are not meant to be evaluated.")
#
#     def abstract(self, unary=True):
#         match self.nodes:
#             case []:
#                 return EmptyExpr(self.pos)
#             case [Concrete() as node]:
#                 return node.abstract()
#             case [node]:
#                 return node
#         match self.type:
#             case ExprType.Default:
#                 return mathological(self.nodes)
#             case ExprType.Braces:
#                 return FunctionLiteral(tuple(n.abstract() for n in self.nodes[1:-1]), self.pos)
#             case ExprType.Block:
#                 return Block(self.nodes, self.pos)
#         raise AssertionError("This method should not be called in this case.  It should have been caught by the Expressionizer.")
#
#
# class UnaryNode(Node):
#     node: Node
#     def __init__(self, node: Node, ):
#         self.node = node
#         self.pos = node.pos
#         self.source_slice = node.source_slice
#
#
# class BinaryNode(Node):
#     left: Node
#     right: Node
#     def __init__(self, left: Node, right: Node):
#         self.left = left
#         self.right = right
#         self.pos = left.pos
#         self.source_slice = slice(left.source_slice.start, right.source_slice.stop)
#
#
# class AbstractExpression(Expression):
#     type = ExprType.Default

class ListNode(Node):
    nodes: list[Node]
    def __init__(self, nodes: list[Node], pos: Position = None):
        self.nodes = nodes
        self.pos = pos  # or Concrete(nodes).pos


class StringNode(ListNode):
    def __repr__(self):
        res = ''.join(str(n) if n.type == TokenType.StringPart else f"{{{repr(n)}}}" for n in self.nodes)
        return f'"{res}"'

    def evaluate(self):
        def eval_string_parts(nodes: list[Node]) -> str:
            for node in nodes:
                if node.type != TokenType.StringPart:
                    yield BuiltIns['str'].call(node.evaluate()).value
                    continue
                text = node.source_text
                start = i = 0
                while i < len(text):
                    if text[i] == '\\':
                        c = text[i+1]
                        if c in "nrtbf'\"{\\":
                            yield text[start:i]
                            i += 2
                            start = i
                            match c:
                                case 'n':
                                    yield '\n'
                                case 'r':
                                    yield '\r'
                                case 't':
                                    yield '\t'
                                case 'b':
                                    yield '\b'
                                case 'f':
                                    yield '\f'
                                case _:
                                    # assume char is one of: ', ", {, \
                                    yield c
                        else:
                            raise SyntaxErr(f"Unrecognized escape character '{c}' @{node.pos}")
                    i += 1
                yield text[start:]

        return py_value(''.join(eval_string_parts(self.nodes)))

# class Expression(Node):
#     def __init__(self, line, source):
#         self.pos = (line, -1)
#         self.line = line
#         self.source_text = source
#
#     def evaluate(self):
#         raise NotImplementedError()
#
#     # def __repr__(self):
#     #     if self.line:
#     #         return f"Line {self.line}: {self.source_text}"
#     #     return self.source_text

class Block(ListNode):
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
                 table_names: set[str] = empty, trait_names: set[str] = empty, func_names: set[str] = empty,
                 pos: Position = None):
        super().__init__(nodes, pos)
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
            return f"Block[{repr(self.statements[0])}]"
        else:
            return f"Block[{len(self.statements)} statements]"


# class ListType(Enum):
#     List = '[list]'
#     Tuple = '(tuple)'
#     Function = "{function}"
#     Args = '[args]'
#     Params = '[params]'
#     FieldMatcher = '(field: matcher)'

class ListLiteral(ListNode):
    def evaluate(self):
        return py_value(list(n.evaluate() for n in self.nodes))

    def eval_pattern(self):
        raise NotImplementedError

    # def old_eval(self):
    #     items = (n.evaluate() for n in self.nodes)
    #     match self.list_type:
    #         case ListType.List:
    #             return py_value(list(items))
    #         case ListType.Tuple:
    #             return py_value(tuple(items))
    #         case ListType.Args:
    #             named_args = {}
    #             flags = set()
    #             def generate_args(nodes: list[Node]):
    #                 for node in nodes:
    #                     match node:
    #                         case OpExpr('=',
    #                                     terms=(Token(type=TokenType.Name, source_text=name), val_node)):
    #                             named_args[name] = val_node.evaluate()
    #                         case OpExpr('!',
    #                                     terms=(Token(type=TokenType.Name, source_text=name),)):
    #                             flags.add(name)
    #                         case _:
    #                             yield node.evaluate()
    #             return Args(*generate_args(self.nodes), flags=flags, named_arguments=named_args)
    #         case ListType.Function:
    #             fn = Function()
    #             Closure(Block(self.items)).execute(fn=fn)
    #             return fn
    #         case ListType.FieldMatcher:
    #             field_dict = {}
    #             def generate_matchers(nodes: list[Node]):
    #                 for node in nodes:
    #                     match node:
    #                         case OpExpr(':',
    #                                     terms=(Token(type=TokenType.Name, source_text=name), patt_node)):
    #                             match patt_node:
    #                                 case Token(type=TokenType.Name, source_text=binding):
    #                                     field_dict[name] = Parameter(AnyMatcher(), binding)
    #                                 case _:
    #                                     field_dict[name] = patternize(patt_node.evaluate())
    #                         case _:
    #                             yield node.evaluate()
    #             return FieldMatcher(tuple(generate_matchers(self.nodes)), field_dict)
    #     raise NotImplementedError(f'Line {self.line}: ListNode<{self.list_type}> not yet implemented.')

    def __repr__(self):
        res = map(str, self.nodes)
        return f"{self.__class__.__name__}[{', '.join(res)}]"


class TupleLiteral(ListLiteral):
    def evaluate(self):
        return py_value(tuple(n.evaluate() for n in self.nodes))


class ArgsNode(ListNode):
    def evaluate(self):
        named_args = {}
        flags = set()

        def generate_args(nodes: list[Node]):
            for node in nodes:
                match node:
                    case OpExpr('=', [Token(type=TokenType.Name, source_text=name), val_node]):
                        named_args[name] = val_node.evaluate()
                    case OpExpr('!', [Token(type=TokenType.Name, source_text=name)]):
                        flags.add(name)
                    case _:
                        yield node.evaluate()

        return Args(*generate_args(self.nodes), flags=flags, named_arguments=named_args)

    def eval_pattern(self):
        raise NotImplementedError

    def __repr__(self):
        return f"ArgsNode{self.nodes}"


class ParamsNode(ListNode):
    named_params: list[Node]
    any_kwargs: bool | str = False
    def __init__(self, nodes: list[Node], named_params: list[Node], pos: Position = None):
        super().__init__(nodes, pos)
        if named_params and named_params[-1].source_text == '*':
            named_params.pop()
            self.any_kwargs = True
        self.named_params = named_params

    def evaluate(self):
        # TODO: enable param flags
        # def gen_params(nodes) -> tuple[str, Parameter]:
        #     for node in nodes:
        #         match node:
        #             case BindExpr(name=name):
        #                 yield name, node.evaluate()
        #             case Token(type=TokenType.Name, source_text=name):
        #                 yield name, Parameter(AnyMatcher(), name)
        #             case OpExpr("=",
        #                         terms=(BindExpr(name=name) | Token(type=TokenType.Name, source_text=name) as p_node,
        #                                default)):
        #                 param: Parameter
        #                 if isinstance(p_node, Token):
        #                     param = Parameter(AnyMatcher(), name)
        #                 else:
        #                     param = p_node.evaluate()
        #                 param.default = default.evaluate()
        #                 yield name, param
        #             case _:
        #                 param = node.evaluate()
        #                 if not isinstance(param, Parameter):
        #                     param = Parameter(param)
        #                     # raise SyntaxErr(f"Line {Context.line}: Can't parse parameter {node}")
        #                 yield param.binding, param
        def gen_params(nodes) -> tuple[str, Parameter]:
            for node in nodes:
                param = node.eval_pattern()
                yield param.binding, param

        return ParamSet(*(p[1] for p in gen_params(self.nodes)),
                        named_params=dict(gen_params(self.named_params)),
                        kwargs=self.any_kwargs)

    eval_pattern = evaluate


class FieldMatcherNode(ListNode):
    def evaluate(self):
        field_dict = {}

        def generate_matchers(nodes: list[Node]):
            for node in nodes:
                match node:
                    case OpExpr(':',
                                terms=(Token(type=TokenType.Name, source_text=name), patt_node)):
                        match patt_node:
                            case Token(type=TokenType.Name, source_text=binding):
                                field_dict[name] = Parameter(AnyMatcher(), binding)
                            case _:
                                field_dict[name] = patternize(patt_node.evaluate())
                    case _:
                        yield node.evaluate()

        return FieldMatcher(tuple(n.eval_pattern() for n in self.nodes), field_dict)

    def eval_pattern(self):
        return Parameter(self.evaluate())


class FunctionLiteral(ListNode):
    def evaluate(self):
        fn = Function()
        Closure(Block(self.nodes)).execute(fn=fn)
        return fn

    def eval_pattern(self):
        raise NotImplementedError


# expressionize = mathological
# Context.make_expr = expressionize

class OpExpr(Node):
    op: Operator
    terms: tuple[Node, ...]
    def __init__(self, op: Operator, *terms: Node, pos: Position = None):
        assert terms
        self.op = op
        self.terms = terms
        self.pos = pos

    @property
    def sym(self):
        return self.op.text

    __match_args__ = ('sym', 'terms')

    def evaluate(self):
        args: Args = self.op.eval_args(*self.terms)
        return self.op.fn.call(args)

    def __repr__(self):
        if self.op.text == '[':
            return f"{self.terms[0]}[{self.terms[1]}]"
        return f"{self.op}({', '.join(map(str, self.terms))})"


class BindExpr(Node):
    node: Node
    name: str
    quantifier: str
    def __init__(self, node: Node, name: str, quantifier: str = '', pos: Position = None):
        self.node = node
        self.name = name
        self.quantifier = quantifier
        self.pos = pos

    def evaluate(self):
        return Parameter(patternize(self.node.evaluate()), self.name, self.quantifier)

    eval_pattern = evaluate

    def __repr__(self):
        return f"Bind({self.node} {self.name}{self.quantifier})"


class IfElse(Node):
    do: Node
    condition: Node
    alt: Node
    # def __new__(cls, do: Node, cond: Node, alt: Node):
    #     match do:
    #         case OpExpr(op_text) as asgt if op_text == ':' or op_text.endswith('='):
    #             asgt.terms = asgt.terms[0], IfElse(asgt.terms[1], cond, alt)
    #             return asgt
    #         case _:
    #             return super().__new__(cls)

    def __init__(self, do: Node, cond: Node, alt: Node, pos: Position = None):
        # i = next((-1 - i for i, node in enumerate(reversed(other_nodes))
        #           if node.type == TokenType.Else),
        #          None)
        self.do = do
        self.condition = cond
        self.alt = alt
        if pos is None:
            self.pos = self.do.pos + self.alt.pos
        else:
            self.pos = pos

    def evaluate(self):
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.do.evaluate()
        else:
            return self.alt.evaluate()

    def eval_pattern(self):
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.do.eval_pattern()
        else:
            return self.alt.eval_pattern()

    def __repr__(self):
        return f"IfElse({self.do} if {self.condition} else {self.alt})"

class ExprWithBlock(Node):
    header: Node
    block: Block | None
    alt: Node | Block | None = None
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        if pos is None:
            self.pos = header.pos + (blk_nodes[-1].pos if blk_nodes else header.pos)
        else:
            self.pos = pos
        self.header = header
        match blk_nodes:
            case [Block() as blk]:
                self.block = blk
            case [Block() as blk, Block() as alt]:
                self.block = blk
                self.alt = alt
            case [Block() as blk, elif_header, *alt_expr_nodes]:
                self.block = blk
                pos = elif_header.pos + alt_expr_nodes[-1].pos
                self.alt = Conditional(elif_header, alt_expr_nodes, pos)
                # try:
                #     i = next(i for i, node in enumerate(alt_expr_nodes) if isinstance(node, Block))
                # except StopIteration:
                #     raise SyntaxErr(f"Line {blk_nodes[1].line}: missing block after control statement.")
                # self.alt = Conditional(alt_expr_nodes[:i], alt_expr_nodes[i:], blk_nodes[1].line, '')
            case []:
                self.block = None
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
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        super().__init__(header, blk_nodes, pos)
        match header:
            case OpExpr('in', terms):
                self.var, self.iterable = terms
        # for i, node in enumerate(header_nodes):
        #     if node.source_text == 'in':
        #         self.var = mathological(header_nodes[:i])
        #         self.iterable = mathological(header_nodes[i+1:])
        #         break
        # else:
            case _:
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

        var_patt = self.var.eval_pattern()

        for val in iterator:
            if (bindings := var_patt.match(val)) is None:
                raise MatchErr(f"Line {self.line}: "
                               f"pattern '{var_patt}' did not match value {val} in {self.iterable.source_text}")
            for k, v in bindings.items():
                Context.env.assign(k, v)
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


class Command(Node):
    command: str
    def __init__(self, cmd: str, pos: Position = None):
        self.pos = pos
        self.command = cmd


class CommandWithExpr(Command):
    expr: Node
    def __init__(self, cmd: str, expr: Node, pos: Position = None):
        super().__init__(cmd, pos)
        self.expr = expr

    def evaluate(self):
        match self.command:
            case 'exit':
                print('Exiting now')
                exit()
            case 'debug':
                Context.debug = True
                print('Start debugging...')
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


class NamedExpr(Command):
    name: str
    expr: Node
    def __init__(self, cmd: str, name: str, pos: Position = None):
        super().__init__(cmd, pos)
        self.name = name

class FunctionExpr(ExprWithBlock):
    fn_name: str
    body: Block
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        if len(blk_nodes) > 1:
            raise SyntaxErr(f"Line {blk_nodes[1].line}: function blocks cannot have else statements.")
        super().__init__(header, blk_nodes, pos)
        match header:
            case Token(type=TokenType.Name, text=name):
                self.fn_name = name
            case _:
                raise SyntaxErr(f"Line {header.line}: "
                                f"Function/Trait syntax should be `function|trait <fn_name> <block>`.")
        self.body = self.block
        # match nodes.nodes:
        #     case [Token(type=TokenType.Name, text=name), Block() as blk]:
        #         self.fn_name = name
        #         self.body = blk
        #     case _:
        #         raise SyntaxErr(f"Line {self.line}: Function syntax should be `function <fn_name> <block>`.")

    def evaluate(self):
        fn = Context.deref(self.fn_name)
        if isinstance(fn, Function):
            del fn.uninitialized
        else:
            fn = Function(name=self.fn_name)
            Context.env.assign(self.fn_name, fn)
        if self.body is not None:
            Closure(self.body).execute(fn=fn)
        return fn

    def __repr__(self):
        return f'Function({self.fn_name}, {self.body})'

class TraitExpr(FunctionExpr):
    # trait_name: str
    # body: Block
    # def __init__(self, cmd: str, nodes: Concrete, pos: Position = None):
    #     super().__init__(cmd, pos)
    #     match nodes.nodes:
    #         case [Token(type=TokenType.Name, text=name), Block() as blk]:
    #             self.trait_name = name
    #             self.body = blk
    #         case _:
    #             raise SyntaxErr(f"Line {self.line}: Trait syntax should be `trait <trait_name> <block>`.")

    def evaluate(self):
        trait = Context.deref(self.fn_name)
        if isinstance(trait, Trait) and hasattr(trait, 'uninitialized'):
            del trait.uninitialized
        else:
            trait = Trait(name=self.fn_name)
            Context.env.assign(self.fn_name, trait)
        if self.body is not None:
            Closure(self.body).execute(fn=trait)
        return trait

    def __repr__(self):
        return f'Trait({self.fn_name}, {self.body})'

class TableExpr(FunctionExpr):
    table_name: str
    traits: list[Node]
    body: Block = None
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        super().__init__(header, blk_nodes, pos)
        match header:
            case Token(type=TokenType.Name, text=name):
                self.table_name = name
                self.traits = []
            case OpExpr('&', [Token(type=TokenType.Name, text=name), FieldMatcherNode(nodes=trait_nodes)]):
                self.table_name = name
                self.traits = list(trait_nodes)
            case _:
                raise SyntaxErr(f"Line {self.line}: Table syntax should be `table <table_name> (<trait>)* <block>`.\n"
                                f"Eg, `table MyList(list, seq, iter)`")

    # def __init__(self, cmd: str, tbl_name: str, nodes: list[Node], pos: Position = None):
    #     super().__init__(cmd, tbl_name, pos)
    #     match nodes.nodes:
    #         case [Token(type=TokenType.Name, text=name), Concrete(nodes=[*trait_nodes, Block() as blk])]:
    #             self.body = blk
    #         # case [Token(type=TokenType.Name, source_text=name), *trait_nodes]:
    #         #     pass
    #         case _:
    #             raise SyntaxErr(f"Line {self.line}: Table syntax should be `table <table_name> (<trait>)* <block>`.\n"
    #                             f"Eg, `table MyList(list, seq, iter)`")
    #     self.table_name = name
    #     match trait_nodes:
    #         case [Token(), Concrete(type=ExprType.Parens, nodes=items), Token()]:
    #             self.traits = items
    #         case []:
    #             self.traits = []
    #         case _:
    #             raise SyntaxErr(f"Line {self.line}: Table syntax should be `table <table_name> (<trait>)* <block>`.\n"
    #                             f"Eg, `table MyList(list, seq, iter)`")
        # for i in range(0, len(trait_nodes), 2):
        #     match trait_nodes[i:i+2]:
        #         case [Token(source_text='@'), Token(type=TokenType.Name, source_text=name)]:
        #             self.traits.append(name)
        #         case _:
        #             raise SyntaxErr(f"Line {self.line}: "
        #                             f"Table traits must be listed like `@trait_name` separated only by whitespace.")

    def evaluate(self):
        # table = ListTable(name=self.table_name)
        # Context.env.assign(self.table_name, table)
        table = Context.deref(self.table_name)
        if isinstance(table, Table):
            del table.uninitialized
        else:
            table = ListTable(name=self.table_name)
            Context.env.assign(self.table_name, table)

        def gen_traits():
            for node in self.traits:
                trait = node.evaluate()
                if not isinstance(trait, Trait):
                    raise TypeErr(f"Line: {Context.line}: expected trait, but '{node}' is {repr(trait)}")
                yield trait

        table.traits = (*table.traits, *gen_traits())
        # for trait, name in zip(table.traits, self.traits):
        #     if not isinstance(trait, Trait):
        #         raise TypeErr(f"Line: {Context.line}: expected trait, but '{name}' is {repr(trait)}")
        if self.body is not None:
            Closure(self.body).execute(fn=table)
        table.integrate_traits()
        return table

    def __repr__(self):
        return f'Table({self.table_name} with {", ".join(map(str, self.traits))}, {self.body})'


class SlotExpr(NamedExpr):
    # slot_name: str
    # slot_type: Node
    default: None | Node | Block | str = None
    def __init__(self, cmd: str, field_name: str, node: Node, pos: Position = None):
        super().__init__(cmd, field_name, pos)
        match node:
            case OpExpr('='|':', terms):
                self.field_type, self.default = terms
            case _:
                self.field_type = node
        # other_nodes = nodes
        # for i, node in enumerate(other_nodes):
        #     if node.source_text == '=':
        #         self.field_type = mathological(other_nodes[:i])
        #         self.default = mathological(other_nodes[i+1:])
        #         break
        #     if node.source_text == ':':
        #         self.field_type = mathological(other_nodes[:i])
        #         if i+2 != len(other_nodes):
        #             raise SyntaxErr(f"Line {self.line}: Slot is missing default block after `:`.")
        #         self.default = other_nodes[i+1]
        #         break
        #     if node.source_text == '?' and i + 1 == len(other_nodes):
        #         self.default = EmptyExpr()
        # else:
        #     self.field_type = mathological(other_nodes)

    def evaluate(self):
        # slot_type = patternize(self.slot_type.evaluate())
        slot_type = self.field_type.eval_pattern()
        # match patternize(self.slot_type.evaluate()):
        #     case ParamSet(parameters=(Parameter(matcher=slot_type),)):
        #         pass
        #     case _:
        #         raise TypeErr(f"Line {Context.line}: Invalid type: {self.slot_type.evaluate()}.  "
        #                       f"Expected value, table, trait, or single-parameter pattern.")
        match self.default:
            case None:
                default = getattr(slot_type, 'default', None)
            case Block() as blk:
                default = Closure(blk)
            case Node():
                default = self.default.evaluate()
                # default = Function({Parameter(AnyMatcher(), 'self'): default_value})
            case _:
                raise ValueError("Unexpected default")
        if default is not None:
            default = Function({Parameter(AnyMatcher(), 'self'): default})
        slot = Slot(self.name, slot_type, default)
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
        return f"Slot({self.name}, {self.field_type}{default})"

class FormulaExpr(NamedExpr):
    formula_name: str
    formula_type: Node
    block: Block
    def __init__(self, cmd: str, field_name: str, nodes: list[Node], pos: Position = None):
        super().__init__(cmd, field_name, pos)
        match nodes:
            case [Token(type=TokenType.Name, text=name), *patt_nodes, Token(text=':'), Block() as blk]:
                self.formula_name = name
                self.formula_type = mathological(patt_nodes)
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

class SetterExpr(NamedExpr):
    field_name: str
    param_nodes: ParamsNode
    block: Block
    def __init__(self, cmd: str, field_name: str, nodes: list[Node], pos: Position = None):
        super().__init__(cmd, field_name, pos)
        match nodes:
            case [Token(type=TokenType.Name, text=name),  # Token(source_text='.'),
                  ParamsNode(nodes=[Node() as param_node]) as param_list,
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
    params: ParamsNode
    return_type: Node | None
    block: Block
    def __init__(self, cmd: str, nodes: Node, pos: Position = None):
        super().__init__(cmd, pos)
        match nodes:
            case [ListNode(nodes=params), *patt_nodes, Token(text=':'), Block() as blk]:
                self.params = params
                self.return_type = mathological(patt_nodes)
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Opt syntax is: `opt [param1, param2, ...] <type_expr>?: <block>`."
                                f"Eg, `opt [int i, int j] str | blank : ...`")

    def evaluate(self):
        match self.return_type.eval_pattern():
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

class Declaration(NamedExpr):
    var_name: str
    # context_expr: None | Expression = None
    value_expr: None | Node = None

    def __init__(self, cmd: str, var_name: str, nodes: Node, pos: Position = None):
        super().__init__(cmd, var_name, pos)
        match nodes:
            case [Token(type=TokenType.Name, text=name)]:
                self.var_name = name
            case [Token(type=TokenType.Name, text=name), Token(text='='), *other_nodes]:
                self.var_name = name
                if other_nodes:
                    self.value_expr = mathological(other_nodes)
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
        value = self.value_expr.evaluate() if self.value_expr else BuiltIns['blank']
        Context.env.locals[self.var_name] = value
        return value

    def eval_pattern(self):
        value = self.value_expr.evaluate() if self.value_expr else BuiltIns['blank']
        Context.env.locals[self.var_name] = value
        return Parameter(AnyMatcher(), self.var_name, default=value)

class VarExpr(Declaration):
    def evaluate(self):
        value = self.value_expr.evaluate() if self.value_expr else BuiltIns['blank']
        Context.env.vars[self.var_name] = value
        return value

    def eval_pattern(self):
        value = self.value_expr.evaluate() if self.value_expr else BuiltIns['blank']
        Context.env.vars[self.var_name] = value
        return Parameter(AnyMatcher(), self.var_name, default=value)

class EmptyExpr(Node):
    def __init__(self, pos: Position = None):
        self.pos = pos

    def evaluate(self):
        return BuiltIns['blank']

    def __repr__(self):
        return "EmptyExpr()"


EXPRMAP = {
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

# def py_eval(code):
#     return piliize(eval(code.value))
#
# class SingleNode(Expression):
#     def __init__(self, node: Node, line: int | None, source: str):
#         super().__init__(line, source)
#         self.node = node
#     def evaluate(self):
#         return eval_node(self.node)
#
#
# def eval_node(node: Node) -> Record:
#     match node:
#         case Expression() as statement:
#             return expressionize(statement).evaluate()
#         case Token() as tok:
#             return eval_token(tok)
#         case Block() as block:
#             return Closure(block).execute(fn=Function())
#         case ListNode(list_type=list_type, items=items):
#             match list_type:
#                 case ListType.List:
#                     return py_value(list(map(eval_node, items)))
#                 case ListType.Args:
#                     return eval_args_list(items)
#                 # case ListType.Params:
#                 #     return ParamSet(*map(make_param, items))
#                 case ListType.Tuple:
#                     return py_value(tuple(map(eval_node, items)))
#                 case ListType.Function:
#                     fn = Function()
#                     Closure(Block(items)).execute(fn=fn)
#                     return fn
#                 case _:
#                     raise NotImplementedError
#         case StringNode(nodes=nodes):
#             return py_value(''.join(map(eval_string_part, nodes)))
#     raise ValueError(f'Could not evaluate node {node} at line: {node.pos}')
#
# def eval_string_part(node: Node) -> str:
#     if node.type == TokenType.StringPart:
#         return eval_string(node.source_text)
#     return BuiltIns['str'].call(node.evaluate()).value
#     # if isinstance(node, Expression):
#     #     return BuiltIns['str'].call(node.evaluate()).value
#     # raise ValueError('invalid string part')
# def eval_string(text: str):
#     value = ""
#     for i in range(len(text)):
#         if i and text[i-1] == '\\':
#             continue
#         if text[i] == '\\':
#             match text[i+1]:
#                 case 'n':
#                     value += '\n'
#                 case 'r':
#                     value += '\r'
#                 case 't':
#                     value += '\t'
#                 case 'b':
#                     value += '\b'
#                 case 'f':
#                     value += '\f'
#                 case c if c in "'\"{\\":
#                     # assume char is one of: ', ", {, \
#                     value += c
#                 case c:
#                     raise SyntaxErr(f"Unrecognized escape character: '{c}'")
#         else:
#             value += text[i]
#     return value
#
# def precook_args(op: Operator, lhs: Node, rhs: Node) -> list[Record]:
#     if op.binop and lhs and rhs:
#         args = [lhs.evaluate(), rhs.evaluate()]
#         # args = [expressionize(lhs).evaluate(), expressionize(rhs).evaluate()]
#     elif op.prefix and rhs or op.postfix and lhs:
#         args = [(rhs or lhs).evaluate()]
#         # args = [expressionize(rhs or lhs).evaluate()]
#     else:
#         raise ArithmeticError("Mismatch between operator type and operand positions.")
#     return args
#
#
# # Operator.eval_args = precook_args
# # I think this has been replaced by a simpler function: Operator.eval_args
#
#
# def eval_token(tok: Token) -> Record:
#     s = tok.source_text
#     match tok.type:
#         case TokenType.Singleton:
#             return py_value(SINGLETONS[s])
#         case TokenType.Number:
#             return py_value(read_number(s, Context.settings['base']))
#         case TokenType.StringLiteral:
#             return py_value(s.strip("`"))
#         case TokenType.Name:
#             if s == 'self':
#                 return Context.deref(s, Context.env.caller)
#             return Context.deref(s)
#         case _:
#             raise Exception("Could not evaluate token", tok)
#
#
# def eval_string(text: str):
#     value = ""
#     for i in range(len(text)):
#         if i and text[i-1] == '\\':
#             continue
#         if text[i] == '\\':
#             match text[i+1]:
#                 case 'n':
#                     value += '\n'
#                 case 'r':
#                     value += '\r'
#                 case 't':
#                     value += '\t'
#                 case 'b':
#                     value += '\b'
#                 case 'f':
#                     value += '\f'
#                 case c if c in "'\"{\\":
#                     # assume char is one of: ', ", {, \
#                     value += c
#                 case c:
#                     raise SyntaxErr(f"Unrecognized escape character: '{c}'")
#         else:
#             value += text[i]
#     return value
#
#
# def eval_args_list(statements: list[Expression]) -> Args:
#     pos_args = []
#     named_args = {}
#     flags = set()
#     for stmt in statements:
#         match stmt.nodes:
#             case [Token(type=TokenType.Name, source_text=name), Token(source_text='='), *expr_nodes]:
#                 named_args[name] = expressionize(expr_nodes).evaluate()
#             case [Token(source_text='!'), Token(type=TokenType.Name, source_text=name)]:
#                 flags.add(name)
#             case nodes:
#                 pos_args.append(expressionize(nodes).evaluate())
#
#     return Args(*pos_args, flags=flags, **named_args)


# class Expressionizer:
#     node: Node
#     def __init__(self, cst: Concrete):
#         self.cst = cst
#         self.ops: list[list[Operator | str]] = []
#         # second (str) element of each list in ops represents the fixity: prefix, binop, postfix
#         self.terms: list[Node] = []
#         self.unary_state = True
#
#         for node in self.prepare_nodes(cst.nodes):
#             # In the unary state, we're expecting either a unary operator or an operand or grouping parenthesis (or others).
#             # otherwise, we are in binary state expecting binary operators, or close parenthesis (or open paren).
#             if self.unary_state:
#                 self.next_operand(node)
#             else:
#                 self.next_operator(node)
#         self.reduce()
#         assert len(self.terms) == 1 and len(self.ops) == 0
#         expr = self.terms[0]
#         if not expr.pos:
#             expr.pos = cst.pos
#         self.node = expr
#
#     def prepare_nodes(self, nodes: list[Concrete | Token]):
#
#         i = 0
#         while i < len(nodes):
#             node = nodes[i]
#             match node.type:
#                 case ExprType.Group:
#                     inner = node.nodes[1]
#                     if inner:
#                         yield inner.abstract()
#                     else:
#                         yield TupleLiteral((), node.pos)
#                 case ExprType.Command:
#                     command = node.nodes[0].source_text
#                     Cmd = EXPRMAP.get(command, CommandWithExpr)
#                     yield Cmd(command, node.nodes[1].abstract(), node.pos)
#                 case ExprType.Assignment:
#                     left, op, right = node.nodes
#                     left: Concrete
#                     right: Concrete
#                     op_sym = op.text
#                     if op_sym == ':':
#                         match left.nodes:
#                             case [*left_nodes, Concrete(type=ExprType.Brackets, nodes=items, pos=pos)]:
#                                 lhs = TargetExpr(Concrete(left_nodes).abstract(),
#                                                  ParamsNode(tuple(n.abstract() for n in items), pos),
#                                                  left.pos)
#                             case _:
#                                 lhs = left.abstract()
#                     else:
#                         lhs = left.abstract()
#                     yield OpExpr(Op[op.text], lhs, right.abstract(), pos=node.pos)
#                 # case ExprType.Parens:
#                 #     items = tuple(n.abstract() for n in node.nodes[1:-1])
#                 #     if i and nodes[i-1].type not in (TokenType.Operator, TokenType.Command, TokenType.Keyword):
#                 #         assert not self.unary_state
#                 #         yield Token('&', TokenType.Operator)
#                 #         yield FieldMatcherNode(items, node.pos)
#                 #     else:
#                 #         yield TupleLiteral(items, node.pos)
#                 case ExprType.Parens:
#                     assert not self.unary_state  # if this was in unary state, it should have become ExprType.Group
#                     if len(node) == 2:  # only nodes are brackets
#                         pass  # don't even bother yielding any nodes because an empty field matcher has no effect
#                     if len(node) > 3:
#                         raise SyntaxErr(node, "Semicolons not allowed in Field Matchers")
#                     items = (n.abstract() for n in node.nodes[1].nodes)
#                     yield Token('&', TokenType.Operator)
#                     yield FieldMatcherNode(tuple(items), node.pos)
#                 case ExprType.Brackets:
#                     next_node = nodes[i+1] if i + 1 < len(nodes) else None
#                     if next_node and next_node.type == TokenType.Operator and next_node.text in (':', '=>'):
#                         yield Token('[params]', TokenType.Operator, node.pos.pos)
#                         match node.nodes:
#                             case [_, _]:
#                                 ordered_params, named_params = [], []
#                             case [_, Concrete(nodes=ordered_params), _]:
#                                 named_params = []
#                             case [_, Concrete(nodes=ordered_params), Token(text=';'), Concrete(nodes=named_params), _]:
#                                 pass
#                             case _:
#                                 raise SyntaxErr(node.pos, "Too many semicolons in parameter list.")
#                         yield ParamsNode(tuple(p.abstract() for p in ordered_params),
#                                          list(p.abstract() for p in named_params), node.pos)
#                     else:
#                         items = tuple(n.abstract() for n in node.nodes[1:-1])
#                         if self.unary_state:
#                             yield ListLiteral(items, node.pos)
#                         else:
#                             yield Token('[call]', TokenType.Operator, node.pos.pos)
#                             yield ArgsNode(items, node.pos)
#                 case ExprType.Braces:
#                     match node.nodes:
#                         case [_, _]:
#                             items = []
#                         case [_, Concrete(nodes=items), _]:
#                             pass
#                         case _:
#                             raise AssertionError(node.pos, "umm... brace nodes are only supposed to have up to 3 nodes")
#                     items = tuple(n.abstract() for n in items)
#                     yield FunctionLiteral(items, pos)
#                 case _:
#                     yield node.abstract()
#             i += 1
#         yield None
#
#     def mathological(self, nodes: list[Node]):
#         self.ops: list[list[Operator | str]] = []
#         # second (str) element of each list in ops represents the fixity: prefix, binop, postfix
#         self.terms: list[Node] = []
#         self.unary_state = True
#
#         def loop_nodes():
#             for node_or_list in cst:
#                 if isinstance(node_or_list, Node):
#                     yield node_or_list
#                 else:
#                     yield AST(node_or_list)
#             yield None
#
#         for node in loop_nodes():
#             # In the unary state, we're expecting either a unary operator or an operand or grouping parenthesis (or others).
#             # otherwise, we are in binary state expecting binary operators, or close parenthesis (or open paren).
#             if self.unary_state:
#                 self.next_operand(node)
#             else:
#                 self.next_operator(node)
#         self.reduce()
#         assert len(self.terms) == 1 and len(self.ops) == 0
#         expr = self.terms[0]
#         expr.line, expr.source_text = line, src
#         return expr
#
#     def reduce(self, prec: int = -1):
#         terms, ops = self.terms, self.ops
#         while ops:
#             op, fixity, op_pos = ops[-1]
#             op_prec = getattr(op, fixity)
#             assert op_prec is not None
#             if op_prec and (op_prec > prec or op.associativity == 'left' and op_prec == prec):
#                 ops.pop()
#                 t1 = terms.pop()
#                 t_last = terms[-1] if terms else None
#                 if isinstance(t_last, OpExpr) and t_last.op == op and op.chainable:
#                     t_last.terms += t1,
#                 elif fixity == 'binop':
#                     t2 = terms.pop()
#                     pos = Concrete([t2, t1]).pos
#                     terms.append(OpExpr(op, t2, t1, pos=pos))
#                 elif fixity == 'postfix' and isinstance(t1, BindExpr):
#                     t1.quantifier = op.text
#                     terms.append(t1)
#                 else:
#                     pos = Position(op_pos.pos, op_pos.start_index, t1.pos.stop_index)
#                     terms.append(OpExpr(op, t1, pos=pos))
#             else:
#                 return
#
#     def next_operand(self, node: Node):
#         terms, ops = self.terms, self.ops
#         match node:
#             case Token(type=TokenType.Operator, text=op_text, pos=op_pos):
#                 try:
#                     op = Op[op_text]
#                 except KeyError:
#                     raise OperatorErr(f"Line {node.line}: unrecognized operator: {op_text}")
#                 if op.prefix:
#                     ops.append([op, 'prefix', op_pos])
#                 elif op.binop and ops and ops[-1][0].postfix:
#                     ops[-1][1] = 'postfix'
#                     ops.append([op, 'binop', op_pos])
#                 elif ops:
#                     raise OperatorErr(f"Line {node.line}: expected term or prefix operator after {ops[-1][0]}.  "
#                                       f"Instead got {op}.")
#                 else:
#                     raise OperatorErr(node.pos, f"{op} is not prefix operator.")
#             case Node():
#                 terms.append(node)
#                 self.unary_state = False
#             case _:
#                 if ops:
#                     op, fixity, op_pos = ops.pop()
#                     if op.postfix:
#                         self.reduce(op.postfix)
#                         ops.append([op, 'postfix', op_pos])
#                     else:
#                         raise OperatorErr(f"Line {node.line}: expected operand after {ops[-1]}")
#                 else:
#                     raise OperatorErr(node, "expected operand.")
#                 # if ops and ops[-1][0].postfix:
#                 #     ops[-1][1] = 'postfix'
#                 # elif ops:
#                 #     raise OperatorErr(f"Line {line}: expected operand after {ops[-1]}")
#                 # else:
#                 #     raise OperatorErr(f"Line")
#
#     def next_operator(self, node: Node):
#         match node:
#             case Token(type=TokenType.Operator, text=op_text, pos=op_pos):
#                 op = Op[op_text]
#                 if op.binop:
#                     self.reduce(op.binop)
#                     self.ops.append([op, 'binop', op_pos])
#                     self.unary_state = True
#                 elif op.postfix:
#                     self.reduce(op.postfix)
#                     self.ops.append([op, 'postfix', op_pos])
#                 else:
#                     raise OperatorErr(f"Line {node.line}: Prefix {op} used as binary/postfix operator.")
#             case Token(type=TokenType.Name, source_text=name):
#                 self.reduce(3)
#                 t = self.terms.pop()
#                 pos = t.pos
#                 self.terms.append(BindExpr(t, name, pos=pos))
#             case None:
#                 pass
#             case _:
#                 raise OperatorErr(f"Line {node.line}: Expected operator.  Got {node}")


if __name__ == "__main__":
    pass