# from stub import *
from enum import Enum, EnumMeta
import math
from . import state
from .state import BuiltIns, Op
from .utils import OperatorErr

print(f'loading {__name__}.py')

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
    # For = 'for'
    # While = 'while'
    # Local = 'local'
    # Var = 'var'
    Return = 'return'
    Break = 'break'
    Continue = 'continue'
    Exit = 'exit'
    Debug = 'debug'
    Import = 'import'
    Label = 'label'
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
    To = 'to'
    By = 'by'
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

    def __add__(self, other):
        if other.stop_index is None:
            # print(f"SyntaxWarning {self.pos}: Missing stop index of added position.")
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
    type = TokenType.Unknown
    pos: Position = None
    source_text: str

    @property
    def line(self):
        return self.pos.ln

    def evaluate(self):
        raise NotImplementedError

    def abstract(self, unary=True):
        if unary:
            return self
        else:
            raise NotImplementedError(f"Cannot abstract {self} as operator")

    @property
    def source_text(self):
        return state.source_code[self.pos.slice()]

    def eval_pattern(self, name_as_any=False):
        raise NotImplementedError('implement this in interpreter.py')
        # return patternize(self.evaluate())


class Token(Node):
    text: str
    __match_args__ = ('type',)
    def __init__(self, text: str,
                 type: TokenType = None,
                 pos: tuple[int, int | None] = None,
                 start: int = None, stop: int = None):
        self.text = text
        if pos:
            self.pos = Position(pos, start, stop)
        self.type = type or TokenType.map.get(text, TokenType.Name)

    def evaluate(self):
        raise NotImplementedError('implement this in interpreter.py')
        # s = self.text
        # match self.type:
        #     case TokenType.Singleton:
        #         return py_value(SINGLETONS[s])
        #     case TokenType.Number:
        #         return py_value(read_number(s, state.settings['base']))
        #     case TokenType.StringLiteral:
        #         return py_value(s.strip("`"))
        #     case TokenType.Name:
        #         if s == 'self':
        #             return state.deref(s, state.env.caller)
        #         return state.deref(s)
        # raise NotImplementedError(f"Line {self.line}: Could not evaluate token", self)

    def eval_pattern(self, name_as_any=False):
        raise NotImplementedError('implement this in interpreter.py')
        # if name_as_any and self.type is TokenType.Name:
        #     return Parameter(AnyMatcher(), self.text)
        # return patternize(self.evaluate())

    def __str__(self):
        return self.text or str(self.type)

    def __repr__(self):
        data = [repr(self.source_text or self.text)]
        if self.pos:
            data.extend(map(str, self.pos.__dict__.values()))
        return f"Token<{self.type.name}>({', '.join(data)})"

class ListNode(Node):
    nodes: list[Node]
    def __init__(self, nodes: list[Node], pos: Position = None):
        self.nodes = nodes
        self.pos = pos  # or Concrete(nodes).pos

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
        raise NotImplementedError("Use Block.execute instead.")
        # self.execute()
        # return BuiltIns['blank']

    def execute(self):
        raise NotImplementedError('implemented in interpreter.py')

    def __repr__(self):
        if not self.statements:
            return 'Block[empty]'
        elif len(self.statements) == 1:
            return f"Block[{repr(self.statements[0])}]"
        else:
            return f"Block[{len(self.statements)} statements]"


def default_op_fn(*args):
    raise OperatorErr(f"Line {state.line}: Operator has no function.")


# default_op_fn = Function({Parameter(AnyMatcher(), None, "*"): default_op_fn})

class Operator:
    # fn: Function = default_op_fn
    def __init__(self,
                 text,
                 fn=None,
                 prefix=None, postfix=None, binop=None,
                 associativity='left',
                 chainable=False):
        Op[text] = self
        self.text = text
        # self.precedence = precedence
        if fn:
            if not fn.name:
                fn.name = text
            BuiltIns[text] = fn
            self.fn = fn
        self.associativity = associativity  # 'right' if 'right' in flags else 'left'
        self.prefix = prefix  # 'prefix' in flags
        self.postfix = postfix  # 'postfix' in flags
        self.binop = binop  # 'binop' in flags
        # self.ternary = ternary
        self.chainable = chainable

        assert self.binop or self.prefix or self.postfix

    @staticmethod
    def eval_args(*terms):
        raise NotImplementedError('This function is defined in interpreter.py')
        # return Args(*(t.evaluate() for t in terms if not isinstance(t, EmptyExpr)))

    def __repr__(self):
        return self.text


Operator(';', binop=1)
Operator(':', binop=2, associativity='right')
Operator('=', binop=2, associativity='right')
for op in ('+', '-', '*', '/', '//', '**', '%', '&', '|', '&&', '||'):
    Operator(op+'=', binop=2, associativity='right')
Operator('??=', binop=2, associativity='right')
Operator('=>', binop=2)
Operator(',', binop=2, postfix=2, chainable=True)
# Operator('if', binop=3, ternary='else')
Operator('??', binop=4)
Operator('or', binop=5, chainable=True)
Operator('||', binop=5, chainable=True)
Operator('and', binop=6, chainable=True)
Operator('&&', binop=6, chainable=True)
Operator('not', prefix=7)
Operator('in', binop=8)
Operator('==', binop=9, chainable=True)
Operator('!=', binop=9, chainable=True)
# Operator('!~', binop=9, chainable=False)
Operator('is', binop=9, chainable=False)
Operator('is not', binop=9, chainable=False)
Operator('|', binop=10, chainable=True)
Operator('<', binop=11, chainable=True)
Operator('>', binop=11, chainable=True)
Operator('<=', binop=11, chainable=True)
Operator('>=', binop=11, chainable=True)
Operator('>>', binop=11, prefix=11, postfix=11)
Operator('to', binop=11, prefix=11, postfix=11)
Operator('by', binop=11)
Operator('+', binop=12, prefix=14, postfix=3)
Operator('-', binop=12, prefix=14)
Operator('*', binop=13, prefix=13, postfix=3)
Operator('/', binop=13, chainable=False)
Operator('//', binop=13, chainable=False)
Operator('%', binop=13, chainable=False)
Operator('**', binop=14, chainable=False, associativity='right')
Operator('^', binop=14, chainable=False, associativity='right')
Operator('?', postfix=15)
Operator('call?', binop=16)
Operator('has', binop=15, prefix=15)
Operator('&', binop=15)
Operator('~', binop=15, prefix=15)
Operator('@', binop=3, prefix=15)
Operator('!', prefix=15)
Operator('.', binop=16, prefix=3)
Operator('.?', binop=16)
Operator('..', binop=16)
Operator('..?', binop=16)
Operator('[', binop=16)
Operator('var', prefix=17)
Operator('local', prefix=17)
