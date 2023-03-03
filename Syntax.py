import math
from enum import Enum, EnumMeta
import re

op_char_patt = r'[:.<>?/~!@#$%^&*+=|-]'

def contains(cls, item):
    if isinstance(item, cls):
        return item.name in cls._member_map_
    else:
        return item in cls._value2member_map_


EnumMeta.__contains__ = contains


class TokenType(Enum):
    Unknown = '?'
    String = 'string'
    Number = 'number'
    Singleton = 'singleton'
    Operator = 'operator'
    OptionSet = ':='
    Command = 'command'  # if, else, for, while, return, break, continue
    Keyword = 'keyword'  # in, with, ...
    Type = "type"
    Name = 'name'
    PatternName = 'pattern name'
    GroupStart = '('
    GroupEnd = ')'
    ListStart = '['
    ListEnd = ']'
    Comma = ','


class Commands(Enum):
    Print = 'print'
    # If = 'if'
    Else = 'else'
    For = 'for'
    While = 'while'
    Return = 'return'
    Break = 'break'
    Continue = 'continue'
    Exit = 'exit'
    Debug = 'debug'


class BasicType(Enum):
    none = 'none'
    Boolean = 'bool'
    Integer = 'int'
    Float = 'float'
    String = 'str'
    Function = 'fn'
    List = 'list'
    Type = 'type'
    Pattern = 'pattern'
    Name = 'NAMEPATTERN'
    Any = 'any'


class OptionType(Enum):
    Function = ":"
    SetValue = "="
    Alias = ":="
    PlusEquals = '+='
    MinusEquals = '-='
    MultEquals = '*='
    DivEquals = '/='
    ModEquals = '%='
    AndEquals = "&="
    OrEquals = "|="
    NullEquals = '?='


class Singletons(Enum):
    none = 'none'
    true = 'true'
    false = 'false'
    inf = 'inf'


class KeyWords(Enum):
    In = 'in'
    And = 'and'
    Or = 'or'
    Is = 'is'
    Not = 'not'
    Of = 'of'
    If = 'if'
    Else = 'else'


class MatchPatternType(Enum):
    Value = 'value'
    Class = 'class'
    SubClass = 'sub_class'


def token_mapper(item: str) -> TokenType:
    return TokenType._value2member_map_.get(item, TokenType.Unknown)
def command_mapper(item: str) -> Commands:
    return Commands._value2member_map_.get(item)
def option_type_mapper(item: str) -> OptionType:
    return OptionType._value2member_map_.get(item, None)
def singleton_mapper(item: str) -> Singletons:
    return Singletons._value2member_map_.get(item, None)
singletons = {'none': None, 'false': False, 'true': True, 'inf': math.inf}
def keyword_mapper(item: str) -> KeyWords:
    return KeyWords._value2member_map_.get(item, None)
def match_pattern_type_mapper(item: str) -> MatchPatternType:
    return MatchPatternType._value2member_map_.get(item, None)
def type_mapper(item: str | type) -> BasicType:
    if isinstance(item, str):
        return BasicType._value2member_map_.get(item, None)


class Node:
    """
    Constituent parts of statement
    One of: Token, List, Statement, Block
    """
    pos: tuple[int, int]
    type = TokenType.Unknown
    source_text: str


class Token(Node):
    """
    source: str
    type: TokenType
    """

    def __init__(self, text: str, pos: tuple[int, int] = (-1, -1)):
        self.pos = pos[0]+1, pos[1]+1
        self.type: TokenType
        self.source_text = text

        if re.match(r'["\'`]', text):
            self.type = TokenType.String
        elif re.fullmatch(r'-?\d+(\.\d*)?', text):
            self.type = TokenType.Number
        elif re.match(op_char_patt, text) or text == 'if':
            if text == ':' or text.endswith('='):
                self.type = TokenType.OptionSet
            else:
                self.type = TokenType.Operator
        elif text in Commands:
            self.type = TokenType.Command
        elif text.lower() in Singletons:
            self.source_text = text.lower()
            self.type = TokenType.Singleton
        elif text in BasicType:
            self.type = TokenType.Type
        elif text in KeyWords:
            self.type = TokenType.Keyword
        elif re.fullmatch(r'\w+', text):
            self.type = TokenType.Name
        else:
            self.type = token_mapper(text)

    def __str__(self):
        return self.source_text

    def __repr__(self):
        return f"<{self.source_text}:{self.type.name}>"


class Line:
    """
    line_number: int    line number in file
    source_text: str    original text from file
    text: str           trimmed text without comments
    indent: int
    tokens: list[Token]
    """

    def __init__(self, line: str, number: int):
        self.source_text = line
        self.line_number = number
        line = re.sub(r'\s{4}', '\t', line)
        tabs = re.match(r'^\t+', line)
        self.indent = tabs.end() if tabs else 0
        line = re.sub(r'(#|//).*', '', line)
        self.text = line.strip()
        self.tokens: list[Token] = []
        # self.read_tokens(ast)
        # tokens are added by the Tokenizer

    def __len__(self):
        return len(self.text)

    def split2(self, pattern):
        # returns -> tuple[list[Token], Token, list[Token]]:
        i, tok = 0, None
        for i, tok in enumerate(self.tokens):
            if re.search(pattern, tok.source_text):
                break
        first = self.tokens[0:i]
        final = self.tokens[i + 1:] if i + 1 < len(self.tokens) else []
        return first, tok, final

    def split(self, pattern):
        # returns -> list[list[Token]]
        token_array: list[list[Token]] = []
        start, i, tok = 0, 0, None
        for i, tok in enumerate(self.tokens):
            if re.search(pattern, tok.source_text):
                token_array.append(self.tokens[start:i])
                start = i + 1
        if start < len(self.tokens):
            token_array.append(self.tokens[start:i])
        return token_array

    def __repr__(self):
        # tok_strings = list(map(repr, self.toks))
        if not self.tokens:
            return f'Line {self.line_number}: ' + '\t' * self.indent + self.text
        toks = list(map(repr, self.tokens))
        return f'Line {self.line_number}: ' + '\t' * self.indent + ' '.join(toks)


class NonTerminal(Node):
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes
        self.pos = nodes[0].pos if nodes else (-1, -1)
        self.source_text = ' '.join(n.source_text for n in nodes)


class Statement(NonTerminal):
    """
    The parts that make up a block, executed in order
    Consists of:
    - Expressions
    - Blocks
    - match-statements
    """
    # nodes: list[MatchSet | _Expression | Block]
    nodes: list[Node]

    def __init__(self, nodes: list[Node]):
        super().__init__(nodes)

    def __repr__(self):
        return ' '.join(repr(node) for node in self.nodes)


class Block(NonTerminal):
    """
    a container for executables (statements, e
    representing the lines of code to put into a function
    """
    statements: list[Statement]

    def __init__(self, nodes: list[Statement], indent=0):
        super().__init__(nodes)
        self.statements = nodes

    def __repr__(self):
        if not self.statements:
            return '[empty]'
        elif len(self.statements) == 1:
            return f"[{repr(self.statements[0])}"
        else:
            return f"[{len(self.statements)} statements]"


class List(NonTerminal):
    """
    [parameter set, or arg set]
    """
    nodes: list[Statement]
    def __init__(self, items: list[Statement]):
        super().__init__(items)

    def __repr__(self):
        return repr(self.nodes)  # f"[{', '.join(repr(node) for node in self.nodes)}]"


if __name__ == "__main__":
    print(TokenType._value2member_map_.get('?', 'None Found'))
    print(repr(Token(',', (0, 0))))



