import math
from fractions import Fraction

from Syntax import Block
from Env import *
from Expressions import Expression

val_types = None | bool | int | float | str | Pattern | list[Function]

class Value(Function):
    value: val_types
    def __init__(self, value: val_types): ...
    def set_value(self, new_value: Value | val_types) -> Value: ...
    def set_type(self, type: Function = None) -> Function: ...
    def is_null(self) -> bool: ...
    def not_null(self) -> bool: ...
    def clone(self) -> Value: ...

class Parameter:
    inverse: bool = False
    pattern: Pattern
    name: str | None
    quantifier: str   #  "+" | "*" | "?" | ""
    count: tuple[int, int|float]
    optional: bool
    multi: bool
    def __init__(self,
                 pattern: Pattern | Function | str,
                 name: str = None,
                 quantifier: str = "",
                 inverse = False): ...
    def specificity(self) -> int: ...
    def match_score(self, value: Function) -> int | float: ...

# score is out 7560, this number is 3*2520 (the smallest int divisible by all integers up to 10)
def match_score(val: Function, param: Parameter) -> int: ...

Guard = Function | callable

class Pattern:
    """
    A Pattern is like a regex for types and parameters; it can match one very specific type, or even
    one specific value, or it can match a type on certain conditions (e.g. int>0), or union of types
    """
    name: str | None
    guard: Guard
    def __init__(self, name: str = None, guard: Guard = None): ...
    def match_score(self, arg: Function) -> int | float: ...
    def zip(self, args: list[Function]) -> dict[Parameter, Function]: ...
    def min_len(self) -> int: ...
    def max_len(self) -> int | float: ...
    def __len__(self): ...
    def __getitem__(self, item): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class AnyPattern(Pattern):
    pass
Any: AnyPattern

class ValuePattern(Pattern):
    value: Function
    def __init__(self, value: Function, name: str = None): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

# class Type(Pattern):
#     type: Function
#     def __init__(self, type: Function, name: str = None, guard: Guard = None): ...
#     def __eq__(self, other): ...
#     def __hash__(self): ...

class Prototype(Pattern):
    prototype: Function
    exprs: tuple[Expression, ...]
    def __init__(self, prototype: Function, name: str = None, guard: Guard = None, *exprs: Expression): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class Union(Pattern):
    patterns: frozenset[Pattern]
    def __init__(self, *patterns: Pattern, name: str = None, guard: Guard = None): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class ListPatt(Pattern):
    parameters: tuple[Parameter, ...]
    def __init__(self, *parameters: Parameter): ...
    def zip(self, args: list[Function] = None) -> dict[Pattern, Function]: ...
    def min_len(self) -> int | float:
        count = 0
        for param in self.parameters:
            count += int(param.quantifier in ("", "+"))
        return count
    def max_len(self) -> int | float:
        count = 0
        for param in self.parameters:
            if param.quantifier in ("+", "*"):
                return math.inf
            count += int(param.quantifier != "?")
        return count
    def __len__(self):
        return len(self.parameters)
    def __getitem__(self, item):
        return self.parameters[item]
    def match_score(self, arg: Function) -> int | float: ...
    def __eq__(self, other): ...
    def __hash__(self): ...

def patternize(val: Function) -> Pattern: ...

def named_patt(name: str) -> ListPatt: ...
def numbered_patt(index: int | Fraction) -> ListPatt: ...

class FuncBlock:
    exprs: list[Expression]
    env: Function
    native: callable
    def __init__(self, block: Block | callable, env: Function = None): ...
    def make_function(self, options: dict[Pattern, opt_type], env: Function = None) -> Function: ...
    def execute(self, args: list[Function] = None, scope: Function = None) -> Function: ...

opt_type = Function | FuncBlock | callable | None

class Option:
    pattern: ListPatt
    resolution: opt_type
    value: Function
    block: FuncBlock
    fn: callable
    dot_option: bool
    def __init__(self, pattern: ListPatt | Pattern | Parameter | str, resolution: opt_type = None): ...
    def is_null(self) -> bool: ...
    def not_null(self) -> bool: ...
    def nullify(self): ...
    def assign(self, val_or_block: opt_type): ...
    def resolve(self, args: list[Function] = None, proto: Function = None) -> Function: ...

class Function:
    name: str
    type: Function # class-like inheritance
    options: list[Option]
    named_options: dict[str, Option]
    # array: list[Option]
    block: Block
    env: Function
    exec: any
    return_value: Function
    is_null: bool
    init: any
    def __init__(self, opt_pattern: Pattern | Parameter | str = None,
                     opt_value: opt_type = None,
                     options: dict[Pattern | Parameter | str, opt_type] = None,
                     # block: Block = None,
                     type: Function = None,
                     env: Function = None,
                     # value: Value = None,
                     # is_null=False,
                    name: str = None
                 ): ...

    def add_option(self, pattern: Pattern | Parameter | str, resolution: opt_type = None) -> Option: ...
    def remove_option(self, pattern: Pattern | Parameter | str): ...
    def assign_option(self, pattern: Pattern, resolution: opt_type = None) -> Function: ...
    def index_of(self, key: list[Function]) -> int | None: ...
    def select(self, key: Pattern | list[Function] | str | int, walk_prototype_chain=True, ascend_env=False) -> Option: ...
    def call(self, key: list[Function], copy_option=True, ascend=False) -> Function: ...
    def deref(self, name: str, ascend_env=True) -> Function: ...
    def init(self, pattern: Pattern, key: list[Function], parent: Function = None, copy=True) -> Function: ...
    def instanceof(self, prototype: Function) -> bool: ...
    def clone(self) -> Function: ...

class Operator:
    text: str
    prefix: int | None
    postfix: int | None
    binop: int | None
    ternary: str | None
    associativity: str
    fn: Function
    chainable: bool
    static: bool | callable
    def __init__(self, text, fn:Function=None, prefix:int=None, postfix:int=None, binop:int=None,
                 ternary:str=None, associativity='left', chainable:bool=False, static=False): ...
    def eval_args(self, lhs, rhs) -> list[Function]: ...
