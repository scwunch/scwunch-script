# from DataStructures import *
from tables import *
from fractions import Fraction

class Context:
    line: int
    _env: list[Closure]
    env: Closure
    root: Closure
    debug: bool
    trace: list[Call]
    break_loop: int
    continue_: int
    settings: dict
    @staticmethod
    def push(line: int, env: Closure, option: Option = None): ...
    @staticmethod
    def _push(line: int, env: Closure, option: Option = None): ...
    @staticmethod
    def pop(): ...
    @staticmethod
    def get_trace() -> str: ...
    @classmethod
    def make_expr(cls, nodes): ...
    @staticmethod
    def deref(name: str, *default): ...

class Call:
    line: int
    fn: Function
    option: Option
    def __init__(self, line: int, fn: Function, option: Option = None): ...
    def __str__(self): ...

class RuntimeErr(Exception): ...
class SyntaxErr(Exception): ...
class KeyErr(RuntimeErr): ...
class NoMatchingOptionError(KeyErr): ...
class MissingNameErr(KeyErr): ...
class OperatorErr(SyntaxErr): ...
class TypeErr(RuntimeErr): ...
class SlotErr(TypeErr): ...

Op: dict[str, Operator]
BuiltIns: dict[str, Function | Pattern | PyFunction]
TypeMap: dict[type, Function]

def read_number(text: str, base=6) -> int | float | Fraction: ...
def write_number(num: int|float|Fraction, base=6, precision=12, sep="_") -> str: ...