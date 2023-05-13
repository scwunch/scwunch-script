from DataStructures import *
from fractions import Fraction

class Context:
    line: int
    _env: list[Function]
    env: Function
    root: Function
    debug: bool
    trace: list[Call]
    break_loop: int
    continue_: int
    settings: dict
    @staticmethod
    def push(line: int, env: Function, option: Option): ...
    @staticmethod
    def _push(line: int, env: Function, option: Option): ...
    @staticmethod
    def pop(): ...
    @staticmethod
    def get_trace() -> str: ...
    @classmethod
    def make_expr(cls, nodes):
        pass

class Call:
    line: int
    fn: Function
    option: Option
    def __init__(self, line: int, fn: Function, option: Option): ...
    def __str__(self): ...

class RuntimeErr(Exception): ...
class SyntaxErr(Exception): ...
class NoMatchingOptionError(RuntimeErr): ...
class OperatorError(SyntaxErr): ...
class TypeErr(RuntimeErr): ...

Op: dict[str, Operator]
BuiltIns: dict[str, Function | Pattern]
TypeMap: dict[type, Function]

def read_number(text: str, base=6) -> int | float | Fraction: ...
def write_number(num: int|float|Fraction, base=6, precision=12, sep="_") -> str: ...