from DataStructures import *

class Context:
    line: int
    _env: list[Function]
    env: Function
    root: Function
    debug: bool
    trace: list[Call]
    @staticmethod
    def push(line: int, env: Function, option: Option): ...
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

Op: dict[str, Operator]
BuiltIns: dict[str, Function]