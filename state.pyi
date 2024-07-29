from runtime import Frame, Record, Function, Option
from syntax import Operator

print(f'loading {__name__}.py')

source_path: str | None
source_code: str | None
stack: list[Frame]
env: Frame
line: int
root: Frame
debug: bool | 0
trace: list[Call]
break_loop: int
continue_: int
settings: dict[str]

Op: dict[str, Operator] = {}
BuiltIns: dict[str, Record] = {}
BASES: dict[str, int] = {'b': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6, 's': 7, 'o': 8, 'n': 9, 'd': 10}


class Call:
    file: str
    line: int
    fn: Function
    option: Option
    def __init__(self, line: int, fn: Function, option: Option | None = None): ...

def push(line: int, env: Frame, option: Option | None = None): ...

def pop(): ...

def get_trace() -> str: ...

def deref(name: str, *default) -> Record: ...