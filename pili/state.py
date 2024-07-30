print(f'loading {__name__}.py')

source_path: str = None
source_code: str = None
stack = []
env = None
line = 0
root = None
debug = False
trace = []
break_loop = 0
continue_ = 0
settings = {'base': 10, 'sort_options': True}

Op = {}
BuiltIns = {}
BASES = {'b': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6, 's': 7, 'o': 8, 'n': 9, 'd': 10}


class Call:
    def __init__(self, file, line, frame, fn=None, option=None):
        self.file = file
        self.line = line
        self.frame = frame
        self.fn = fn
        self.option = option

    def __str__(self):
        if self.option:
            return f"> Line {self.line}:  {self.fn} -> {self.option.pattern}"
        if self.fn:
            return f"> Line {self.line}:  {self.fn}"
        return f"> Line {self.line}: {self.frame}"


def push(frame, fn=None, option=None):
    global env
    stack.append(frame)
    env = frame
    trace.append(Call(source_path, line, frame, fn, option))

def pop():
    global env, line
    stack.pop()
    env = stack[-1]
    # if Context._env:
    #     Context.env = Context._env[-1]
    # else:
    #     Context.env = None  # BuiltIns['pili']
    line = trace.pop().line

def get_trace():
    if not trace:
        print("<no trace available>")
    file = None
    trace_lines = []
    for call in trace:
        if call.file != file:
            file = call.file
            trace_lines.append(f'In file "{file}"')
        trace_lines.append(str(call))
    return "\n".join(trace_lines)

def deref(name: str, *default):
    raise NotImplementedError('Implemented in utils.py')