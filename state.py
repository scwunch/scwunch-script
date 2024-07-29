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
    def __init__(self, line, fn, option=None):
        self.line = line
        self.fn = fn
        self.option = option

    def __str__(self):
        if self.option:
            return f"> Line {self.line}:  {self.fn} -> {self.option.pattern}"
        return f"> Line {self.line}:  {self.fn}"

def first_push(line, env, option=None):
    global push
    BuiltIns['root'] = env
    push = _push
    push(line, env, option)


push = first_push

def _push(line, env_, option=None):
    global env
    stack.append(env_)
    env = env_
    trace.append(Call(line, env, option))

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
    return "\n".join(str(ct) for ct in trace)

def deref(name: str, *default):
    raise NotImplementedError('Implemented in utils.py')