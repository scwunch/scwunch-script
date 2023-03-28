class Context:
    _env = []
    env = None
    line = 0
    root = None
    debug = False
    trace = []
    break_ = 0
    continue_ = 0

    @staticmethod
    def push(line, env, option):
        Context._env.append(env)
        Context.env = env
        Context.trace.append(Call(line, env, option))

    @staticmethod
    def pop():
        Context._env.pop()
        Context.env = Context._env[-1]
        Context.trace.pop()

    @staticmethod
    def get_trace():
        return "\n".join(str(ct) for ct in Context.trace)

class Call:
    def __init__(self, line, fn, option):
        self.line = line
        self.fn = fn
        self.option = option

    def __str__(self):
        return f"> Line {self.line}:  {self.fn} -> {self.option.pattern}"

class RuntimeErr(Exception):
    def __str__(self):
        return f"\n\nContext.Trace:\n{Context.get_trace()}\n> {super().__str__()}"
class SyntaxErr(Exception):
    pass
class NoMatchingOptionError(RuntimeErr):
    pass
class OperatorError(SyntaxErr):
    pass


Op = {}
BuiltIns = {}
TypeMap = {}