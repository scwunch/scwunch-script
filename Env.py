from fractions import Fraction


class Context:
    _env = []
    env = None
    line = 0
    root = None
    debug = False
    trace = []
    break_loop = 0
    continue_ = 0
    settings = {'base': 10, 'sort_options': True}

    @staticmethod
    def push(line, env, option=None):
        BuiltIns['root'] = env
        Context.push = Context._push
        Context.push(line, env, option)

    @staticmethod
    def _push(line, env, option=None):
        Context._env.append(env)
        Context.env = env
        Context.trace.append(Call(line, env, option))

    @staticmethod
    def pop():
        Context._env.pop()
        Context.env = Context._env[-1]
        # if Context._env:
        #     Context.env = Context._env[-1]
        # else:
        #     Context.env = None  # BuiltIns['pili']
        Context.line = Context.trace.pop().line

    @staticmethod
    def get_trace():
        return "\n".join(str(ct) for ct in Context.trace)

    @staticmethod
    def deref(name: str, *default):
        scope = Context.env
        while scope:
            try:
                return scope.names[name]
            except KeyError:
                scope = scope.scope
        if default:
            return default[0]
        raise MissingNameErr(f"Line {Context.line}: Cannot find name '{name}' in current scope.")


class Call:
    def __init__(self, line, fn, option=None):
        self.line = line
        self.fn = fn
        self.option = option

    def __str__(self):
        if self.option:
            return f"> Line {self.line}:  {self.fn} -> {self.option.pattern}"
        return f"> Line {self.line}:  {self.fn}"

class RuntimeErr(Exception):
    def __str__(self):
        return f"\n\nContext.Trace:\n{Context.get_trace()}\n> {super().__str__()}"
class SyntaxErr(Exception):
    pass
class KeyErr(RuntimeErr):
    pass
class NoMatchingOptionError(KeyErr):
    pass
class MissingNameErr(KeyErr):
    pass
class OperatorErr(SyntaxErr):
    pass
class TypeErr(RuntimeErr):
    pass
class SlotErr(TypeErr):
    pass


Op = {}
BuiltIns = {}
TypeMap = {}


def read_number(text: str, base) -> int | float | Fraction:
    """ take a number in the form of a string of digits, or digits separated by a / or .
        if the number ends with a d, it will be read as a decimal"""
    if isinstance(text, int) or isinstance(text, float) or isinstance(text, Fraction):
        return text
    if text.endswith('d'):
        text = text[:-1]
        try:
            return int(text)
        except ValueError:
            if '/' in text:
                return Fraction(text)
            return float(text)
    try:
        return int(text, base)
    except ValueError:
        if '/' in text:
            numerator, _, denominator = text.partition('/')
            return Fraction(int(numerator, base), int(denominator, base))
        whole, _, frac = text.partition('.')
        try:
            val = int(whole, base) if whole else 0
            pow = base
            for c in frac:
                val += int(c) / pow
                pow *= base
            return val
        except ValueError as e:
            raise TypeErr(f"Line {Context.line}: {e}")


def write_number(num: int|float|Fraction, base, precision=12, sep="_", grouping=4) -> str:
    """ take a number and convert to a string of digits, possibly separated by / or . """
    if base == 10:
        grouping = 3
        # return str(num)
    if isinstance(num, Fraction):
        return write_number(num.numerator, base, sep=sep) + "/" + write_number(num.denominator, base, sep=sep)
    sign = "-" * (num < 0)
    num = abs(num)
    whole = int(num)
    frac = num - whole

    digits = get_digits(whole, base)
    if sep:
        for i in range(len(digits)-grouping, 0, -grouping):
            digits.insert(i, sep)
    ls = sign + ''.join(digits)
    if isinstance(num, int):
        return ls
    rs = frac_from_base(frac, base, precision)
    return f"{ls}.{''.join(str(d) for d in rs)}"

def get_digits(num: int, base) -> list[str]:
    if num < base:
        return [str(num)]
    else:
        result = get_digits(num // base, base)
        result.append(str(num % base))
        return result

def frac_from_base(num: float, base, precision=12):
    digits = []
    # remainders = []
    for i in range(precision):
        tmp = num * base
        itmp = int(tmp)
        num = tmp - itmp
        if itmp == 0 and num < base ** (i-precision):
            break
        elif 1-num < base ** (i-precision):
            digits += [itmp+1]
            break
        digits += [itmp]
    # optionally fix an issue where numbers less than 1 but very close to one print as '0.6'
    # if digits == [base]:
    #     return [base-1] * precision
    return digits


def call(fn, args):
    kwargs = {**args.named_arguments, **dict(zip(args.flags, [BuiltIns['true']] * len(args.flags)))}
    return fn(*args.positional_arguments, **kwargs)