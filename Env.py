from fractions import Fraction


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


def read_number(text: str, base=6) -> int | float | Fraction:
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
        val = int(whole, base) if whole else 0
        pow = base
        for c in frac:
            if int(c) >= base:
                raise ValueError("invalid digit for base "+str(base))
            val += int(c) / pow
            pow *= base
        return val

def write_number(num: int|float|Fraction, base=6, precision=12) -> str:
    """ take a number and convert to a string of digits, possibly separated by / or . """
    if isinstance(num, Fraction):
        return write_number(num.numerator) + "/" + write_number(num.denominator)
    sign = "-" * (num < 0)
    num = abs(num)
    whole = int(num)
    frac = num - whole

    ls = sign + nat2str(whole, base)
    if frac == 0:
        return ls
    rs = frac_from_base(frac, base, precision)
    return f"{ls}.{''.join(str(d) for d in rs)}"

def nat2str(num: int, base=6):
    if num < base:
        return str(num)
    else:
        return nat2str(num // base, base) + str(num % base)

def frac_from_base(num: float, base=6, precision=12):
    digits = []
    remainders = []
    for i in range(precision):
        tmp = num * base
        itmp = int(tmp)
        num = tmp - itmp
        if 1-num < base ** (i-precision):
            digits += [itmp+1]
            break
        digits += [itmp]
    return digits