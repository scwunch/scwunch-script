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
    settings = {'base': 6}

    @staticmethod
    def push(line, env, option):
        BuiltIns['root'] = env
        # env.name = 'root'
        from DataStructures import ListPatt, Parameter
        # BuiltIns['pili'].add_option(ListPatt(Parameter('pili')), env)
        Context.push = Context._push
        Context.push(line, env, option)

    @staticmethod
    def _push(line, env, option):
        Context._env.append(env)
        Context.env = env
        Context.trace.append(Call(line, env, option))

    @staticmethod
    def pop():
        Context._env.pop()
        if Context._env:
            Context.env = Context._env[-1]
        else:
            Context.env = BuiltIns['pili']
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
class TypeErr(RuntimeErr):
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
        val = int(whole, base) if whole else 0
        pow = base
        for c in frac:
            if int(c) >= base:
                raise ValueError("invalid digit for base "+str(base))
            val += int(c) / pow
            pow *= base
        return val

def write_number(num: int|float|Fraction, base, precision=12, sep="_") -> str:
    """ take a number and convert to a string of digits, possibly separated by / or . """
    if isinstance(num, Fraction):
        return write_number(num.numerator, base, sep=sep) + "/" + write_number(num.denominator, base, sep=sep)
    sign = "-" * (num < 0)
    num = abs(num)
    whole = int(num)
    frac = num - whole

    digits = get_digits(whole, base)
    if sep:
        for i in range(len(digits)-4, 0, -4):
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
