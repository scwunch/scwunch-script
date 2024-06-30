import math
from fractions import Fraction

print(f"loading module: {__name__} ...")

class Context:
    stack = []
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
        Context.stack.append(env)
        Context.env = env
        Context.trace.append(Call(line, env, option))

    @staticmethod
    def pop():
        Context.stack.pop()
        Context.env = Context.stack[-1]
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
            val = scope.locals.get(name, scope.vars.get(name, False))
            # I think this maybe should be val = scope[name]
            if val:
                return val
            if val is None:
                raise MissingNameErr(f"Line {Context.line}: '{name}' is not yet initialized.")
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

class PiliException(Exception):
    pass
class RuntimeErr(PiliException):
    def __str__(self):
        return f"\n\nContext.Trace:\n{Context.get_trace()}\n> {super().__str__()}"
class SyntaxErr(PiliException):
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
            # return float(text)
            base = 10
    if text.endswith('f'):
        force_float = True
        text = text[:-1]
    else:
        try:
            return int(text, base)
        except ValueError:
            force_float = False
    if '/' in text:
        numerator, _, denominator = text.partition('/')
        if force_float:
            return int(numerator, base) / int(denominator, base)
        return Fraction(int(numerator, base), int(denominator, base))
    sign = 1
    if text.startswith('-'):
        sign = -1
        text = text[1:]
    whole, _, frac = text.partition('.')
    if not whole:
        whole = '0'
    if not frac:
        frac = '0'
    denominator = base ** len(frac)
    numerator = int(whole, base) * denominator + int(frac, base)
    if force_float:
        return sign * numerator / denominator
    return Fraction(sign * numerator, denominator)

    # try:
    #     val = int(whole, base) if whole else 0
    #     pow = base
    #     for c in frac:
    #         val += int(c) / pow
    #         pow *= base
    #     return val
    # except ValueError as e:
    #     raise TypeErr(f"Line {Context.line}: {e}")


def prime_factors(n: int) -> set[int]:
    factors = set()
    d = 2
    while n > 1:
        if n % d == 0:
            factors.add(d)
            n //= d
            while n % d == 0:
                n //= d
        d += 1
        if d * d > n:
            if n > 1:
                factors.add(n)
            break
    return factors

def write_number_OLD(num: int|float|Fraction, base, precision=12, sep="_", grouping=None) -> str:
    """ take a number and convert to a string of digits, possibly separated by / or . """
    if grouping is None:
        if base < 7:
            grouping = 4
        elif base < 16:
            grouping = 3
        else:
            grouping = 2

    if base == 10:
        grouping = 3  # override the grouping parameter?  why?
        # return str(num)

    # if isinstance(num, Fraction):
    #     factors_of_base = prime_factors(base)
    #     n = num.denominator
    #     factors = set()
    #     p = 2
    #     flag = True
    #     while n > 1:
    #         while n % p == 0:
    #             if p not in factors_of_base:
    #                 flag = False
    #                 break
    #             n //= p
    #         p += 1
    #         if p * p > n:
    #             if n > 1 and n not in factors_of_base:
    #                 flag = False
    #             break
    #     if flag:
    #         # write number as finite decimal expansion
    #
    #     if not "simple implementation":
    #         # simpler implementation that only prints decimal style when the denominator _is_ a power of base
    #         radix = math.log(num.denominator, base)
    #         if radix.is_integer():  # ie, number is divisible by a power of base
    #             radix = int(radix)
    #             digits = str(num.numerator)
    #             return digits[:-radix] + "." + digits[-radix:]
    #         return write_number(num.numerator, base, sep=sep) + "/" + write_number(num.denominator, base, sep=sep)

    # num is int | float
    sign = "-" * (num < 0)
    n = abs(num)
    whole = int(n)
    frac = n - whole
    rs = fractional_digits(frac, base, precision)
    if isinstance(num, Fraction) and len(rs) >= precision:
        return write_number(num.numerator, base, sep=sep) + "/" + write_number(num.denominator, base, sep=sep)
    if isinstance(num, float):
        mantissa, exponent = math.frexp(n)
        power = exponent / math.log2(base)
        if abs(power) > precision:
            digits = get_mantissa(n, base, precision)
            # significand = str(rs[0]) + ''.join(map(str, rs[1:]))
            return str(digits[0]) + '.' + ''.join(map(str, digits[1:])) \
                + 'e' + '+'*(power>0) + str(math.floor(power))

    digits = get_digits(whole, base)
    if sep:
        for i in range(len(digits)-grouping, 0, -grouping):
            digits.insert(i, sep)
    ls = sign + ''.join(map(str, digits))
    if isinstance(num, int):
        return ls

    return f"{ls}.{''.join(map(str, rs))}"

def write_number(num: int|float|Fraction, base, precision=12, sep="_", grouping=None) -> str:
    """ take a number and convert to a string of digits, possibly separated by / or . """
    if grouping is None:
        if base < 7:
            grouping = 4
        elif base < 16:
            grouping = 3
        else:
            grouping = 2
    if isinstance(num, Fraction):
        return (f"{write_number(num.numerator, base, precision, sep, grouping)}"
                f"/{write_number(num.denominator, base, precision, sep, grouping)}")
    sign = "-" * (num < 0)
    num = abs(num)
    if isinstance(num, float):
        mantissa, exponent = math.frexp(num)
        power = exponent / math.log2(base)
        if abs(power) > precision:
            digits = get_mantissa(num, base, precision)
            # significand = str(rs[0]) + ''.join(map(str, rs[1:]))
            return str(digits[0]) + '.' + ''.join(map(str, digits[1:])) \
                + 'e' + '+' * (power > 0) + write_number(math.floor(power), base)
    int_part = int(num)
    digits = get_digits(int_part, base)
    if sep:
        for i in range(len(digits)-grouping, 0, -grouping):
            digits.insert(i, sep)
    ls = sign + ''.join(map(str, digits))
    if isinstance(num, int):
        return ls
    # else float
    rs = fractional_digits(num - int_part, base, precision)
    return f"{ls}.{''.join(map(str, rs))}"


def get_digits(num: int, base, num_digits=None) -> list[int]:
    if num < base:
        return [num]
    else:
        result = get_digits(num // base, base)
        result.append(num % base)
        return result

def fractional_digits(num: float | Fraction, base, precision=12) -> list[int]:
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

def get_mantissa(num: float, base, precision=12) -> list[int]:
    digits = []
    magnitude = math.floor(math.log(num, base))
    pow = base ** magnitude
    for i in range(precision + 1):
        d = num // pow
        digits.append(int(d))
        num -= d * pow
        pow /= base
    # round off and drop last digit
    if digits.pop() >= base/2:
        digits[-1] += 1
        for i in range(-1, -precision, -1):
            if digits[i] == base:
                digits[i] = 0
                digits[i-1] += 1
    return digits


def call(fn, args):
    """ call a python function on an Args object"""
    kwargs = {**args.named_arguments, **dict(zip(args.flags, [BuiltIns['true']] * len(args.flags)))}
    return fn(*args.positional_arguments, **kwargs)
