import math
from fractions import Fraction
from . import state
from .state import BASES, BuiltIns

print(f'loading {__name__}.py')

class PiliException(Exception):
    pass
class RuntimeErr(PiliException):
    def __str__(self):
        return f"\n\nTraceback:\n{state.get_trace()}\n> {super().__str__()}"
class SyntaxErr(PiliException):
    pass
class ContextErr(SyntaxErr):
    pass
class KeyErr(RuntimeErr):
    pass
class SlotErr(RuntimeErr):
    pass
class NoMatchingOptionError(KeyErr):
    pass
class MissingNameErr(KeyErr):
    pass
class InitializationErr(KeyErr):
    pass
class OperatorErr(SyntaxErr):
    pass
class PatternErr(RuntimeErr):
    pass
class TypeErr(PatternErr):
    pass
class MatchErr(PatternErr):
    pass

def state_deref(name: str, *default):
    scope = state.env
    while scope:
        val = scope.locals.get(name, scope.vars.get(name, False))
        # this should be scope[name] but for some reason it breaks if I don't search local first...
        if val:
            return val
        if val is None:
            raise MissingNameErr(f"Line {state.line}: '{name}' is not yet initialized.")
        scope = scope.scope
    if default:
        return default[0]
    raise MissingNameErr(f"Line {state.line}: Cannot find name '{name}' in current scope.")
state.deref = state_deref

class frozendict(dict):
    hash: int

    def __hash__(self):
        try:
            return self.hash
        except AttributeError:
            try:
                self.hash = hash(frozenset(self.items()))
            except TypeError:
                self.hash = hash(frozenset(self))
            return self.hash

    def __setitem__(self, *args, **kwargs):
        raise RuntimeError(f'Cannot change values of immutable {repr(self)}.')

    __delitem__ = pop = popitem = clear = update = setdefault = __setitem__

    def __add__(self, other: tuple | dict):
        new = frozendict(self)
        dict.update(new, other)
        return new

    def __sub__(self, other):
        new = frozendict(self)
        dict.__delitem__(new, other)
        return new

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return f"frozendict({super().__repr__()})"


def read_number(text: str, base, force_float=False) -> int | float | Fraction:
    """ take a number in the form of a string of digits, or digits separated by a / or .
        if the number ends with a d, it will be read as a decimal"""
    if isinstance(text, int | float | Fraction):
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
    if text[-1] in BASES:
        base = BASES[text[-1]]
        text = text[:-1]
    if not force_float:
        try:
            return int(text, base)
        except ValueError:
            pass
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
    digits: list[int | str] = get_digits(int_part, base)
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

def pili(*args):
    global pili
    from pili import pili as pili_function
    pili = pili_function
    return pili(*args)