import math
import types
from fractions import Fraction
from Syntax import Block
from Env import *

"""
A Pattern is like a regex for types; it can match one very specific type, or even one specific value
or it can match a type on certain conditions (eg int>0), or union of types
"""
class Pattern:
    def __init__(self, name: str = None, guard=None):
        self.name = name
        if guard and not isinstance(guard, Function):
            guard = Function(ListPatt(Parameter(Any)), guard)
        self.guard = guard
    def match_score(self, arg):
        if isinstance(arg, Value) and arg.value == self.name:
            return 1
        match self:
            case ValuePattern(value=value):
                return int(arg == value)
            case Prototype(prototype=prototype):
                score = (arg.instanceof(prototype)) * 2/3
            case Union(patterns=patterns):
                count = len(self)
                # if BasicType.Any in (getattr(p, "basic_type", None) for p in patterns):
                #     count += len(BasicType) - 1
                for patt in patterns:
                    m_score = patt.match_score(arg)
                    if m_score:
                        score = m_score / count
                        break
                else:
                    score = 0
            case Pattern(name=name):
                return int(isinstance(arg, Value) and name == arg.value)
            # NOTE: ListPatt overrides match_score method, since more complex logic is required
            case _:
                raise Exception(f"Line: {Context.line}: Unknown pattern type: ", self)
        if not score:
            return 0
        if self.guard:
            result = self.guard.call([arg])
            score *= BuiltIns['bool'].call([result]).value  # noqa
        return score
    def __repr__(self):
        return f"Pattern({self.name})"
    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name and self.guard == other.guard
    def __hash__(self):
        return hash((self.name, self.guard))

class AnyPattern(Pattern):
    def __init__(self, name=None):
        super().__init__(name, None)
    def match_score(self, arg):
        return 1
    def __repr__(self):
        return "Any"


Any = AnyPattern()

class Parameter:
    def __init__(self, pattern, name=None, quantifier="", inverse=False):
        if isinstance(pattern, str):
            self.name = pattern
            self.pattern = ValuePattern(Value(pattern), pattern)
        elif isinstance(pattern, Function):
            self.pattern = Prototype(pattern)
            self.name = name
        else:
            self.name = name or pattern.name
            self.pattern = pattern
        self.quantifier = quantifier
        match quantifier:
            case "":  self.count = (1, 1)
            case "?": self.count = (0, 1)
            case "+": self.count = (1, math.inf)
            case "*": self.count = (0, math.inf)
        self.optional = quantifier in ("?", "*")
        self.multi = quantifier in ("+", "*")
        self.inverse = inverse
    def specificity(self) -> int: ...
    def match_score(self, arg):
        if arg.type == BuiltIns['str'] and arg.value == self.name:
            return 1
        return self.pattern.match_score(arg)
    def __eq__(self, other):
        return self.pattern == other.pattern and self.name == other.name and \
            self.quantifier == other.quantifier and self.inverse == other.inverse
    def __hash__(self):
        return hash((self.pattern, self.name, self.quantifier, self.inverse))
    def __repr__(self):
        return f"{'!' * self.inverse}{self.pattern}" + (' ' + self.name if self.name else '')

class ValuePattern(Pattern):
    def __init__(self, value, name: str = None):
        super().__init__(name)
        self.value = value
    def __eq__(self, other):
        return isinstance(other, ValuePattern) and self.value == other.value and self.name == other.name
    def __hash__(self):
        return hash((self.value, self.name))
    def __repr__(self):
        return "ValuePattern("+repr(self.value)+")"

# class Type(Pattern):
#     basic_type: BasicType
#     def __init__(self, basic_type, name=None, guard=None):
#         super().__init__(name, guard)
#         self.basic_type = basic_type
#     def __eq__(self, other):
#         return isinstance(other, Type) and self.basic_type == other.basic_type and super().__eq__(other)
#     def __hash__(self):
#         return hash((self.basic_type, super()))
#     def __repr__(self):
#         return self.basic_type.value + ( '[]' if self.guard else '' )

class Prototype(Pattern):
    def __init__(self, prototype, name=None, guard=None, *exprs):
        super().__init__(name, guard)
        self.prototype = prototype
        self.exprs = exprs
    def __eq__(self, other):
        return isinstance(other, Prototype) and \
            id(self.prototype) == id(other.prototype) and super().__eq__(other)
    def __hash__(self):
        return hash((id(self.prototype), super()))
    def __repr__(self):
        return f"Prototype({self.prototype}{'[expr]' if self.exprs else ''}{'[guard]' if self.guard else ''})"

class Union(Pattern):
    patterns: frozenset[Pattern]
    def __init__(self, *patterns: Pattern, name=None, guard=None):
        super().__init__(name, guard)
        patts: list[Pattern] = []
        for i, patt in enumerate(patterns):
            if isinstance(patt, Union) and not (patt.name or patt.guard):
                list(patts.append(p) for p in patt.patterns)
            else:
                patts.append(patt)
        self.patterns = frozenset(patts)
    def __len__(self):
        return len(self.patterns)
    def __eq__(self, other):
        return isinstance(other, Union) and self.patterns == other.patterns and super().__eq__(other)
    def __hash__(self):
        return hash((self.patterns, self.name, self.guard))
    def __repr__(self):
        return '|'.join(map(repr, self.patterns))

class ListPatt(Pattern):
    def __init__(self, *parameters: Parameter, name=None, guard=None):
        super().__init__(name, guard)
        self.parameters = tuple(parameters)
    def zip(self, args: list):
        d = {}
        if args is None:
            return d
        assert self.min_len() <= len(args) <= self.max_len()
        if len(args) == 0:
            return d
        params = (p for p in self.parameters)
        for arg in args:
            param = next(params)
            if param.name:
                patt = ListPatt(Parameter(param.name))
                d[patt] = arg
        return d
    def min_len(self):
        count = 0
        for param in self.parameters:
            # count += int(param.quantifier in ("", "+"))
            count += not param.optional
        return count
    def max_len(self):
        for param in self.parameters:
            if param.quantifier in ("+", "*"):
                return math.inf
        return len(self.parameters)
    def __len__(self):
        return len(self.parameters)
    def __getitem__(self, item):
        return self.parameters[item]
    def match_score(self, list_value):
        if list_value.type != BuiltIns['list']:
            return 0
        args = list_value.value
        if not self.min_len() <= len(args) <= self.max_len():
            return 0
        if len(args) == 0:
            if len(self) == 0:
                return 1
            else:
                return int(self.min_len() == 0) / len(self)
        params = (param for param in self.parameters)
        score = 0
        for arg in args:
            param = next(params, None)
            if not param:
                pass
            p_score = param.match_score(arg)
            if not p_score:
                return 0
            score += p_score
            # not yet implemented param.multi and param.optional
        return score / len(args)
    def __eq__(self, other):
        return isinstance(other, ListPatt) and self.parameters == other.parameters and super().__eq__(other)
    def __hash__(self):
        return hash((*self.parameters, self.name, self.guard))
    def __repr__(self):
        return f"[{', '.join(map(repr, self.parameters))}]"


def patternize(val):
    if isinstance(val, Value):
        if isinstance(val.value, Pattern):
            return val.value
        return ValuePattern(val)
    return Prototype(val)

def named_patt(name: str) -> ListPatt:
    return ListPatt(Parameter(Pattern(name)))
def numbered_patt(index: int | Fraction) -> ListPatt:
    return ListPatt(Parameter(ValuePattern(Value(index))))

class ReusableMap:
    fn: types.FunctionType
    iterable: any
    indices: set[int]
    def __init__(self, fn, iterable):
        self.fn = fn
        self.iterable = iterable.copy()
        self.indices = set([])
    def __getitem__(self, item):
        if item not in self.indices:
            self.iterable[item] = self.fn(self.iterable[item])
        return self.iterable[item]


class FuncBlock:
    native = None
    def __init__(self, block, env=None):
        if isinstance(block, Block):
            self.exprs = list(map(Context.make_expr, block.statements))
        else:
            self.native = block
        self.env = env or Context.env
    def make_function(self, options, prototype):
        return Function(options=options, type=prototype, env=self.env)
    def execute(self, args=None, scope=None):
        if scope:
            def break_():
                Context.pop()
                return scope.return_value or scope
        else:
            scope = Context.env

            def break_():
                return Value(None)

        if self.native:
            result = self.native(scope, *args)
            return break_() and result
        for expr in self.exprs:
            Context.line = expr.line
            expr.evaluate()
            if scope.return_value:
                break
            if Context.break_:
                Context.break_ -= 1
                break
        return break_()


class Option:
    resolution = None
    value = None
    block = None
    fn = None
    dot_option = False
    def __init__(self, pattern, resolution=None):
        match pattern:
            case ListPatt():
                self.pattern = pattern
            case str():
                self.pattern = ListPatt(Parameter(pattern))
            case Parameter():
                self.pattern = ListPatt(pattern)
            case Pattern():
                self.pattern = ListPatt(Parameter(pattern))
        if resolution is not None:
            self.assign(resolution)
    def is_null(self):
        return (self.value and self.block and self.fn) is None
    def not_null(self):
        return (self.value or self.block or self.fn) is not None
    def nullify(self):
        self.resolution = None
        if self.value:
            del self.value
        if self.block:
            del self.block
        if self.fn:
            del self.fn
    def assign(self, resolution):
        self.resolution = resolution
        match resolution:
            case Function(): self.value = resolution
            case FuncBlock(): self.block = resolution
            case types.FunctionType(): self.fn = resolution
            case _:
                raise ValueError("Could not assign resolution: ", resolution)
    def resolve(self, args=None, proto=None):
        if self.value:
            return self.value
        if self.fn:
            return self.fn(*args)
        if self.block is None:
            raise NoMatchingOptionError("Could not resolve null option")
        if self.dot_option:
            proto = args[0]
        fn = self.block.make_function(self.pattern.zip(args), proto)
        Context.push(Context.line, fn, self)
        return self.block.execute(args, fn)

    def __repr__(self):
        if self.value:
            return f"{self.pattern}={self.value}"
        if self.block or self.fn:
            return f"{self.pattern}: {self.block or self.fn}"
        return f"{self.pattern} -> null"


class Function:
    return_value = None
    def __init__(self, opt_pattern=None, resolution=None, options=None, type=None, env=None, name=None):
        self.name = name
        self.type = type or BuiltIns['fn']
        self.env = env or Context.env
        self.options = []  # [Option(Pattern(), self)]
        self.named_options = {}
#        self.array = [Option(ListPatt(Parameter(ValuePattern(Value(0)))))]
        if options:
            for patt, val in options.items():
                self.add_option(patt, val)
        if opt_pattern is not None:
            self.assign_option(opt_pattern, resolution)

    def add_option(self, pattern, resolution=None):
        option = Option(pattern, resolution)
        name = num = None
        match pattern:
            case int() as i:  num = i
            case str():       name = pattern
            case Parameter(): name = pattern.name
            case ListPatt():  name = pattern[0].name if len(pattern) == 1 else None
            case Pattern():   name = pattern.name
        if name is not None:
            self.named_options[name] = option
        if hasattr(self, "value") and isinstance(self.value, list):
            if not num and len(option.pattern) == 1:
                patt = option.pattern.parameters[0].pattern
                if isinstance(patt, ValuePattern) and isinstance(patt.value, Value):
                    num = patt.value.value
            if num == len(self.value):
                self.value.append(option)
        self.options.insert(0, option)
        # self.options.sort(key=lambda opt: opt.pattern.specificity, reverse=True)
        return option

    def remove_option(self, pattern):
        opt = self.select(pattern)
        opt.nullify()

    def assign_option(self, pattern, resolution):
        try:
            opt = self.select(pattern)
            opt.nullify()
            opt.assign(resolution)
        except NoMatchingOptionError:
            self.add_option(pattern, resolution)
        if isinstance(resolution, Value):
            return resolution
        else:
            return Value(None)

    def index_of(self, key) -> int | None:
        idx = None
        high_score = 0
        arg_list = Value(key)
        for i, opt in enumerate(self.options):
            score = opt.pattern.match_score(arg_list)
            if score == 1:
                return i
            if score > high_score:
                high_score = score
                idx = i
        return idx

    def select(self, key, walk_prototype_chain=True, ascend_env=False):
        try:
            match key:
                case Pattern():
                    opt = [opt for opt in self.options if opt.pattern == key]
                    if opt:
                        return opt[0]
                case str() as key:
                    return self.named_options[key]
                # case int() as key if key > 0:
                #     return self.array[key]
                case [arg] if arg.instanceof(BuiltIns['str']):
                    return self.named_options[arg.value]
                # case [arg] if arg.instanceof(BuiltIns['int']) and arg.value > 0:
                #     return self.array[arg.value]
        except (IndexError, KeyError):
            pass
        i = self.index_of(key)
        if i is not None:
            return self.options[i]
        if walk_prototype_chain and self.type:
            try:
                return self.type.select(key, ascend_env=ascend_env)
            except NoMatchingOptionError:
                pass
        if ascend_env and self.env:
            try:
                return self.env.select(key, walk_prototype_chain)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in function {self}")

    def call(self, key, ascend=False):
        try:
            option = self.select(key)
        except NoMatchingOptionError as e:
            if ascend and self.env:
                try:
                    return self.env.call(key, ascend=True)
                except NoMatchingOptionError:
                    pass
            raise e
        return option.resolve(key, self)

    def deref(self, name: str, ascend_env=True):
        option = self.select(name, ascend_env=ascend_env)
        return option.resolve(None, self)

    def instanceof(self, prototype):
        return bool(self.type == prototype or self.type and self.type.instanceof(prototype))

    def clone(self):
        if hasattr(self, "value"):
            fn = Value(self.value)
            fn.type = self.type
            fn.env = self.env
        else:
            fn = Function(type=self.type, env=self.env)
        for opt in self.options:
            fn.assign_option(opt.pattern,
                             opt.value.clone() if opt.value else opt.block or opt.fn)
        return fn

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        if not isinstance(other, Function):
            return False
        if self.type != other.type:
            return False
        if getattr(self, "value", object()) == getattr(other, "value", object()):
            return True
        if self.env != other.env:
            return False
        try:
            for opt in self.options:
                assert opt.resolution == other.select(opt.pattern).resolution
        except NoMatchingOptionError:
            return False
        return True

    def __repr__(self):
        if self == Context.root:
            return 'root'
        if self.type == Context.root:
            return 'root.main'
        try:
            return repr(self.value)  # noqa
        except AttributeError:
            pass
        prefix = self.name or ""
        if len(self.options) == 1:
            return f"{prefix}{{{self.options[0]}}}"
        else:
            return f"{prefix}{self.named_options}"

class Value(Function):
    def __init__(self, value, type_=None):
        super().__init__(type=type_ or TypeMap[type(value)])
        if isinstance(value, Fraction) and value.numerator % value.denominator == 0:
            self.value = int(value)
        else:
            self.value = value

    def set_value(self, new_value):
        if isinstance(new_value, Value):
            self.value = new_value.value
            self.type = new_value.type
        else:
            self.value = new_value
            self.type = TypeMap[type(new_value.value)]
        return self

    def is_null(self):
        return self.value is None
    def not_null(self):
        return self.value is not None
    def clone(self):
        c = super().clone()
        c.value = self.value
        return c
    # def clone(self):
    #     val = self.value
    #     if val == BasicType.Function:
    #         val = val.clone()
    #     if val == BasicType.List:
    #         val = val.copy()
    #     return Value(val, self.type)

    def __eq__(self, other):
        try:
            # assume List
            return tuple(self.value) == tuple(other.value)
        except TypeError:
            return isinstance(other, Value) and self.value == other.value
    def __hash__(self):
        if self.type == BuiltIns['list']:
            return hash(tuple(self.value))
        return hash(self.value)
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return repr(self.value)

# class ListFunc(Function):
#     def __init__(self, *values: Value):
#         super().__init__(type=BuiltIns['List'])
#         for i, val in enumerate(values):
#             self.add_option(ListPatt(Parameter(ValuePattern(Value(i)))), val)
#     def push(self, val):
#         self.add_option(numbered_patt(self.len()+1), val)
#         return self
#     def pop(self, index: int = -1) -> Value:
#         return self.remove_option(Value(index))
#     def len(self) -> int:
#         return len(self.options)


class Operator:
    def __init__(self, text, fn=None,
                 prefix=None, postfix=None, binop=None, ternary=None,
                 associativity='left',
                 chainable=False,
                 static=False):
        Op[text] = self
        self.text = text
        # self.precedence = precedence
        self.fn = fn
        self.associativity = associativity  # 'right' if 'right' in flags else 'left'
        self.prefix = prefix  # 'prefix' in flags
        self.postfix = postfix  # 'postfix' in flags
        self.binop = binop  # 'binop' in flags
        self.ternary = ternary
        self.static = static  # 'static' in flags
        self.chainable = chainable

        assert self.binop or self.prefix or self.postfix or self.ternary

    def eval_args(self, lhs, rhs) -> list[Value]:
        raise NotImplemented('Operator.prepare_args not implemented')

    def __repr__(self):
        return self.text


if __name__ == "__main__":
    pass
