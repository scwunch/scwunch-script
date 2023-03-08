import math
import types

from Syntax import BasicType, Block, Node

class Context:
    _env = []
    env = None
    line = 0
    root = None

    @staticmethod
    def push(env):
        Context._env.append(env)
        Context.env = env

    @staticmethod
    def pop():
        Context._env.pop()
        Context.env = Context._env[-1]

class Value:
    def __init__(self, value, basic_type: BasicType = None):
        self.value = value
        self.set_type(basic_type)

    def set_value(self, new_value):
        if isinstance(new_value, Value):
            self.value = new_value.value
            self.set_type(new_value.type)
        else:
            self.value = new_value
            self.set_type()
        return self

    def set_type(self, basic_type=None):
        if basic_type:
            self.type = basic_type
        else:
            match self.value:
                case BasicType(): self.type = BasicType.Type
                case Function(): self.type = BasicType.Function
                case Pattern(): self.type = BasicType.Pattern
                case None: self.type = BasicType.none
                case bool(): self.type = BasicType.Boolean
                case int(): self.type = BasicType.Integer
                case float(): self.type = BasicType.Float
                case str(): self.type = BasicType.String
                case list(): self.type = BasicType.List  # noqa
        return self.type

    def is_null(self):
        return self.value is None
    def not_null(self):
        return self.value is not None
    def clone(self):
        val = self.value
        if val == BasicType.Function:
            val = val.clone()
        if val == BasicType.List:
            val = val.copy()
        return Value(val, self.type)

    def __eq__(self, other):
        return isinstance(other, Value) and self.value == other.value
    def __hash__(self):
        return hash(self.value)
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return f"<{self.type.value}:{repr(self.value)}>"


class Parameter1:
    def __init__(self, name: str = None, value: Value = None, basic_type=None, fn=None):
        assert (0 if name else 1) <= bool(value) + bool(basic_type) < 2
        self.name = name
        self.value = value
        try:
            self.base_types = frozenset(basic_type or [])
        except TypeError:
            self.base_types = frozenset([basic_type])
        assert bool(fn) <= bool(basic_type)
        self.fn: Function = fn

    def specificity(self) -> int:
        if (self.name or self.value) and not self.base_types and not self.fn:
            return 7560
        spec = 0  # 2520 * bool(self.name or self.value)
        if self.base_types:
            spec += 252 if BasicType.Any in self.base_types else int(2520 / len(self.base_types))
        return spec + 2520 * bool(self.fn)

    def __eq__(self, other):
        return (self.name, self.value, self.base_types, self.fn) \
                == (other.name, other.value, other.base_types, other.fn)
    def __hash__(self):
        return hash((self.name, self.value, self.base_types, self.fn))
    def __repr__(self):
        return ('|'.join(t.value for t in self.base_types) + ' ' if self.base_types else '') \
            + f"{self.value or ''}{'[fn] ' if self.fn else ''} {self.name or ''}"

def is_match(val: Value | Parameter, param: Parameter):
    if isinstance(val, Parameter):
        return val == param
    if param.name and val.type == BasicType.String and param.name == val.value:
        return True
    if param.value:
        return param.value == val
    if BasicType.Any in param.base_types:
        return True
    if val.type not in param.base_types:
        return False
    if param.fn:
        return param.fn.call([val]).value
    # other subtype checks?
    return True

# score is out 7560, this number is 3*2520 (the smallest int divisible by all integers up to 10)
def match_score(val: Value, param: Parameter) -> int:
    if val == param.value or val.value == param.name:
        return 7560
    score = 0
    if BasicType.Any in param.base_types:
        score += 252
    elif val.type in param.base_types:
        score += int(2520 / len(param.base_types))
    if param.fn:
        score += 2520 * param.fn.call([val]).value
    # other subtype checks?
    return score


"""
A Pattern is like a regex for types; it can match one very specific type, or even one specific value
or it can match a type on certain conditions (eg int>0), or union of types
"""
class Pattern:
    def __init__(self, name: str = None, guard: Guard = None):
        self.name = name
        self.guard = guard
    def match_score(self, arg):
        if arg.type == BasicType.String and arg.value == self.name:
            return 7560
        match self:
            case ValuePattern(value=value):
                return 7560 * (arg == value)
            case Type(basic_type=basic_type):
                score = 2520 * (arg.type == basic_type)
            case Prototype(prototype=prototype):
                if arg.type != BasicType.Function:
                    return 0
                fn: Function = arg.value
                score = 5040 * (fn.instanceof(prototype))
            case Union(patterns=patterns):
                count = len(self)
                if BasicType.Any in (getattr(p, "basic_type", None) for p in patterns):
                    count += len(BasicType)
                for patt in patterns:
                    m_score = patt.match_score(arg)
                    if m_score:
                        score = m_score // count
                        break
                else:
                    score = 0
            # NOTE: ListPatt overrides match_score method, since more complex logic is required
            case _:
                raise NotImplemented(f"Line: {Context.line}: Unknown pattern type: ", self)
        if not score:
            return 0
        if self.guard:
            score += 2520 * evaluate(self.guard)
        return score
    def __repr__(self):
        return f"Pattern({self.name})"
    def __str__(self):
        return f"{self.name}"
    def __eq__(self, other):
        return hash(self) == hash(other)
    def __hash__(self):
        return hash((self.name, self.guard))

class Parameter:
    def __init__(self, pattern, name=None, quantifier="", inverse=False):
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
        if arg.type == BasicType.String and arg.value == self.name:
            return 7560
        score = 0
        score += self.pattern.match_score(arg)
        return score
    def __eq__(self, other):
        return self.pattern == other.pattern and self.name == other.name and \
            self.quantifier == other.quantifier and self.inverse == other.inverse
    def __hash__(self):
        return hash((self.pattern, self.name, self.quantifier, self.inverse))

class ValuePattern(Pattern):
    value: Value
    def __init__(self, value: Value, name: str = None):
        super().__init__(name)
        self.value = value
    def __eq__(self, other):
        return self.value == other.value and self.name == other.name
    def __hash__(self):
        return hash((self.value, self.name))

class Type(Pattern):
    basic_type: BasicType
    def __init__(self, basic_type, name=None, guard=None):
        super().__init__(name, guard)
        self.basic_type = basic_type
    def __eq__(self, other):
        return isinstance(other, Type) and self.basic_type == other.basic_type and super().__eq__(super(other))
    def __hash__(self):
        return has((self.basic_type, self.))

class Prototype(Pattern):
    def __init__(self, prototype, name=None, guard=None):
        super().__init__(name, guard)
        self.prototype = prototype

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


class ListPatt(Pattern):
    def __init__(self, *parameters: Parameter, name=None, guard=None):
        super().__init__(name, guard)
        self.parameters = tuple(parameters)
    def zip(self, args: list):
        assert self.min_len() <= len(args) <= self.max_len()
        d = {}
        params = (p for p in self.parameters)
        param = next(params)
        for arg in args:
            if param.name:
                patt = Pattern(param.name)
                d[patt] = arg
            param = next(params)
        return d
    def min_len(self):
        count = 0
        for param in self.parameters:
            # count += int(param.quantifier in ("", "+"))
            count += not param.optional
        return count
    def max_len(self):
        count = 0
        for param in self.parameters:
            if param.quantifier in ("+", "*"):
                return math.inf
            count += int(param.quantifier != "?")
        return count
    def __len__(self):
        return len(self.parameters)
    def __getitem__(self, item):
        return self.parameters[item]
    def match_score(self, list_value):
        if list_value.type != BasicType.List:
            return 0
        params = (param for param in self.parameters)
        param = next(params)
        score = 0
        for arg in list_value.value:
            p_score = param.match_score(arg)
            if not p_score:
                return 0
            score += p_score
            param = next(params)
            # not yet implemented param.multi and param.optional
        return score

def make_expr(nodes: list[Node]):
    raise NotImplemented


# opt_value_type = Value | Block | callable | None

class Option:
    value = None
    block = None
    fn = None
    def __init__(self, pattern, value=None):
        if isinstance(pattern, str):
            self.pattern = Pattern(pattern)
        elif isinstance(pattern, Parameter):
            self.pattern = ListPatt(pattern)
        else:
            self.pattern = pattern
        self.assign(value)

    def is_null(self):
        return (self.value and self.block and self.fn) is None
    def not_null(self):
        return (self.value or self.block or self.fn) is not None
    def assign(self, val_block_fn):
        match val_block_fn:
            case Value(): self.value = val_block_fn
            case Block(): self.block = val_block_fn
            case types.FunctionType: self.fn = val_block_fn
            case types.FunctionType(): self.fn = val_block_fn
            case fn if callable(fn): self.fn = fn

    def execute(self, args, proto=None, env=Context.env):
        if self.value:
            return self.value
        if self.fn:
            return self.fn(*args)
        assert self.block is not None
        fn = Function(options=self.pattern.zip(args), prototype=proto, env=env)
        Context.push(fn)
        for statement in self.block.statements:
            Context.line = statement.pos[0]
            expr = Context.make_expr(statement.nodes)
            result = expr.evaluate()
            if fn.return_value:
                Context.pop()
                return fn.return_value
        Context.pop()
        return Value(fn)

    def __repr__(self):
        if self.value:
            return f"{self.pattern}={self.value}"
        if self.block or self.fn:
            return f"{self.pattern}: {self.block or self.fn}"

class RuntimeErr(Exception):
    pass
class SyntaxErr(Exception):
    pass
class NoMatchingOptionError(RuntimeErr):
    pass
class OperatorError(SyntaxErr):
    pass

class Function:
    return_value = None
    # prototype = None
    def __init__(self, opt_pattern=None, opt_value=None, options=None, prototype=None, env=Context.env):
        # self.block = block or Block([])
        self.prototype = prototype
        self.env = env
        self.options = []  # [Option(Pattern(), self)]
        self.named_options = {}
        if options:
            for patt, val in options.items():
                self.add_option(patt, val)
        if opt_pattern is not None:
            self.assign_option(opt_pattern, opt_value)
        # self.return_value = value
        # self.is_null = is_null
        # if prototype:
        #     self.options += [opt.copy() for opt in prototype.options.copy()]

    def add_option(self, pattern, val_or_block=None):
        option = Option(pattern, val_or_block)
        name = None
        match pattern:
            case str(): name = pattern
            case Parameter(): name = pattern.name
            case Pattern(): name = pattern[0].name if len(pattern) == 1 else None
        if name is not None:
            self.named_options[name] = option
        self.options.insert(0, option)
        self.options.sort(key=lambda opt: opt.pattern.specificity, reverse=True)
        return option

    def assign_option(self, pattern, val_or_block):
        try:
            opt = self.select(pattern)
            opt.assign(val_or_block)
        except NoMatchingOptionError:
            self.add_option(pattern, val_or_block)
        if isinstance(val_or_block, Value):
            return val_or_block
        else:
            return Value(None)

    def index_of(self, key: list[Value]) -> int | None:
        idx = None
        high_score = 0
        for i, opt in enumerate(self.options):
            params = opt.pattern.all_parameters
            if not (len(opt.pattern.required_parameters) <= len(key) <= len(params)):
                continue
            score = 0
            for j in range(len(params)):
                if j == len(key):
                    return i
                param_score = match_score(key[j], params[j])
                if not param_score:
                    break  # no match; continue outer loop
                score += param_score
            else:
                if score == 7560 * len(key):  # perfect match
                    return i
                elif score > high_score:
                    high_score = score
                    idx = i
        return idx

    def select(self, key: Pattern | list[Value], walk_prototype_chain=True, ascend_env=False):
        """
        :param key: list of args, or pattern of params
        :param walk_prototype_chain: bool=True search options of prototype if not found
        :param ascend_env: bool=False search containing environment if not found
        :param create_if_not_exists: create a null function option if not found
                (I think these two options should probably be mutually exclusive)
        :returns the matching option, creating a null option if none exists
        """
        if isinstance(key, Pattern):
            opt = [opt for opt in self.options if opt.pattern == key]
            if opt:
                return opt[0]
            # if ascend_env:
            #     return self.add_option(key)
            # raise NoMatchingOptionError(f"No option found in function {self} matching pattern {key}")
        else:
            if len(key) == 1 and key[0].type == BasicType.String:
                option = self.named_options.get(key[0].value, None)
                if option:
                    return option
            i = self.index_of(key)
            if i is not None:
                return self.options[i]
        # if create_if_not_exists:
        #     return self.add_option(Pattern(*[Parameter(value=k) for k in key]))
        if walk_prototype_chain and self.prototype:
            try:
                return self.prototype.select(key)
            except NoMatchingOptionError:
                pass
        if ascend_env and self.env:
            try:
                return self.env.select(key)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in function {self}")

    def call(self, key: list[Value], ascend=False) -> Value | None:
        try:
            option = self.select(key)
        except NoMatchingOptionError as e:
            if ascend and self.env:
                try:
                    return self.env.call(key, ascend=True)
                except NoMatchingOptionError:
                    pass
            raise e
        return option.execute(key, self, self)

    def deref(self, name: str, ascend_env=True):
        return self.call([Value(name)], ascend=ascend_env)

    # def init(self, pattern: Pattern, key: list[Value], parent=None, copy=True):
    #     parent = parent or self
    #     fn = Function(block=self.block, prototype=parent, env=parent.env) if copy else self
    #     for arg, param in zip(key, pattern.required_parameters):
    #         fn.assign_option(Pattern(param), Function(value=arg))
    #     fn.args = key
    #     return fn

    def instanceof(self, prototype):
        return self == prototype or \
            bool(self.prototype) and (self.prototype == prototype or self.prototype.instanceof(prototype))

    def clone(self):
        fn = Function(prototype=self.prototype, env=self.env)
        for opt in self.options:
            fn.assign_option(opt.pattern,
                             opt.value.clone() if opt.value else opt.block or opt.fn)
        return fn

    def __repr__(self):
        if self == Context.root:
            return 'root'
        if len(self.options) == 1 or True:
            return f"{{{self.options}}}"
        else:
            return f"{{{self.block}}}"


Op = {}
BuiltIns = {}


class Operator:
    text: str
    prefix = False
    postfix = False
    binop = False
    ternary = False
    precedence: int
    associativity: str
    fn: Function

    def __init__(self, text, fn=None,
                 prefix=None, postfix=None, binop=None, ternary=None,
                 associativity='left', static=False):
        Op[text] = self
        self.text = text
        # self.precedence = precedence
        self.fn = fn
        self.associativity = associativity  # 'right' if 'right' in flags else 'left'
        self.prefix = prefix  # 'prefix' in flags
        self.postfix = postfix  # 'postfix' in flags
        self.binop = binop  # 'binop' in flags
        self.static = static  # 'static' in flags
        self.ternary = ternary
        assert self.binop or self.prefix or self.postfix or self.ternary

    def prepare_args(self, lhs, mid, rhs) -> list[Value]:
        raise NotImplemented('Operator.prepare_args not implemented')

    def __repr__(self):
        return self.text


def clone(val: Value | Function):
    match val:
        case Function():
            return val.clone()
        case Pattern():
            return val
        case Value():
            return Value(val.value, val.type)


if __name__ == "__main__":
    pass
