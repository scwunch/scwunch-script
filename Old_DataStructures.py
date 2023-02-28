from DataStructures import BasicType, Block, type_mapper


class Value:
    value: any
    type: BasicType

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

    def set_type(self, basic_type: BasicType = None):
        if basic_type:
            self.type = basic_type
        else:
            match self.value:
                case BasicType(): self.type = BasicType.Type
                case Function(): self.type = BasicType.Function
        elif isinstance(self.value, BasicType):
            self.type = BasicType.Type
        else:
            self.type = type_mapper(type(self.value))
        return self.type

    def is_null(self):
        return self.value is None
    def not_null(self):
        return self.value is not None

    def __eq__(self, other):
        return isinstance(other, Value) and self.value == other.value

    def __hash__(self):
        return hash(self.value)
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return f"<{self.type.value}:{repr(self.value)}>"


# class Type(Value):
#     base: BasicType
#     # union_types:
#     # condition: (any) -> bool
#     # function_class: any
#     def __init__(self, value=None, python_type: type | None = None):
#         self.value = self
#         self.type = self
#         self.base = BasicType.Type
#         if isinstance(value, Block):
#             self.base = BasicType.Function
#         elif isinstance(value, Pattern):
#             self.base = BasicType.Pattern
#         elif isinstance(value, BasicType):
#             self.base = value
#         elif isinstance(value, Type):
#             self.base = BasicType.Type
#         else:
#             # assert python_type or value ~ (Value | str)
#             key = python_type or (value if isinstance(value, str) else type(value.value))
#             self.base = type_mapper(key)
#
#     def __str__(self):
#         return self.base.value
# function tyoe pattern
# class Class:
#     required_options: list[Key]
#
#
# class PatternGroup(Pattern):
#     patterns: list[Pattern]
#     union: True # if false, treat as intersection
#     def __init__(self, patterns: list[Pattern], union=True):
#         self.patterns = patterns
#         self.union = bool(union)
#
#
#
#
# class Key:
#     source: list[Node]
#     name: str
#     pattern: Pattern
#     # type: Key_Type
#
#     def __init__(self, nodes: list[Node]):
#         self.source = nodes
#         last_node = nodes[-1]
#         if last_node.type == TokenType.name:
#             self.name = last_node.source_text
#             self.type = MatchPatternType.Value
#         if len(nodes) > 1:
#             self.pattern = Pattern(nodes[:-1])
#
#     def __str__(self):
#         return "".join(map(str, self.source))
#         # return reduce(Token.__add__, self.source, "")
#
#     def __repr__(self):
#         return f'Key({self}): {self.type}'


class Parameter:
    name: str | None
    value: Value | None
    base_types: frozenset[BasicType]
    fn: any    # (any) -> bool

    def __init__(self, name: str | None = None, value: Value = None, basic_type=None, fn=None):
        assert (0 if name else 1) <= bool(value) + bool(basic_type) < 2
        self.name = name
        self.value = value
        try:
            self.base_types = frozenset(basic_type or [])
        except TypeError:
            self.base_types = frozenset([basic_type])
        assert bool(fn) <= bool(basic_type)
        self.fn: Function = fn

    # def copy(self):
    #     return Parameter(self.name, self.value, self.base_types, self.fn)
    def specificity(self):
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
def match_score(val: Value, param: Parameter):
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
This class is larger than Type, which just refers to the single type of a value.
A Pattern is like a regex for types; it can match one very specific type, or even one specific value
or it can match a type on certain conditions (eg int>0), or union of types
"""
class Pattern(Value):
    required_parameters: tuple[Parameter, ]
    optional_parameters: tuple[Parameter, ]
    def __init__(self, *required_params: Parameter, optional_parameters=None):
        super().__init__(required_params, BasicType.Pattern)
        self.required_parameters = required_params
        self.optional_parameters = optional_parameters or tuple([])
        self.specificity = sum(p.specificity() for p in self.all_parameters)
    @property
    def all_parameters(self):
        return self.required_parameters + self.optional_parameters
    # def specificity(self):
    #     return sum(p.specificity for p in self.all_parameters)
    def __len__(self):
        return len(self.required_parameters)
    def __getitem__(self, item):
        return self.required_parameters[item]
    def __repr__(self):
        return f"[{', '.join(map(repr, self.required_parameters))}]"
    def __str__(self):
        return f"[{', '.join(map(repr, self.required_parameters))}]"
    def __eq__(self, other):
        return (self.required_parameters, self.optional_parameters) \
                == (other.required_parameters, other.optional_parameters)
    def __hash__(self):
        return hash((self.required_parameters, self.optional_parameters))


class Option:
    pattern: Pattern
    value: Value
    def __init__(self, params: Pattern, value: Value = None):
        # super().__init__(value)
        # self.type = Type(basic_type=BasicType.Option)
        self.pattern = params
        self.value = value or Value(None)
        # if isinstance(block_or_value, Block):
        #     self.block = block_or_value
        # elif isinstance(block_or_value, Value):
        #     self.value = block_or_value
        # else:
        #     raise Exception('wrong value type to initialize option; expected Block or Value but got ', block_or_value)

    def is_null(self):
        return self.value.is_null()

    def __repr__(self):
        return f"{self.pattern} => {self.value}"


class OptionTree(dict):
    value: Value
    def __init__(self, value: Value = None):
        super().__init__()
        self.value: Value = value

    def __getitem__(self, item):
        if isinstance(item, Parameter):
            return self.get(item.name or item.value or item.base_types, None)
        elif isinstance(item, Value):
            name = item.value if item.type == BasicType.String else None
            return self.get(name, self.get(item, self.get(item.type, None)))

    def __setitem__(self, key: Parameter, value: dict | Value):
        if key.name:
            super().__setitem__(key.name, value)
        if key.value or key.base_types:
            super().__setitem__(key.value or key.base_types, value)


class _Function(Value):
    prototype: Value | None
    # options: list[Option]


class Function(_Function):
    prototype: _Function | None  # class-like inheritance
    args: list[Value]
    options: list[Option]
    named_options: dict[str, Option]
    block: Block
    dot_arg = None
    env: _Function | None  # namespace
    exec: any
    return_value = None

    def __init__(self, opt_pattern=None, opt_value=None, options=None, block: Block = None, prototype=None, env=None):
        self.block = block or Block([])
        super().__init__(self.block, BasicType.Function)
        self.options = []  # [Option(Pattern(), self)]
        self.named_options = {}
        if options:
            for patt, val in options.items():
                self.add_option(patt, val)
        if opt_pattern:
            self.assign_option(opt_pattern, opt_value)
        self.prototype: Function = prototype
        self.env: Function = env
        # if prototype:
        #     self.options += [opt.copy() for opt in prototype.options.copy()]

    # def get_option_tree(self, key: tuple[Value, ]) -> Value | None:
    #     d = self.options
    #     for arg in key:
    #         d = d[arg]
    #         if not d:
    #             return None
    #     return d.value
    #
    # def assign_option_tree(self, pattern: tuple[Parameter, ], value: Value):
    #     d = self.options
    #
    #     for param in pattern:
    #         d_next = d[param]
    #         if not d_next:
    #             d_next = d[param] = OptionTree()
    #         d = d_next
    #     print('setting option value to ', value)
    #     d.value = value
    #     print('just set option value to ', d.value)
    #     return value

    def add_option(self, pattern: Pattern, value: Value = None):
        opt = Option(pattern, value)
        if len(pattern) == 1 and pattern[0].name:
            self.named_options[pattern[0].name] = opt
        self.options.insert(0, opt)
        self.options.sort(key=lambda opt: opt.pattern.specificity, reverse=True)
        return opt

    def assign_option(self, pattern: Pattern, value: Value) -> Value:
        opt = self.select(pattern)
        opt.value = value
        return value
        # i = self.index_of(pattern)
        # if i is None:
        #     self.add_option(pattern, value)
        #     return value
        # else:
        #     return self.options[i].value.set_value(value)

    def index_of(self, key: Pattern | list[Value]) -> int | None:
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
                return bool(param.fn.call([val]).value)
            # other subtype checks?
            return True

        for i, opt in enumerate(self.options):
            params = opt.pattern.all_parameters
            if not (len(opt.pattern.required_parameters) <= len(key) <= len(params)):
                continue
            for j in range(len(params)):
                if j == len(key):
                    return i
                arg, param = key[j], params[j]
                if arg is None or not is_match(arg, param):
                    if arg is None:
                        breakpoint()  # why is arg None here?
                    break  # no match; continue outer loop
                    # but
            else:
                return i
        return None

    def select(self, key: Pattern | list[Value]) -> Option | None:
        """
        :param key: list of args, or pattern of params
        :returns the matching option, creating a null option if none exists
        """
        if isinstance(key, Pattern):
            opt = [opt for opt in self.options if opt.pattern == key]
            return opt[0] if opt else self.add_option(key)

        if len(key) == 1 and key[0].type == BasicType.String:
            option = self.named_options.get(key[0].value, None)
            if option:
                return option

        i = self.index_of(key)
        if i is not None:
            return self.options[i]
        if None in key:
            breakpoint()
        # return phantom option
        return self.add_option(Pattern(*(Parameter(value=k) for k in key)))

    def call(self, key: list[Value] | str, copy_option=True, ascend=False) -> Value | None:
        if isinstance(key, str):
            # option = self.named_options.get(key, self.env.call(key) if self.env else None)
            key = [Value(key, BasicType.String)]
            ascend = True
        option = self.select(key)
        if option.value.is_null() and ascend and self.env:
            option = self.env.call(key, ascend=True)
        if option.value.is_null():
            # raise Exception("Function called on args with no matching option.")
            print('Warning: Function called on args with no matching option.')
        val = option.value
        if isinstance(val, Function):
            fn = val.init(option.pattern, key, self, copy_option)
            return fn.execute()
        return val

    def execute(self):
        return self.exec()

    # def deref(self, name: str) -> Value | None:
    #     opt = self.named_options.get(name, None)
    #     if opt:
    #         if isinstance(opt.value, Function):
    #             fn = opt.value.init(opt.pattern, [Value(name, BasicType.String)], self, False)
    #         return opt.value
    #     if self.env:
    #         return self.env.deref(name)
    #     return None
    #
    # def init_option(self, option: Option, key: list[Value]):
    #     assert isinstance(option.value, Function)
    #     fn = Function(block=option.value.block, prototype=self)
    #     for arg, param in zip(key, option.pattern.required_parameters):
    #         fn.add_option(Option(Pattern(param), arg))
    #     fn.args = key
    #     return fn

    def init(self, pattern: Pattern, key: list[Value], parent, copy=True):
        fn = Function(block=self.block, prototype=parent, env=parent.env) if copy else self
        for arg, param in zip(key, pattern.required_parameters):
            fn.assign_option(Pattern(param), arg)
        fn.args = key
        return fn

    def clone(self):
        fn = Function(block=self.block, prototype=self, env=self.env)
        for opt in self.options:
            fn.assign_option(opt.pattern, clone(opt.value))
        return fn

    def __repr__(self):
        if self == Context.root:
            return 'root'
        if len(self.options) == 1 or True:
            return f"{{{self.options} => Block: {self.block}}}"
        else:
            return f"{{{self.block}}}"


class Action(Value):
    action: str  # return, assign,
    def __init__(self, value, action: str, data=None):
        super().__init__(value)
        self.action = action
        if action == 'assign':
            self.pattern = data


class Native(Function):
    fn: any

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def execute(self):
        return self.fn(*self.args)

    def init(self, pattern: Pattern, key: list[Value], parent, copy=False):
        self.args = key
        return self

    def __repr__(self):
        return 'Native'


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

    def __init__(self, text, fn=None, prefix=None, postfix=None, binop=None, ternary=None, associativity='left', static=False):
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
        raise Exception('Operator.prepare_args not implemented')

    def __repr__(self):
        return self.text


class Context:
    _env = []
    env: Function
    line = 0
    root = None

    @staticmethod
    def push(env: Function):
        Context._env.append(env)
        Context.env = env

    @staticmethod
    def pop():
        Context._env.pop()
        Context.env = Context._env[-1]


def clone(val: Value):
    match val:
        case Function():
            return val.clone()
        case Pattern():
            return val
        case Value():
            return Value(val.value, val.type)






if __name__ == "__main__":
    pass
