import math
import types
from fractions import Fraction
from enum import Enum

import Env
from Syntax import Block, FunctionLiteral
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
                score = int(arg == prototype) or (arg.instanceof(prototype)) * 35/36
            case Union(patterns=patterns):
                count = len(self)
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
        return 1/36
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
        return f"{'!' * self.inverse}{self.pattern}" + (' ' + self.name if self.name else '') + self.quantifier

class ValuePattern(Pattern):
    def __init__(self, value, name: str = None):
        super().__init__(name)
        self.value = value
    def __eq__(self, other):
        return isinstance(other, ValuePattern) and self.value == other.value and self.name == other.name
    def __hash__(self):
        return hash((self.value, self.name))
    def __repr__(self):
        return "ValuePattern("+str(self.value)+")"

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

class StateOLD:
    """ a state has EITHER a pattern and success, OR two branches """
    def __init__(self, param=None, success=None, branches: tuple = None):
        self.pattern = getattr(param, 'pattern', None)
        self.multi = getattr(param, 'quantifier', None) in ("+", "*")
        self.success = success
        self.branches = branches
    def __repr__(self):
        if self.pattern:
            return f"{self.pattern} --> {self.success}"
        elif self.branches:
            return "< branch"
        else:
            return "(SUCCESS)"


class State:
    """ a state has EITHER a pattern and success, OR two branches """
    pattern = None
    success = None
    branches = None
    def __init__(self):
        pass
    def leaves(self):
        return {self}


class ParamState(State):
    # pattern: Pattern | None
    # name: str
    # multi: bool
    param: Parameter
    success: State
    def __init__(self, param: Parameter, success: State):
        Parameter.__init__(self, param.pattern, param.name, param.quantifier, param.inverse)
        self.param = param
        self.pattern = param.pattern
        self.name = param.name
        self.multi = param.multi
        self.success = success
    def __repr__(self):
        return f"State{repr((self.param, self.success))}"


class Branch(State):
    branches: tuple[State, State]
    def __init__(self, b1: State, b2: State):
        self.branches = (b1, b2)
    def leaves(self):
        return set.union(self.branches[0].leaves(), self.branches[1].leaves())


Success = State()

class Instruction:
    pass
    # Char = 'char'
    # Jump = 'jump'
    # Split = 'split'
    # Save = 'save'
    # Match = 'match'

class Char(Instruction):
    pattern: Pattern
    def __init__(self, pattern: Pattern):
        self.pattern = pattern
    def __repr__(self):
        return f"Char({self.pattern})"

class Jump(Instruction):
    index: int
    def __init__(self, index: int):
        self.index = index
    def __repr__(self):
        return f"Jump({self.index})"

class Split(Instruction):
    index1: int
    index2: int
    def __init__(self, i1: int, i2: int):
        self.index1 = i1
        self.index2 = i2
    def __repr__(self):
        return f"Split{(self.index1, self.index2)}"

class Mark(Instruction):
    pass
class Save(Instruction):
    index: int
    multi: bool
    def __init__(self, index: int, multi):
        self.index = index
        self.multi = multi
    def __repr__(self):
        return f"Save({self.index})"

class Match(Instruction):
    pass

def copy_bindings(saves):
    new_saves = {}
    for key, value in saves.items():
        if isinstance(value.value, list):
            new_saves[key] = Value(value.value.copy())
        else:
            new_saves[key] = value
    return new_saves


class ListPatt(Pattern):
    def __init__(self, *parameters: Parameter):
        if True:
            self.__init__1(*parameters)
        else:
            super().__init__()
            self.parameters = tuple(parameters)
            if len(parameters) == 1:
                self.name = parameters[0].name
        # ***********************************
        self.instructions = []
        for i, param in enumerate(parameters):
            i_inst = len(self.instructions)
            self.instructions.append(Mark())
            match param.quantifier:
                case "":
                    self.instructions.append(Char(param.pattern))
                case "?":
                    self.instructions.append(Split(i_inst+1, i_inst+2))
                    self.instructions.append(Char(param.pattern))
                case "??":
                    self.instructions.append(Split(i_inst + 2, i_inst + 1))
                    self.instructions.append(Char(param.pattern))
                case "+":
                    self.instructions.append(Char(param.pattern))
                    self.instructions.append((Split(i_inst, i_inst + 2)))
                case "+?":
                    self.instructions.append(Char(param.pattern))
                    self.instructions.append((Split(i_inst + 2, i_inst)))
                case "*":
                    self.instructions.append(Jump(i_inst + 2))
                    self.instructions.append(Char(param.pattern))
                    self.instructions.append(Split(i_inst + 1, i_inst + 3))
                case "*?":
                    self.instructions.append(Jump(i_inst + 2))
                    self.instructions.append(Char(param.pattern))
                    self.instructions.append(Split(i_inst + 3, i_inst + 1))
            if param.name:
                self.instructions.append(Save(param.name, param.multi))
        self.instructions.append(Match())

    def __init__0(self, *parameters: Parameter, name=None, guard=None):
        super().__init__(name, guard)
        self.parameters = tuple(parameters)
        if self.name:
            pass
        if len(parameters) == 1:
            self.name = parameters[0].name
    def __init__1(self, *parameters: Parameter):
        super().__init__()
        self.parameters = tuple(parameters)
        if len(parameters) == 1:
            self.name = parameters[0].name
        # **********************
        if len(parameters) == 0:
            self.start = Success
        else:
            p = parameters[0]
            next = ListPatt(*parameters[1:]).start
            this = ParamState(p, next)
            split = Branch(this, next)
            match p.quantifier:
                case "":
                    self.start = this
                case "?":
                    self.start = split
                case "*":
                    self.start = split
                    this.success = self.start
                case "+":
                    self.start = this
                    this.success = split

    def match_zip_VM(self, args: list = None, p_start=0, i_inst=0, i_arg=0, score=0, saves=None):
        if saves is None:
            saves = {}
        while True:
            match self.instructions[i_inst]:
                case Char(pattern=patt):
                    if i_arg >= len(args):
                        return 0
                    match_value = patt.match_score(args[i_arg])
                    if not match_value:
                        return 0
                    score += match_value
                    i_arg += 1
                    i_inst += 1
                case Jump(index=index):
                    i_inst = index
                case Split(index1=i1, index2=i2):
                    case1 = self.match_zip_VM(args, i1, i_arg, score)
                    if case1:
                        return case1
                    i_inst = i2
                case Mark():
                    p_start = i_arg
                case Save(name=name, multi=multi):
                    pass
                case Match():
                    return (score / len(args), saves)
        return 0

    def match_zip(self, args):
        if args is None:
            return 1, {}
        # if len(args) == 0 == len(self.parameters):
        #     return 1, {}
        if not self.min_len() <= len(args) <= self.max_len():
            return 0, {}
        return self.match_zip2(args)


    def match_zip2(self, args: list = None, i_inst=0, i_arg=0, score=0, sub_score=0, saves=None):
        if saves is None:
            saves = {}
        while True:
            if not (i_inst < len(self.parameters) and i_arg < len(args)):
                if i_inst == len(self.parameters) and i_arg == len(args):
                    break
                # elif i_arg >= len(args):
                #     return 0
                elif i_inst >= len(self.parameters):
                    pass
            # if i_arg >= len(args):
            #     return 0
            param = self.parameters[i_inst]
            key: str|int = param.name or i_arg
            sub_score *= param.multi
            match_value = param.pattern.match_score(args[i_arg]) if i_arg < len(args) else 0
            # if not param.optional and not match_value:
            #     return 0
            match param.quantifier:
                case "":
                    # match patt, save, and move on
                    if not match_value:
                        return 0, {}
                    saves[key] = args[i_arg]
                    score += match_value
                    i_arg += 1
                    i_inst += 1
                case "?":
                    # try match patt and save... move on either way
                    if match_value:
                        branch_saves = copy_bindings(saves)
                        branch_saves[key] = args[i_arg]
                        branch = self.match_zip2(args, i_inst+1, i_arg+1, score+match_value, 0, branch_saves)
                        if branch[0]:
                            return branch
                    # saves[key] = Value(None)  # for some reason this line causes error later on in the execution!  I have no idea why, but I'll have to debug later
                    i_inst += 1
                case "+":
                    if key not in saves:
                        if not match_value:
                            return 0, {}
                        saves[key] = Value([])
                    if match_value:
                        branch_saves = copy_bindings(saves)
                        branch_saves[key].value.append(args[i_arg])
                        sub_score += match_value
                        i_arg += 1
                        branch = self.match_zip2(args, i_inst, i_arg, score, sub_score, branch_saves)
                        if branch[0]:
                            return branch
                    score += sub_score / len(saves[key].value)
                    i_inst += 1
                case "*":
                    if key not in saves:
                        saves[key] = Value([])
                    if match_value:
                        branch_saves = copy_bindings(saves)
                        branch_saves[key].value.append(args[i_arg])
                        branch = self.match_zip2(args, i_inst, i_arg + 1, score, sub_score + match_value, branch_saves)
                        if branch[0]:
                            return branch
                    if len(saves[key].value):
                        score += sub_score / len(saves[key].value)
                    else:
                        score += 1/36
                    i_inst += 1
        return score/len(self.parameters), saves

    def match_zip1(self, args: list = None):
        options: dict[str, Function] = {}
        states = self.start.leaves()
        # if self.start.branches:
        #     states = set(self.start.branches)
        # else:
        #     states = {self.start}
        for i, arg in enumerate(args):
            next_states: set[ParamState] = set()
            for state in states:
                if state.branches:
                    pass
                if state.pattern and state.pattern.match_score(arg):
                    name = state.name
                    if name:
                        if state.multi:
                            if name not in options:
                                options[name] = Value([])
                            options[name].value.append(arg)
                        else:
                            options[name] = arg
                    next_states.update(state.success.leaves())
                    # if state.success.branches:
                    #     next_states.update(state.success.branches)
                    # else:
                    #     next_states.add(state.success)
            if not next_states:
                return 0
            states = next_states
        if Success in states:
            return options
        return 0

    def zip(self, args: list):
        d = {}
        if args is None:
            return d
        named_options = self.match_zip(args)
        for key, value in named_options.items():
            patt = ListPatt(Parameter(key))
            d[patt] = value
        return d
        if not self.min_len() <= len(args) <= self.max_len():
            pass
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
        return self.match_zip(list_value)[0]
        if list_value.type != BuiltIns['list']:
            return 0
        args = list_value.value
        if not self.min_len() <= len(args) <= self.max_len():
            return 0
        return self.match_zip2(args) and self.match_zip2(args)[0]
        return self.match_zip_VM(args)
        return self.match_zip(args) != 0
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
            if not p_score and not param.optional:
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
        match val.value:
            case Pattern():
                return val.value
            case str() as name:
                return ValuePattern(val, name)
            case _:
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
        if hasattr(block, 'statements'):
            self.exprs = list(map(Context.make_expr, block.statements))
        else:
            self.native = block
        self.env = env or Context.env
    def make_function(self, options, env=None):
        return Function(options=options, type=env, env=env or self.env)
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
            result = self.native(scope, *(args or []))
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

    def __repr__(self):
        if self.native:
            return 'FuncBlock(native)'
        if len(self.exprs) == 1:
            return f"FuncBlock({self.exprs[0]})"
        return f"FuncBlock({len(self.exprs)} exprs)"


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
    def resolve(self, args=None, env=None, bindings={}):
        if self.value:
            return self.value
        if self.fn:
            return self.fn(*args)
        if self.block is None:
            raise NoMatchingOptionError("Could not resolve null option")
        if self.dot_option:
            env = args[0]
            # btw: this is possibly the third time the env is getting overriden: it's first set when the FuncBlock is
            # defined, then overriden by the env argument passed to self.resolve, and finally here if it is a dot-option
            # ... I should consider making a multi-layer env, or using the prototype, or multiple-inheritance type-thing
        # fn = self.block.make_function(self.pattern.match_zip(args)[1], env)
        fn = self.block.make_function(bindings, env)
        Context.push(Context.line, fn, self)
        return self.block.execute(args, fn)
    def __eq__(self, other):
        return isinstance(other, Option) and (self.pattern, self.resolution) == (other.pattern, other.resolution)

    def __repr__(self):
        if self.value:
            return f"{self.pattern}={self.value}"
        if self.block or self.fn:
            return f"{self.pattern}: {self.block or self.fn}"
        return f"{self.pattern} -> null"


class Function:
    return_value = None
    value = NotImplemented
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
        opt = self.select_by_pattern(pattern)
        if opt is None:
            raise NoMatchingOptionError(f'cannot find option "{pattern}" to remove')
        opt.nullify()

    def assign_option(self, pattern, resolution=None):
        opt = self.select_by_pattern(pattern)
        if opt is None:
            return self.add_option(pattern, resolution)
        opt.nullify()
        opt.assign(resolution)
        return opt

    def index_of(self, key) -> int:
        idx = None
        high_score = 0
        arg_list = Value(key)
        for i, opt in enumerate(self.options):
            score = opt.pattern.match_score(arg_list)
            score, saves = opt.pattern.match_zip(arg_list)
            if score == 1:
                return i
            if score > high_score:
                high_score = score
                idx = i
        if idx is None:
            raise IndexError
        return idx

    def get_option_and_binding_inner(self, key):
        option = saves = None
        high_score = 0
        arg_list = Value(key)
        for opt in self.options:
            score, saves = opt.pattern.match_zip(arg_list)
            if score == 1:
                return opt, saves
            if score > high_score:
                high_score = score
                option = opt
        if option is None:
            raise IndexError
        return option, saves

    def select_and_bind(self, key: list, walk_prototype_chain=True, ascend_env=False):
        if len(key) == 1 and key[0].instanceof(BuiltIns['str']):
            try:
                return self.select_by_name(key[0].value, ascend_env), {}
            except NoMatchingOptionError:
                pass
        option = bindings = saves = high_score = 0
        for opt in self.options:
            score, saves = opt.pattern.match_zip(key)
            if score == 1:
                return opt, saves
            if score > high_score:
                high_score = score
                option, bindings = opt, saves
        if option:
            return option, bindings
        # try:
        #     match key:
        #         case ListPatt():
        #             return [opt for opt in self.options if opt.pattern == key][0], {}
        #         case str() as key:
        #             opt = self.named_options[key]
        #             # I may want to remove this case
        #             # BECAUSE foo.string should NOT match a possible [any] pattern on foo.
        #             # It should ONLY match "string" named option on foo first, then string function globally second
        #         case [arg] if arg.instanceof(BuiltIns['str']):
        #             opt = self.named_options.get(arg.value, None)
        #         case _:
        #             assert isinstance(key, list)
        #             opt = None
        #     if opt:
        #         return opt, {}
        #     return self.get_option_and_binding_inner(key)
        # except (KeyError, IndexError):
        #     pass
        if walk_prototype_chain and self.type:
            try:
                return self.type.select_and_bind(key, True, ascend_env)
            except NoMatchingOptionError:
                pass
        if ascend_env and self.env:
            try:
                return self.env.select_and_bind(key, walk_prototype_chain, True)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in function {self}")
        # -> tuple[Option, dict[str, Function]]:

    def select_by_pattern(self, patt=None, default=None, ascend_env=False):
        # return [*[opt for opt in self.options if opt.pattern == patt], None][0]
        for opt in self.options:
            if opt.pattern == patt:
                return opt
        if ascend_env and self.env:
            return self.env.select_by_pattern(patt, default)
        return default

    def select_by_name(self, name: str, ascend_env=True):
        if name in self.named_options:
            return self.named_options[name]
        env = self.type
        while env:
            if name in env.named_options:
                return env.named_options[name]
            env = env.type
        # if self.type:
        #     try:
        #         return self.type.select_by_name(name, False)
        #     except NoMatchingOptionError:
        #         pass
        if ascend_env and self.env:
            try:
                return self.env.select_by_name(name, True)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: '{name}' not found in current context")

    def select(self, key, walk_prototype_chain=True, ascend_env=False):
        try:
            match key:
                case ListPatt():
                    return [opt for opt in self.options if opt.pattern == key][0]
                case str() as key:
                    opt = self.named_options[key]
                    # I may want to remove this case
                    # BECAUSE foo.string should NOT match a possible [any] pattern on foo.
                    # It should ONLY match "string" named option on foo first, then string function globally second
                case [arg] if arg.instanceof(BuiltIns['str']):
                    opt = self.named_options.get(arg.value, None)
                case _:
                    assert isinstance(key, list)
                    opt = None
            if opt:
                return opt
            return self.options[self.index_of(key)]
        except (KeyError, IndexError):
            pass
        if walk_prototype_chain and self.type:
            try:
                return self.type.select(key, True, ascend_env)
            except NoMatchingOptionError:
                pass
        if ascend_env and self.env:
            try:
                return self.env.select(key, walk_prototype_chain, True)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in function {self}")

    def call(self, key, ascend=False):
        try:
            # option = self.select(key)
            option, bindings = self.select_and_bind(key)
            # for opt in self.options:
            #     bindings = opt.pattern.match_zip(key)
        except NoMatchingOptionError as e:
            if ascend and self.env:
                try:
                    return self.env.call(key, ascend=True)
                except NoMatchingOptionError:
                    pass
            raise e
        return option.resolve(key, self, bindings)

    def deref(self, name: str, ascend_env=True):
        # option = self.select(name, ascend_env=ascend_env)
        option = self.select_by_name(name, ascend_env)
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

    def to_string(self):
        if hasattr(self, 'value') and self.value is not NotImplemented:
            if self.instanceof(BuiltIns['num']) and not self.type == BuiltIns['bool']:
                return Value(write_number(self.value))
            if self.instanceof(BuiltIns['list']):
                return Value(f"[{', '.join(v.to_string().value for v in self.value)}]")
            return Value(str(self.value))
        if self.name:
            return Value(self.name)
        # if self.instanceof(BuiltIns['BasicType']):
        #     return Value(self.name)
        return Value(str(self))

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Function):
            return False
        if self.type != other.type:
            return False
        if getattr(self, "value", object()) == getattr(other, "value", object()) and self.value is not NotImplemented:
            return True
        if self.env != other.env or self.name != other.name:
            return False
        # try:
        #     for opt in self.options:
        #         assert opt.resolution == other.select(opt.pattern).resolution
        # except (NoMatchingOptionError, AssertionError):
        #     return False
        for opt in self.options:
            if opt.resolution != getattr(other.select_by_pattern(opt.pattern), 'resolution', None):
                return False
        return True

    def __repr__(self):
        # if self is Context.root:
        #     return 'root'
        if self.type is Context.root:
            return 'root.main'
        if self.value is not NotImplemented:
            try:
                return repr(self.value)  # noqa
            except AttributeError:
                pass
        prefix = self.name or ""
        return prefix + "{}"
        try:
            if len(self.options) == 1:
                return f"{prefix}{{{self.options[0]}}}"
            else:
                return f"{prefix}{self.named_options}"
        except RecursionError:
            return prefix

class Value(Function):
    def __init__(self, value, type_=None):
        if isinstance(value, Fraction) and value.denominator == 1:
            self.value = int(value)
        else:
            self.value = value
        super().__init__(type=type_ or TypeMap[type(self.value)])

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
        return hasattr(other, 'value') and self.value == other.value
        try:
            # assume List
            return tuple(self.value) == tuple(other.value)
        except (TypeError, AttributeError):
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
        if fn:
            if not fn.name:
                fn.name = text
            BuiltIns[text] = fn
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
