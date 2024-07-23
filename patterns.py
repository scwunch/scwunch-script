import math
from tables import *

print(f"loading module: {__name__} ...")

class Pattern(Record):
    def __init__(self):
        super().__init__(BuiltIns['Pattern'])

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        raise NotImplementedError(self.__class__.__name__)

    def match_and_bind(self, arg: Record):
        match = self.match(arg)
        if match is None:
            raise MatchErr(f"Line {Context.line}: "
                           f"pattern '{self}' did not match value {arg}")
        for target, value in match:
            target.bind(value)


    def bytecode(self):
        raise NotImplementedError(self.__class__.__name__)


class Matcher:
    # guard = None
    invert = False
    hash: int
    # def __init__(self, name: str = None, guard: Function | PyFunction = None, inverse=False):
    #     if name is not None or guard is not None or inverse:
    #         raise Exception("Check this out.  Can we get rid of these properties entirely?")
    #     self.name = name
    #     self.guard = guard
    #     self.invert = inverse
    #     super().__init__()

    def match_score(self, arg: Record) -> None | dict[str, Record]:
        if self.invert:
            raise NotImplementedError
            return not self.basic_score(arg)
        try:
            return self.match(arg)
        except NotImplementedError:
            if self.basic_score(arg):
                return {}
        # return self.basic_score(arg)

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        if self.basic_score(arg):
            return {}
        # raise NotImplementedError(f'Implement {self.__class__.__name__}.match()')

    def basic_score(self, arg):
        # implemented by subclasses
        raise NotImplementedError

    def issubset(self, other):
        print('WARNING: Matcher.issubset method not implemented properly yet.')
        return self.equivalent(other)

    def equivalent(self, other):
        return True
        # return (other.guard is None or self.guard == other.guard) and self.invert == other.invert

    def bytecode(self):
        return VM(self)

    # def call_guard(self, arg: Record) -> bool:
    #     if self.guard:
    #         result = self.guard.call(arg)
    #         return BuiltIns['bool'].call(result).value
    #     return True

    def get_rank(self):
        return self.rank
        # rank = self.rank
        # if self.invert:
        #     rank = tuple(100 - n for n in rank)
        # if self.guard:
        #     rank = (rank[0], rank[1] - 1, *rank[1:])
        # return rank

    def __lt__(self, other):
        return self.get_rank() < other.get_rank()

    def __le__(self, other):
        match other:
            case IntersectionMatcher(matchers=patterns):
                return all(self < p for p in patterns)
            case UnionMatcher(matchers=patterns):
                return any(self <= p for p in patterns)
            case Matcher():
                return self.get_rank() <= other.get_rank()
            # case Parameter(pattern=pattern):
            #     return self <= pattern

            case _:
                return NotImplemented

    def __repr__(self):
        return (f"{'!' * self.invert}{self.__class__.__name__}"
                f"{tuple(v for k, v in self.__dict__.items() if k not in ('hash', 'invert'))}")

    def __eq__(self, other):
        return (self.__class__ is other.__class__
                and hash(self) == hash(other)
                and self.__dict__ == other.__dict__)
        # return self.invert == other.invert and self.guard == other.guard
        # raise NotImplementedError(f'Please implement __eq__ of {self.__class__}')

    def __hash__(self):
        try:
            return self.hash
        except AttributeError:
            props = dict(self.__dict__)
            props['class'] = self.__class__
            self.hash = hash(frozenset(props.items()))
            return self.hash

class TableMatcher(Matcher):
    table: Table
    rank = 5, 0

    def __init__(self, table):
        assert isinstance(table, Table)
        self.table = table

    def basic_score(self, arg: Record) -> bool:
        return arg.table == self.table

    def issubset(self, other):
        match other:
            case TableMatcher(table=table):
                return table == self.table
            case TraitMatcher(trait=trait):
                return trait in self.table.traits
        return False

    def equivalent(self, other):
        return isinstance(other, TableMatcher) and self.table == other.table

    def __repr__(self):
        return f"TableMatcher({self.table})"

class TraitMatcher(Matcher):
    trait: Trait
    rank = 6, 0

    def __init__(self, trait):
        self.trait = trait

    def basic_score(self, arg: Record) -> bool:
        return self.trait in arg.table.traits

    def issubset(self, other):
        return isinstance(other, TraitMatcher) and other.trait == self.trait

    def equivalent(self, other):
        return isinstance(other, TraitMatcher) and other.trait == self.trait

    # def __repr__(self):
    #     return f"TraitMatcher({self.trait})"


class ValueMatcher(Matcher):
    value: Record
    rank = 1, 0

    def __init__(self, value):
        self.value = value

    def basic_score(self, arg: Record) -> bool:
        return arg == self.value

    def issubset(self, other):
        match other:
            case ValueMatcher(value=value):
                return value == self.value
            case TableMatcher(table=table):
                return self.value.table == table
            case TraitMatcher(trait=trait):
                return trait in self.value.table.traits
        return False

    def equivalent(self, other):
        return isinstance(other, ValueMatcher) and other.value == self.value

class ArgsMatcher(Matcher):
    rank = 5, 0
    params = None  # : ParamSet | None = None

    def __init__(self, *params):
        match params:
            case [ParamSet() as params]:
                self.params = params
            case _ if params:
                self.params = ParamSet(*params)

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        if not isinstance(arg, Args):
            return
        if self.params is None:
            return {}
        return self.params.match(arg)


class FunctionMatcher(Matcher):
    # signature: ParamSet
    # return_type: Matcher
    def __init__(self, signature, return_type):
        self.signature = signature
        self.return_type = return_type

    def basic_score(self, arg):
        if not hasattr(arg, 'op_list'):
            return False
        arg: Function

        def options():
            yield from arg.op_list
            yield from arg.op_map.values()

        if all(option.pattern.issubset(self.signature)
               and option.return_type.issubset(self.return_type)
               for option in options()):
            return True

    def issubset(self, other):
        match other:
            case FunctionMatcher(signature=patt, return_type=ret):
                return self.signature.issubset(patt) and self.return_type.issubset(ret)
            case TraitMatcher(trait=BuiltIns.get('fn')) | TableMatcher(table=BuiltIns.get('Function')):
                return True
        return False

    def equivalent(self, other):
        return (isinstance(other, FunctionMatcher)
                and other.signature == self.signature
                and other.return_type == self.return_type)

    def __repr__(self):
        return '!' * self.invert + f"FunctionMatcher({self.signature} => {self.return_type})"


class AnyMatcher(Matcher):
    rank = 100, 0
    def basic_score(self, arg: Record) -> True:
        return True

    def issubset(self, other):
        return isinstance(other, AnyMatcher)

    def equivalent(self, other):
        return isinstance(other, AnyMatcher)

    # def __repr__(self):
    #     return f"AnyMatcher()"

class EmptyMatcher(Matcher):
    rank = 3, 0
    def basic_score(self, arg: Record) -> bool:
        match arg:
            case VirtTable():
                return False
            case PyValue(value=str() | tuple() | frozenset() | list() | set() as v) | Table(records=v):
                return len(v) == 0
            case Function(op_list=options, op_map=hashed_options):
                return bool(len(options) + len(hashed_options))
            case Record(table=Table(traits=traits)) if BuiltIns['seq'] in traits:
                return BuiltIns['len'].call(arg).value == 0
            case _:
                return False

    def issubset(self, other):
        return isinstance(other, EmptyMatcher)

    def equivalent(self, other):
        return isinstance(other, EmptyMatcher)

    # def __repr__(self):
    #     return f"EmptyMatcher()"


class IterMatcher(Matcher):
    parameters: tuple
    def __init__(self, *params):
        raise NotImplementedError
        self.parameters = params

    def basic_score(self, arg: Record):
        raise NotImplementedError
        return self.match_zip(arg)[0]

    def match_zip(self, arg: Record):
        raise NotImplementedError
        try:
            it = iter(arg)  # noqa
        except TypeError:
            return 0, {}
        state = MatchState(self.parameters, list(it))
        return state.match_zip()

def dot_fn(a: Record, b: Record, *, caller=None, suppress_error=False):
    if hasattr(a, "uninitialized"):
        raise InitializationErr(f"Line {Context.line}: "
                                f"Cannot call or get property of {a.table} {a.name or str(a)} before initialization.")
    match b:
        case Args() as args:
            return a.call(args, caller=caller)
        case PyValue(value=str() as name):
            prop = a.get(name, None)
            if prop is not None:
                return prop
            fn = a.table.get(name, Context.deref(name, None))
            if fn is None:
                if suppress_error:
                    return  # this is for pattern matching
                raise MissingNameErr(f"Line {Context.line}: {a.table} {a} has no field \"{name}\", "
                                     f"and also not found as function name in scope.")
            return fn.call(a, caller=caller)
        # case PyValue(value=tuple() | list() as args):
        #     return a.call(*args)
        case _:
            print(f"WARNING: Line {Context.line}: "
                  f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
            return a.call(b)
    # raise OperatorError(f"Line {Context.line}: "
    #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")

class FieldMatcher(Matcher):
    ordered_fields: tuple
    fields: dict  # dict[str, Parameter]
    def __init__(self, ordered_fields: tuple[Pattern, ...], fields: dict = None, **kwargs):
        self.ordered_fields = tuple(f if isinstance(f, Parameter) else Parameter(f)
                                    for f in ordered_fields)
        if fields is None:
            fields = kwargs
        else:
            fields.update(kwargs)
        for f, p in fields.items():
            if not isinstance(p, Parameter):
                fields[f] = Parameter(p)
        self.fields = frozendict(fields)

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        bindings = {}
        for key, param in self.items():
            if isinstance(key, int):
                try:
                    prop = arg.data[key]
                except IndexError:
                    if param.required:
                        raise RuntimeErr(f"Line {Context.line}: "
                                         f"{arg} does not have enough slots to unpack for Field Matcher.  "
                                         f"Try using fewer fields, or named fields instead.")
                    continue
            else:
                prop = dot_fn(arg, py_value(key), suppress_error=True)
            if prop is None:
                if param.required:
                    return None
                continue
            sub_bindings = param.match(prop)
            if sub_bindings is None:
                return None
            else:
                bindings.update(sub_bindings)
        return bindings

    def items(self):
        yield from self.fields.items()
        yield from enumerate(self.ordered_fields)

    def get_rank(self):
        return 2, -1

    # def __repr__(self):
    #     return f"FieldMatcher{self.fields}"

class ExprMatcher(Matcher):
    expr: any  # Node
    rank = 2, 0
    def __init__(self, expr):
        self.expr = expr

    def basic_score(self, arg: Record) -> bool:
        return self.expr.evaluate().truthy

class LambdaMatcher(Matcher):
    """ this matcher is only used internally, users cannot create LambdaMatchers """
    fn: PyFunction
    rank = 2, 0

    def __init__(self, fn: PyFunction):
        self.fn = fn

    def basic_score(self, arg: Record) -> bool:
        return self.fn(arg)


class NotMatcher(Matcher):
    def __init__(self, matcher: Matcher):
        self.matcher = matcher

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        if self.matcher.match(arg) is None:
            return {}

    def get_rank(self):
        return AnyMatcher.rank[0] - self.matcher.get_rank()[0], 0

    def __lt__(self, other):
        return other <= self.matcher

    def __le__(self, other):
        return other < self.matcher

class IntersectionMatcher(Matcher):
    # I'm confused.  I think I made this inherit from "Pattern" rather than "Matcher" so that you could do intersections of multiple parameters in a row
    # eg foo[(num+) & (int*, ratio*)]: ...
    # but somehow it's getting compared with matchers now.
    matchers: tuple[Matcher, ...]
    def __init__(self, *matchers: Matcher):
        # if binding is not None:
        #     raise Exception("This should be a parameter, not an Intersection.")
        match len(matchers):
            case 0:
                raise ValueError(f"Line {Context.line}: IntersectionMatcher called with 0 matchers.  "
                                 f"Use AnyMatcher(invert=True) instead.")
            case 1:
                raise ValueError(f"Line {Context.line}: IntersectionMatcher called with only 1 matcher."
                                 f"Catch this and return that single matcher na lang.")
        self.matchers = matchers

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        bindings = {}
        for m in self.matchers:
            sub_match = m.match(arg)
            if sub_match is None:
                return
            bindings.update(sub_match)
        return bindings

    def get_rank(self):
        ranks = [m.get_rank()[0] for m in self.matchers]
        return tuple(sorted(ranks))

    def issubset(self, other):
        match other:
            case Matcher() as other_matcher:
                return any(m.issubset(other_matcher) for m in self.matchers)
            case IntersectionMatcher() as patt:
                return any(matcher.issubset(patt) for matcher in self.matchers)
            case UnionMatcher(matchers=patterns):
                return any(self.issubset(patt) for patt in patterns)
            case Parameter(pattern=pattern):
                return self.issubset(pattern)
        return False

    def bytecode(self):
        try:
            inst = Inst().match_all(*self.matchers)
            inst.next = Inst().success()
            return VM(inst, inst.next)
            vm = VM(Inst().match_all(*self.matchers))
            vm.tail.success()
            return vm
        except AssertionError:
            raise NotImplementedError("I don't yet know how to do intersection of multi-parameters")
        # if any(isinstance(patt, Parameter) and patt.multi or isinstance(patt, ParamSet)
        #        for patt in self.patterns):
        #     raise NotImplementedError("I don't yet know how to do intersection of multi-parameters")

    def __lt__(self, other):
        match other:
            case IntersectionMatcher(matchers=other_matchers):
                return (len(self.matchers) > len(other_matchers)
                        or len(self.matchers) == len(other_matchers) and self.matchers < other_matchers)
            case UnionMatcher(matchers=patterns):
                return any(self <= p for p in patterns)
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case IntersectionMatcher(matchers=other_matchers):
                return (len(self.matchers) > len(other_matchers)
                        or len(self.matchers) == len(other_matchers) and self.matchers <= other_matchers)
            case UnionMatcher(matchers=patterns):
                return any(self <= p for p in patterns)
            case _:
                raise NotImplementedError

    # def __hash__(self):
    #     return hash(frozenset(self.matchers))

    # def __eq__(self, other):
    #     match other:
    #         case IntersectionMatcher(matchers=matchers):
    #             return matchers == self.matchers
    #         case UnionMatcher(matchers=(Matcher() as patt, )):
    #             return self == patt
    #         case Matcher() as m:
    #             return len(self.matchers) == 1 and self.matchers[0] == m
    #     return False

    # def __repr__(self):
    #     return f"Intersection{self.matchers}"


class UnionMatcher(Matcher):
    rank = 7, 0
    matchers: tuple[Matcher, ...]
    def __init__(self, *matchers):
        match len(matchers):
            case 0:
                raise ValueError(f"Line {Context.line}: UnionMatcher called with 0 matchers.  "
                                 f"Use AnyMatcher() instead.")
            case 1:
                raise ValueError(f"Line {Context.line}: UnionMatcher called with only 1 matcher."
                                 f"Catch this and return that single matcher na lang.")
        self.matchers = matchers

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        for m in self.matchers:
            sub_match = m.match(arg)
            if sub_match is None:
                continue
            return sub_match

    # def match_score(self, arg: Record):
    #     return any(p.match_score(arg) for p in self.matchers)

    def issubset(self, other):
        return all(p.issubset(other) for p in self.matchers)

    def bytecode(self):
        machine = VM()
        machine.tail = Inst().success()
        heads = []
        for vm in (patt.bytecode() for patt in self.matchers):
            vm.tail.jump(machine.tail)
            heads.append(vm.head)
        machine.head = Inst().split(*heads)
        return machine

    def __lt__(self, other):
        match other:
            case IntersectionMatcher():
                return all(p < other for p in self.matchers)
            case UnionMatcher(matchers=patterns):
                return self.matchers < patterns
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case UnionMatcher(matchers=patterns):
                return self.matchers <= patterns
            case Matcher() | IntersectionMatcher():
                return all(p < other for p in self.matchers)
            case _:
                raise NotImplementedError

    # def __eq__(self, other):
    #     match self.matchers:
    #         case ():
    #             return isinstance(other, UnionMatcher) and other.matchers == ()
    #         case (Pattern() as patt, ):
    #             return patt == other
    #     return isinstance(other, UnionMatcher) and set(self.matchers) == set(other.matchers)
    #
    # __hash__ = Matcher.__hash__

    # def __repr__(self):
    #     return f"Union{self.matchers}"


class Parameter(Pattern):
    pattern: Matcher | None = None
    binding: BindTarget = None  # property
    quantifier: str  # "+" | "*" | "?" | "!" | ""
    optional: bool
    required: bool
    multi: bool
    default = None
    def __init__(self, pattern, binding: BindTarget | str = None, quantifier="", default=None):
        self.pattern = patternize(pattern)
        if isinstance(binding, str):
            self.binding = BindTargetName(binding)
        self.binding = binding
        self.quantifier = quantifier
        if default:
            if self.multi:
                raise SyntaxErr(f"Line {Context.line}: parameters matching multiple args cannot have a default defined.")
            if not quantifier:
                self.quantifier = '?'
        if not default and quantifier.startswith('?'):
            self.default = BuiltIns['blank']
        else:
            self.default = default
        super().__init__()

    def issubset(self, other):
        if not isinstance(other, Parameter):
            raise NotImplementedError(f"Not yet implemented Parameter.issubset({other.__class__})")
        if self.multi and not other.multi or self.optional and other.required:
            return False
        return self.pattern.issubset(other.pattern)

    optional = property(lambda self: self.default or self.quantifier[:1] in ('?', '*'))
    required = property(lambda self: self.default is None and self.quantifier[:1] in ('', '+'))
    multi = property(lambda self: self.quantifier[:1] in ('+', '*'))

    # def match_score(self, value) -> int | float: ...
    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        bindings = self.pattern.match(arg)
        if bindings is None:
            return
        if self.binding:
            bindings.update({self.binding: arg})
        return bindings

    def compare_quantifier(self, other):
        return "_?+*".find(self.quantifier) - "_?+*".find(other.quantifier)

    def bytecode(self):
        vm: list
        match self.pattern:
            case Matcher() as matcher:
                vm = [Inst().match(matcher, self.binding if not self.multi else None)]  # , self.default)]
            case _:
                vm = self.pattern.bytecode()

        match self.quantifier:
            case '':
                pass
            case '?':
                prepend = [Inst().bind(self.binding, self.default)] * bool(self.binding) \
                          + [Inst().split(1, len(vm)+1)]
                vm[:0] = prepend
            case '+':
                vm.append(Inst().split(-len(vm), 1))
            case '*':
                vm = [Inst().jump(len(vm)+1), *vm, Inst().split(-len(vm), 1)]
            case '??':
                # prioritize the non-matching (default) branch
                prepend = [Inst().bind(self.binding, self.default)] * bool(self.binding) \
                          + [Inst().split(len(vm) + 1, 1)]
                vm[:0] = prepend
            case '+?':
                # prioritize the shortest branch
                vm.append(Inst().split(1, -len(vm)))
            case '*?':
                # prioritize the shortest branch
                vm = [Inst().jump(len(vm)+1), *vm, Inst().split(1, -len(vm))]
            case _:
                assert False

        if self.multi and self.binding:
            vm.insert(0, Inst().save(self.binding))
            vm.append(Inst().save(self.binding))

        return vm

    def __lt__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern < other.pattern
            case Matcher():
                return self < Parameter(other)
        return NotImplemented

    def __le__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern <= other.pattern
            case Matcher():
                return self <= Parameter(other)
        return NotImplemented

    def __eq__(self, other):
        match other:
            case Parameter() as param:
                pass
            case Matcher() | Pattern():
                param = Parameter(other)
            case ParamSet(parameters=(param, ), named_params={}):
                pass
            case _:
                return False
        return self.quantifier == param.quantifier and self.pattern == param.pattern and self.default == param.default

    def __hash__(self):
        return hash((self.pattern, self.quantifier, self.default))

    def __gt__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern > other.pattern
            case Matcher():
                return self > Parameter(other)
        return NotImplemented

    def __ge__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern >= other.pattern
            case Matcher():
                return self >= Parameter(other)
        return NotImplemented

    def __repr__(self):
        return (f"Parameter({self.pattern}{' ' + self.binding if self.binding else ''}{self.quantifier}"
                f"{'='+str(self.default) if self.default else ''})")

class Inst:
    opcode: str = 'tail'
    next = None  # Inst | None
    matcher: Matcher = None
    matchers: tuple[Matcher, ...] = ()
    i: int = None
    name: str = None
    binding: str = None
    default: Record = None
    branches = None
    Match = 'Match'
    MatchName = 'MatchName'
    MatchAll = 'MatchAll'
    Success = 'Success'
    Jump = 'Jump'
    Split = 'Split'
    Save = 'Save'
    Bind = 'Bind'
    BindRemaining = 'BindRemaining'
    BackRef = 'BackRef'
    Merge = 'Merge'
    # def __init__(self, opcode: str = 'tail', *, ch=None, i=None, next=None, branches=None, complements=None):
    #     self.opcode = opcode
    #     if ch is not None:
    #         self.ch = ch
    #     if i is not None:
    #         self.i = i
    #     if next is not None:
    #         self.next = next
    #     if branches is not None:
    #         self.branches = branches
    #     if complements is not None:
    #         self.complements = complements

    def match(self, matcher: Matcher, binding: str = None, default: Record = None):
        self.opcode = Inst.Match
        self.matcher = matcher
        self.binding = binding
        self.default = default
        return self

    def match_name(self, name: str, matcher: Matcher, next=None):
        self.opcode = Inst.MatchName
        self.name = name
        self.matcher = matcher
        self.next = next
        return self

    def match_all(self, *matchers: Matcher, binding: str = None, default: Record = None):
        self.opcode = Inst.MatchAll
        self.matchers = matchers
        self.binding = binding
        self.default = default
        return self

    def success(self):
        self.opcode = Inst.Success
        return self

    def jump(self, next=None):
        self.opcode = Inst.Jump
        self.next = next
        return self

    def split(self, next, *branches):
        self.opcode = Inst.Split
        self.next = next
        self.branches = branches
        return self

    def save(self, name: str, next=None):
        self.opcode = Inst.Save
        self.name = name
        self.next = next
        return self

    def bind(self, name: str, default):
        self.opcode = Inst.Bind
        self.name = name
        self.default = default
        return self

    def bind_remaining(self, name: str):
        self.opcode = Inst.BindRemaining
        self.name = name
        return self

    def back_ref(self, i: int, next=None):
        self.opcode = Inst.BackRef
        self.i = i
        self.next = next
        return self

    def merge(self, step, count=2, next=None):
        self.step = step
        self.count = count
        self.next = next
        return self

    def __iter__(self):
        node = self
        seen = {self}

        while True:
            yield node
            node = self.next
            if node in seen:
                raise ValueError("Cycle found.")
            seen.add(node)

    def __str__(self):
        if self.opcode == 'tail':
            return f"tail ({hex(hash(self) % 65536)[-4:]})"
        res = self.str_node().strip()
        return res + self.str_branches()

    def __repr__(self):
        match self.opcode:
            case Inst.Match:
                props = self.matcher, self.binding, self.default
            case Inst.Jump:
                props = self.next,
            case Inst.Split:
                props = self.next, *self.branches
            case Inst.Save:
                props = self.name, self.i
            case Inst.Bind:
                props = self.name, self.default
            case _:
                props = ()
        props = (str(el) for el in props if el is not None)
        return f"{self.opcode} {' '.join(props)}"

    def tree_repr(self):
        return '\n'.join(self.treer(1, 0, {}, []))

    def str_node(self):
        args = (f" {el}" for el in (self.matcher, self.name) if el is not None)
        return f"{self.opcode}{''.join(args)}"

    def str_branches(self, seen=None, max_depth=10):
        if max_depth == 0:
            return ' > ...'
        if seen is None:
            seen = set()
        if self in seen:
            return f' > (cycle {10-max_depth})'
        seen.add(self)
        try:
            if self.next:
                # if self.next in seen:
                #     return f' > (cycle {10 - max_depth})'
                # seen.add(self.next)
                return ' > ' + self.next.str_node() + ' > ' + self.next.str_branches(seen, max_depth-1)
            elif self.branches:
                return ' >> ' + ' / '.join(branch.str_node() + ' > ' + branch.str_branches(seen, max_depth-1)
                                           for branch in self.branches)
            else:
                return ''
        except:
            return ' > ???'

    def treer(self, idx: int, depth: int, seen: dict, acc: list, branch_sym=''):
        tab = ' ' * (depth-len(branch_sym)) + branch_sym + f"{idx}. "
        if self in seen:
            acc.append(f'{tab}(Cycle: {seen[self]}. {self.opcode})')
        else:
            seen[self] = idx
            acc.append(f"{tab}{self.str_node()}")
            if self.next:
                self.next.treer(idx + 1, depth + 4*(self.opcode == Inst.Match), seen, acc)
            elif self.branches:
                for node in self.branches:
                    node.treer(idx + 1, depth+2, seen, acc, '> ')
        return acc


class VM:
    head: Inst
    tail: Inst
    value: Record | None = None
    def __new__(cls, val: Record | Matcher | Inst = None, tail: Inst = None):
        match val:
            case VM():
                return val
            # case Inst():
            #     vm = super().__new__(cls)
            #     vm.head = val
            # case Inst(), Inst():
            #     vm = super().__new__(cls)
            #     vm.head = val
        return super().__new__(cls)

    def __init__(self, val: Record | Matcher | Inst = None, tail: Inst = None):
        self.tail = Inst() if tail is None else tail
        match val:
            case Matcher() as matcher:
                self.head = Inst().match(matcher, self.tail)
            case Table() as tab:
                self.head = Inst().match(TableMatcher(tab), self.tail)
            case Trait() as trt:
                self.head = Inst().match(TraitMatcher(trt), self.tail)
            case Record() as val:
                self.head = Inst().match(ValueMatcher(val), self.tail)
                self.value = val
            # case ValueMatcher(value=self.value) as matcher:
            #     self.head = Inst().match(matcher, self.tail)
            #     self.value = self.value
            case Inst() as head:
                self.head = head
                if tail is None:
                    raise ValueError(f"Line {Context.line}: "
                                     f"when passing an Inst to a VM, you must pass the tail along with it.")
            case None:
                self.head = self.tail
            case _:
                raise TypeErr(f"Line {Context.line}: Could not patternize {val}")
        self.tail.success()

    def print(self):
        VM.printr(self.head, 1, 0, {})

    @staticmethod
    def printr(head, idx: int, depth: int, seen: dict):
        tab = '\t'*depth + f"{idx}. "
        if head in seen:
            print(f'{tab}(Cycle: {seen[head]}. {head.opcode})')
            return
        seen[head] = idx
        print(f"{tab}{head.str_node()}")
        if head.next:
            VM.printr(head.next, idx+1, depth+(head.opcode == Inst.Match), seen)
        elif head.branches:
            for node in head.branches:
                VM.printr(node, idx+1, depth+1, seen)

    def __iadd__(self, other):
        self.tail.jump(other.head)
        return self

    def run(self, args: list[Record], kwargs: dict):
        return virtualmachine(self.head, args, kwargs)


class Thread:
    id: int
    step: int
    bindings: frozendict[str, Record]
    saved: frozendict[str, int | slice]
    def __init__(self, step: int, saved: frozendict[str, int | slice], bindings: frozendict[str, Record], id: int):
        self.id = id
        self.step = step
        self.saved = saved
        self.bindings = bindings

    def save(self, name: str, idx: int):
        item = self.saved.get(name, None)
        if item is None:
            self.saved += {name: idx}
        else:
            self.saved += {name: slice(item, idx)}

    def is_parent(self, other):
        return self.id % other.id == 0

    def is_child(self, other):
        return other.id % self.id == 0

    # def copy_saves(self):
    #     return frozendict((n, s.copy()) for (n, s) in self.saved.items())

    def __repr__(self):
        return f"Thread{self.step, str(self.bindings), str(self.saved)}"

    def __hash__(self):
        """
        bindings is included in the hash function because each thread needs to keep track of which named args were used.
        saved is not needed because past saves do not affect future execution, and therefore this thread will take
        priority over any threads that come after it.
        """
        return hash((self.step, self.bindings))

    def __eq__(self, other):
        return self.step == other.step and self.bindings.keys() == other.bindings.keys()


class ThreadStack(list):
    seen: set[Thread]
    def __init__(self, *initial_threads: Thread):
        super().__init__(initial_threads)
        self.seen = set(initial_threads)

    # def push(self, step: Inst, saved: dict[str, list[int, int]], bindings: dict[str, Record]):
    #     if step not in self.seen:
    #         self.append(Thread(step, saved, bindings))
    #         self.seen.add(step)
    #     else:
    #         pass

    def push(self, thread: Thread, jump: int = 1):
        thread.step += jump
        if thread not in self.seen:
            self.append(thread)
            self.seen.add(thread)

    def new_thread(self, parent: Thread, jump: int):
        thread_id = parent.id * prime(len(self.seen))
        self.push(Thread(parent.step, parent.saved, parent.bindings, thread_id), jump)


def OLDER_virtualmachine(prog: Inst, args: list[Record], kwargs: dict[str, Record]):
    saved = {}
    bindings = {}
    current = ThreadStack(Thread(prog, saved, bindings))
    next = ThreadStack()
    matched = 0

    def outer_loop():
        yield from enumerate(args)
        yield len(args), None

    for arg_idx, arg in outer_loop():
        while current:
            thread: Thread = current.pop()
            step: Inst = thread.step
            match step.opcode:
                case Inst.Match:
                    if arg is not None and step.matcher.match_score(arg):
                        if step.binding:
                            thread.bindings = thread.bindings.copy()
                            thread.bindings[step.binding] = arg
                        next.push(step.next, thread.saved, thread.bindings)
                case Inst.MatchAll:
                    if arg is not None and all(patt.match_score(arg) for patt in step.matchers):
                        if step.binding:
                            thread.bindings = thread.bindings.copy()
                            thread.bindings[step.binding] = arg
                        next.push(step.next, thread.saved, thread.bindings)
                case Inst.MatchName:
                    if step.name in kwargs:
                        if step.matcher.match_score(kwargs[step.name]):
                            thread.bindings = thread.bindings.copy()
                            thread.bindings[step.name] = kwargs[step.name]
                            current.push(step.next, thread.saved, thread.bindings)
                        else:
                            # the name was specified, but didn't match.  Guaranteed whole pattern failure.
                            return 0, {}
                case Inst.Success:
                    if arg_idx >= len(args) - 1:
                        saved = thread.saved
                        bindings = thread.bindings
                        if all(name in bindings or name in saved for name in kwargs):
                            matched = 1
                            break
                case Inst.Jump:
                    current.push(step.next, thread.saved, thread.bindings)
                case Inst.Split:
                    for branch in reversed(step.branches):
                        current.push(branch, thread.saved, thread.bindings)
                case Inst.Save:
                    thread.saved = thread.copy_saves()
                    thread.save(step.name, arg_idx)
                    current.push(step.next, thread.saved, thread.bindings)
                # case Inst.Save:
                #     thread.saved = thread.saved
                #     thread.saved[step.i] = arg_idx
                #     current.push(step.next, thread.saved, thread.bindings)
                case Inst.Bind:
                    thread.bindings = thread.bindings.copy()
                    thread.bindings[step.name] = args[arg_idx-1]
                    current.push(step.next, thread.saved, thread.bindings)
                case Inst.Merge:
                    pass
                # case Inst.BackRef:
                #     start, stop = thread.saved[step.i], thread.saved[-step.i]
                #     matched = args[start:stop]
                #     if len(matched) == 0:
                #         current.push(step.next, thread.saved, thread.bindings)
                #     else:
                #         head = tail = Inst()
                #         for c in matched:
                #             tail = tail.Char(c, Inst()).next
                #         tail.jump(step.next)
                #         current.push(head, thread.saved, thread.bindings)
                case _:
                    raise AssertionError("unrecognized opcode: ", step.opcode)
        current, next = next, ThreadStack()

    for name, [start, end] in saved.items():
        bindings[name] = py_value(tuple(args[start:end]))

    return matched, bindings
    # submatches = {}
    # for i in saved:
    #     if i > 0 and i in saved:
    #         start, end = saved[i], saved[-i]
    #         submatches[start] = input[start:end]
    #
    # return matched, submatches


PRIMES: list[int] = [2, 3, 5, 7, 11, 13, 17]
def prime(index: int):
    """ get the nth prime number, starting with 2 at index 0 """
    try:
        return PRIMES[index]
    except IndexError:
        while len(PRIMES) <= index:
            n = PRIMES[-1] + 2
            while 1:
                for p in PRIMES:
                    if n % p == 0:
                        n += 2
                        break
                else:
                    break
            PRIMES.append(n)
        return PRIMES[index]


def OLD_virtualmachine(prog: list[Inst], args: list[Record], kwargs: dict[str, Record]):
    saved = frozendict()
    bindings = {}
    current = ThreadStack(Thread(0, saved, saved, 1))
    next = ThreadStack()
    matched = 0

    def outer_loop():
        yield from enumerate(args)
        yield len(args), None

    for arg_idx, arg in outer_loop():
        while current:
            thread: Thread = current.pop()
            step: Inst = prog[thread.step]
            match step.opcode:
                case Inst.Match:
                    if arg is not None and step.matcher.match_score(arg):
                        if step.binding:
                            thread.bindings += {step.binding: arg}
                        next.push(thread)
                case Inst.MatchAll:
                    if arg is not None and all(patt.match_score(arg) for patt in step.matchers):
                        if step.binding:
                            thread.bindings += {step.binding: arg}
                        next.push(thread)
                case Inst.MatchName:
                    if step.name in kwargs:
                        if step.matcher.match_score(kwargs[step.name]):
                            thread.bindings += {step.name: kwargs[step.name]}
                            current.push(thread)
                        else:
                            # the name was specified, but didn't match.  Guaranteed whole pattern failure.
                            return 0, {}
                case Inst.Success:
                    if arg_idx >= len(args) - 1:
                        saved = thread.saved
                        bindings = dict(thread.bindings)
                        if all(name in bindings or name in saved for name in kwargs):
                            matched = 1
                            break
                case Inst.Jump:
                    current.push(thread)
                case Inst.Split:
                    for branch in reversed(step.branches):
                        # current.push(Thread(thread.step, thread.saved, thread.bindings, step.id * prime(current_thread_count)),
                        #              branch)
                        current.new_thread(thread, branch)
                case Inst.Save:
                    thread.save(step.name, arg_idx)
                    current.push(thread)
                # case Inst.Save:
                #     thread.saved = thread.saved
                #     thread.saved[step.i] = arg_idx
                #     current.push(step.next, thread.saved, thread.bindings)
                case Inst.Bind:
                    thread.bindings = thread.bindings.copy()
                    thread.bindings[step.name] = args[arg_idx-1]
                    current.push(step.next, thread.saved, thread.bindings)
                case Inst.Merge:
                    for slice in reversed(step.slices):
                        matched, bindings = virtualmachine(prog[slice], args, kwargs)
                # case Inst.BackRef:
                #     start, stop = thread.saved[step.i], thread.saved[-step.i]
                #     matched = args[start:stop]
                #     if len(matched) == 0:
                #         current.push(step.next, thread.saved, thread.bindings)
                #     else:
                #         head = tail = Inst()
                #         for c in matched:
                #             tail = tail.Char(c, Inst()).next
                #         tail.jump(step.next)
                #         current.push(head, thread.saved, thread.bindings)
                case _:
                    raise AssertionError("unrecognized opcode: ", step.opcode)
        current, next = next, ThreadStack()

    for name, [start, end] in saved.items():
        bindings[name] = py_value(tuple(args[start:end]))

    return matched, bindings

from collections import deque
class ThreadList(deque):
    pass

def virtual_machine(prog: list[Inst], args: tuple, kwargs: dict, initial_bindings: dict, allow_arbitrary_kwargs=False):
    arg_idx: int
    arg: Record
    # kwargs = dict(args.named_arguments)
    # args = args.positional_arguments
    initial_bindings = frozendict(initial_bindings)

    def outer_loop():
        yield from enumerate(args)
        yield len(args), None

    def make_thread(step_idx: int, bindings=initial_bindings, saved=frozendict(), kwargs=frozendict(kwargs)):  # noqa
        # arg: Record = args[arg_idx]
        while step_idx < len(prog):
            step = prog[step_idx]
            match step.opcode:
                case Inst.Match:
                    if step.binding in kwargs:
                        # if step.binding in bindings:
                        #     print('DEBUG NOTE: what if positional param already matched by named arg?')
                        #     pass
                        if (match := step.matcher.match(kwargs[step.binding])) is not None:
                            bindings += match
                            bindings += {step.binding: kwargs[step.binding]}
                            kwargs -= step.binding
                        else:
                            # name found in kwargs, but did not match.  No possibility of any threads succeeding.
                            yield 'WHOLE PATTERN FAILURE'
                    elif arg is not None and (match := step.matcher.match(arg)) is not None:
                        bindings += match
                        if step.binding:
                            bindings += {step.binding: arg}
                        yield 'NEXT'
                    # elif step.default:
                    #     bindings += {step.binding: step.default}
                    else:
                        yield 'DEAD'
                case Inst.MatchAll:
                    raise DeprecationWarning
                    if step.binding in kwargs:
                        pass  # positional param already matched by named arg
                    elif arg is not None and all(patt.match_score(arg) for patt in step.matchers):
                        if step.binding:
                            bindings += {step.binding: arg}
                        yield 'NEXT'
                    elif step.default:
                        bindings += {step.binding: step.default}
                    else:
                        yield 'DEAD'
                # case Inst.MatchName:
                #     if step.name in kwargs:
                #         if step.matcher.match_score(kwargs[step.name]):
                #             bindings += {step.name: kwargs[step.name]}
                #         else:
                #             # the name was specified, but didn't match.  Guaranteed whole pattern failure.
                #             yield 'WHOLE PATTERN FAILURE'
                #             # return 'WHOLE PATTERN FAILURE'
                case Inst.Jump:
                    step_idx += step.next
                    continue
                case Inst.Split:
                    for branch in reversed(step.branches):
                        current.append(make_thread(step_idx + branch, bindings, saved, kwargs))
                    step_idx += step.next
                    continue
                case Inst.Save:
                    item = saved.get(step.name, None)
                    if item is None:
                        saved += {step.name: arg_idx}
                    else:
                        saved += {step.name: slice(item, arg_idx)}
                case Inst.Bind:
                    bindings += {step.name: step.default}
                # case Inst.BindRemaining:
                #     remaining = {k: v for k, v in kwargs.items() if k not in bindings}
                #     bindings += {step.name: py_value(remaining)}
                # case Inst.Merge:
                #     for slc in reversed(step.slices):
                #         matched, bindings = virtual_machine(prog[slc], args, kwargs)
                case _:
                    raise AssertionError("unrecognized opcode: ", step.opcode)
            step_idx += 1
        # prog reached the end
        if arg_idx == len(args) and (not kwargs or allow_arbitrary_kwargs):
            # and all(name in bindings or name in saved for name in kwargs):
            # SUCCESS
            bindings = dict(bindings)
            if allow_arbitrary_kwargs:
                bindings.update(kwargs)
            for name, part in saved.items():
                bindings[name] = py_value(tuple(args[part]))
            yield bindings
        yield 'DEAD'

    current = ThreadList([make_thread(0)])
    pending = ThreadList()

    for arg_idx, arg in outer_loop():
        while current:
            thread = current.pop()
            match next(thread):
                case dict() as bindings:
                    return bindings
                case 'WHOLE PATTERN FAILURE':
                    return
                case 'NEXT':
                    pending.appendleft(thread)
                case 'DEAD':
                    pass
                case other:
                    raise AssertionError(f"thread yielded unexpected {other}")
        current, pending = pending, ThreadList()
    # return 0, {}


class ParamSet(Pattern):
    parameters: tuple
    named_params: frozendict
    names_of_ordered_params: frozenset
    allow_arbitrary_kwargs: str | bool | None
    vm: list
    def __init__(self, *parameters, named_params: dict = None, kwargs: str = None):
        self.parameters = parameters
        self.named_params = frozendict(named_params or {})
        self.names_of_ordered_params = frozenset(param.binding for param in parameters
                                                 if param.binding and not param.multi)
        self.allow_arbitrary_kwargs = kwargs
        # self.vm = VM(parameters)
        super().__init__()
        self.vm = []
        for param in self.parameters:
            self.vm.extend(param.bytecode())
        # if kwargs:
        #     self.vm.append(Inst().bind_remaining(kwargs))

    def match_score(self, *values: Record) -> int | float:
        raise DeprecationWarning("Use ParamSet.match() instead.")
        return self.match_zip(values)[0]

    def prepend(self, param: Parameter):
        self.parameters = (param, *self.parameters)
        self.vm[:0] = param.bytecode()

    def issubset(self, other):
        return (isinstance(other, ParamSet)
                and all(p1.issubset(p2) for (p1, p2) in zip(self.parameters, other.parameters))
                and all(self.named_params[k].issubset(other.named_params[k])
                        for k in set(self.named_params).union(other.named_params)))

    def __len__(self):
        return len(self.parameters) + len(self.named_params)

    def __getitem__(self, item):
        return self.named_params.get(item, self.parameters[item])

    def to_tuple(self):
        if self.named_params:
            return None
        key: list[Record] = []
        for parameter in self.parameters:
            match parameter.pattern.matchers:
                case (ValueMatcher(value=value), ) if value.hashable():
                    key.append(value)
                case _:
                    return None
        return tuple(key)

    def to_args(self):
        pos_args = []
        names = {}
        for id, param in self:
            match param:
                case Parameter(quantifier='', pattern=ValueMatcher(value=val)):
                    pass
                case _:
                    return None
            # if (val := param.pattern.value) is None:
            #     return None
            if isinstance(id, int):
                pos_args.append(val)
            else:
                names[id] = val
            # match param.pattern.matchers:
            #     case (ValueMatcher(value=value), ) if value.hashable():
            #         if isinstance(id, int):
            #             pos_args.append(value)
            #         else:
            #             names[id] = value
            #     case _:
            #         return None
        return Args(*pos_args, **names)

    def __iter__(self):
        yield from enumerate(self.parameters)
        yield from self.named_params.items()

    @memoize
    def min_len(self) -> int:
        count = 0
        for param in self.parameters:
            count += not param.optional
        return count

    @memoize
    def max_len(self) -> int | float:
        for param in self.parameters:
            if param.quantifier in ("+", "*"):
                return math.inf
        return len(self.parameters)

    def match(self, args=None) -> None | dict[str, Record]:
        kwargs: dict[str, Record] = dict(args.named_arguments)
        for f in args.flags:
            kwargs[f] = BuiltIns['true']
        bindings: dict[str, Record] = {}

        # check for agreement of named parameters
        for name, param in self.named_params.items():
            param: Parameter
            if name in kwargs:
                match = param.match(kwargs[name])
                if match is None:
                    return
                bindings.update(match)
                bindings[name] = kwargs[name]
                del kwargs[name]
            elif param.default:
                bindings[name] = param.default
            elif param.required:
                return

        if not self.allow_arbitrary_kwargs:
            # check for illegal kwargs
            for name in kwargs:
                if name not in self.names_of_ordered_params and name not in bindings:
                    return

        return virtual_machine(self.vm, args.positional_arguments, kwargs, bindings, self.allow_arbitrary_kwargs)

        return self.vm.run(args.positional_arguments, args.named_arguments)
        return MatchState(self.parameters, args).match_zip()
        # state = MatchState(self.parameters, args)
        # return self.match_zip_recursive(state)

    def __lt__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters < other.parameters

    def __le__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters <= other.parameters

    def __eq__(self, other):
        return (isinstance(other, ParamSet)
                and self.parameters == other.parameters
                and self.named_params == other.named_params)

    def __hash__(self):
        return hash((self.parameters, self.named_params))

    def __gt__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters > other.parameters

    def __ge__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters >= other.parameters

    def __repr__(self):
        return f"ParamSet({', '.join(map(repr, self.parameters))}{'; ' + str(self.named_params) if self.named_params else ''})"


class VarPatt(Pattern):
    def __init__(self, name: PyValue[str]):
        self.dec_name = name.value

    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        Context.env.vars[self.dec_name] = arg
        return {}


class LocalPatt(VarPatt):
    def match(self, arg: Record) -> None | dict[BindTarget, Record]:
        Context.env.locals[self.dec_name] = arg
        return {}


# class Pattern(Record):
#     vm: VM
#     def __init__(self, vm: VM):
#         self.vm = vm
#         super().__init__(BuiltIns['Pattern'])
#
# class Matcher(Pattern):
#     guard = None
#     invert = False
#     def __init__(self, name: str = None, guard: Function | PyFunction = None, inverse=False):
#         self.name = name
#         self.guard = guard
#         self.invert = inverse
#
#     @property
#     def vm(self):
#         return VM(self)
#
#     def match_score(self, arg: Record) -> bool | float:
#         return self.basic_score(arg)
#         # score = self.basic_score(arg)
#         # if self.invert:
#         #     score = not score
#         # if score and self.guard:
#         #     result = self.guard.call(arg)
#         #     return score * BuiltIns['bool'].call(result).value
#         # return score
#
#     def basic_score(self, arg):
#         # implemented by subclasses
#         raise NotImplementedError
#
#     def issubset(self, other):
#         print('WARNING: Matcher.issubset method not implemented properly yet.')
#         return self.equivalent(other)
#
#     def equivalent(self, other):
#         return True
#         # return (other.guard is None or self.guard == other.guard) and self.invert == other.invert
#
#     # def call_guard(self, arg: Record) -> bool:
#     #     if self.guard:
#     #         result = self.guard.call(arg)
#     #         return BuiltIns['bool'].call(result).value
#     #     return True
#
#     def get_rank(self):
#         return self.rank
#         # rank = self.rank
#         # if self.invert:
#         #     rank = tuple(100 - n for n in rank)
#         # if self.guard:
#         #     rank = (rank[0], rank[1] - 1, *rank[1:])
#         # return rank
#
#     def __lt__(self, other):
#         return self.get_rank() < other.get_rank()
#
#     def __le__(self, other):
#         return self.get_rank() <= other.get_rank()
#
#     # def __eq__(self, other):
#     #     return self.get_rank() == other.get_rank()
#
# class TableMatcher(Matcher):
#     table: Table
#     rank = 5, 0
#
#     def __init__(self, table, name=None, guard=None, inverse=False):
#         assert isinstance(table, Table)
#         self.table = table
#         super().__init__(name, guard, inverse)
#
#     def basic_score(self, arg: Record) -> bool:
#         return arg.table == self.table or self.table in arg.table.traits
#
#     def issubset(self, other):
#         match other:
#             case TableMatcher(table=table):
#                 return table == self.table
#             case TraitMatcher(trait=trait):
#                 return trait in self.table.traits
#         return False
#
#     def equivalent(self, other):
#         return isinstance(other, TableMatcher) and self.table == other.table
#
#     def __repr__(self):
#         return f"TableMatcher({self.table})"
#
# class TraitMatcher(Matcher):
#     trait: Trait
#     rank = 6, 0
#
#     def __init__(self, trait):
#         self.trait = trait
#
#     def basic_score(self, arg: Record) -> bool:
#         return self.trait in arg.table.traits
#
#     def issubset(self, other):
#         return isinstance(other, TraitMatcher) and other.trait == self.trait
#
#     def equivalent(self, other):
#         return isinstance(other, TraitMatcher) and other.trait == self.trait
#
#     def __repr__(self):
#         return f"TraitMatcher({self.trait})"
#
#
# class ValueMatcher(Matcher):
#     value: Record
#     rank = 1, 0
#
#     def __init__(self, value):
#         self.value = value
#
#     def basic_score(self, arg: Record) -> bool:
#         return arg == self.value
#
#     def issubset(self, other):
#         match other:
#             case ValueMatcher(value=value):
#                 return value == self.value
#             case TableMatcher(table=table):
#                 return self.value.table == table
#             case TraitMatcher(trait=trait):
#                 return trait in self.value.table.traits
#         return False
#
#     def equivalent(self, other):
#         return isinstance(other, ValueMatcher) and other.value == self.value
#
#     def __repr__(self):
#         return f"ValueMatcher({self.value})"
#
#
# class FunctionMatcher(Matcher):
#     # pattern: ParamSet
#     # return_type: Matcher
#     def __init__(self, pattern, return_type, name=None, guard=None, inverse=False):
#         self.pattern = pattern
#         self.return_type = return_type
#         super().__init__(name, guard, inverse)
#
#     def basic_score(self, arg):
#         if not hasattr(arg, 'op_list'):
#             return False
#         arg: Function
#
#         def options():
#             yield from arg.op_list
#             yield from arg.op_map.values()
#
#         if all((option.pattern.issubset(self.pattern) and option.return_type.issubset(self.return_type)
#                 for option in options())):
#             return True
#
#     def issubset(self, other):
#         match other:
#             case FunctionMatcher(pattern=patt, return_type=ret):
#                 return self.pattern.issubset(patt) and self.return_type.issubset(ret)
#             case TraitMatcher(trait=BuiltIns.get('fn')) | TableMatcher(table=BuiltIns.get('Function')):
#                 return True
#         return False
#
#     def equivalent(self, other):
#         return (isinstance(other, FunctionMatcher)
#                 and other.pattern == self.pattern
#                 and other.return_type == self.return_type)
#
#
# class AnyMatcher(Matcher):
#     rank = 100, 0
#     def basic_score(self, arg: Record) -> True:
#         return True
#
#     def issubset(self, other):
#         return isinstance(other, AnyMatcher)
#
#     def equivalent(self, other):
#         return isinstance(other, AnyMatcher)
#
#     def __repr__(self):
#         return f"AnyMatcher()"
#
# class EmptyMatcher(Matcher):
#     rank = 3, 0
#     def basic_score(self, arg: Record) -> bool:
#         match arg:
#             case VirtTable():
#                 return False
#             case PyValue(value=str() | tuple() | frozenset() | list() | set() as v) | Table(records=v):
#                 return len(v) == 0
#             case Function(op_list=options, op_map=hashed_options):
#                 return bool(len(options) + len(hashed_options))
#             case _:
#                 return False
#
#     def issubset(self, other):
#         return isinstance(other, EmptyMatcher)
#
#     def equivalent(self, other):
#         return isinstance(other, EmptyMatcher)
#
#     def __repr__(self):
#         return f"EmptyMatcher()"
#
#
# class ExprMatcher(Matcher):
#     def __init__(self, expr):
#         self.expression = expr
#     def basic_score(self, arg):
#         print(f"Line {Context.line}: WARNING: expr pattern not fully implemented yet.")
#         return self.expression.evaluate().truthy
#
#
# class IterMatcher(Matcher):
#     parameters: tuple
#     def __init__(self, *params):
#         self.parameters = params
#
#     def basic_score(self, arg: Record):
#         return self.match_zip(arg)[0]
#
#     def match_zip(self, arg: Record):
#         try:
#             it = iter(arg)  # noqa
#         except TypeError:
#             return 0, {}
#         state = MatchState(self.parameters, list(it))
#         return state.match_zip()
#
#
# class FieldMatcher(Matcher):
#     fields: dict
#     def __init__(self, **fields):
#         self.fields = fields
#
#     def basic_score(self, arg: Record):
#         for name, param in self.fields.items():
#             prop = arg.get(name, None)
#             if prop is None:
#                 if not param.optional:
#                     return False
#                 continue
#             if not param.pattern.match_score(prop):
#                 return False
#         return True
#
#     def match_zip(self, arg: Record):
#         raise NotImplementedError
#         # state = MatchState((), Args(**dict(((name, arg.get(name)) for name in self.fields))))
#         # return state.match_zip()
#
#
# class Parameter(Pattern):
#     pattern: Pattern | None = None
#     binding: str | None
#     quantifier: str  # "+" | "*" | "?" | "!" | ""
#     count: tuple[int, int | float]
#     optional: bool
#     required: bool
#     multi: bool
#     default = None
#
#     # @property
#     # def binding(self): return self.pattern.binding
#
#     def __init__(self, pattern, binding: str = None, quantifier="", default=None):
#         self.pattern = patternize(pattern)
#         # self.name = self.pattern.binding
#         if default:
#             if isinstance(default, Option):
#                 self.default = default
#             else:
#                 self.default = Option(ParamSet(), default)
#             match quantifier:
#                 case "":
#                     quantifier = '?'
#                 case "+":
#                     quantifier = "*"
#         self.quantifier = quantifier
#
#     @property
#     def vm(self):
#         vm = self.pattern.vm
#         head, tail = vm.head, vm.tail
#
#         match self.quantifier:
#             case '?':
#                 tail.bind(self.binding, Inst())
#                 tail = tail.next
#                 head = Inst().split(head, tail)
#             case '+':
#                 head = Inst().save(self.binding, head)
#                 t = Inst().save(self.binding, Inst())
#                 tail.split(head, t)
#                 tail = t.next
#             case '*':
#                 t = Inst()
#                 h = Inst().split(head, t)
#                 tail.jump(h)
#                 head = h
#                 tail = t
#             case '??':
#                 head = Inst().split(tail, head)
#             case '+?':
#                 t = Inst()
#                 tail.split(t, head)
#                 tail = t
#             case '*?':
#                 t = Inst()
#                 h = Inst().split(t, head)
#                 tail.jump(h)
#                 head = h
#                 tail = t
#             case '':
#                 tail.bind(self.binding, Inst())
#                 tail = tail.next
#             case _:
#                 assert False
#
#         tail.success()
#         return VM(head, tail)
#
#     def issubset(self, other):
#         if not isinstance(other, Parameter):
#             raise NotImplementedError(f"Not yet implemented Parameter.issubset({other.__class__})")
#         if self.count[1] > other.count[1] or self.count[0] < other.count[0]:
#             return False
#         return self.pattern.issubset(other.pattern)
#
#     def _get_quantifier(self) -> str:
#         return self._quantifier
#     def _set_quantifier(self, quantifier: str):
#         self._quantifier = quantifier
#         match quantifier:
#             case "":
#                 self.count = (1, 1)
#             case "?":
#                 self.count = (0, 1)
#             case "+":
#                 self.count = (1, math.inf)
#             case "*":
#                 self.count = (0, math.inf)
#             case "!":
#                 self.count = (1, 1)
#                 # union matcher with `nonempty` pattern
#         self.optional = quantifier in ("?", "*")
#         self.required = quantifier in ("", "+")
#         self.multi = quantifier in ("+", "*")
#     quantifier = property(_get_quantifier, _set_quantifier)
#
#     def match_score(self, value) -> int | float: ...
#
#     def compare_quantifier(self, other):
#         return "_?+*".find(self.quantifier) - "_?+*".find(other.quantifier)
#
#     def __lt__(self, other):
#         match other:
#             case Parameter():
#                 q = self.compare_quantifier(other)
#                 return q < 0 or q == 0 and self.pattern < other.pattern
#         return NotImplemented
#
#     def __le__(self, other):
#         match other:
#             case Parameter():
#                 q = self.compare_quantifier(other)
#                 return q < 0 or q == 0 and self.pattern <= other.pattern
#         return NotImplemented
#
#     def __eq__(self, other):
#         match other:
#             case Parameter() as param:
#                 pass
#             case Matcher() | Pattern():
#                 param = Parameter(other)
#             case ParamSet(parameters=(param, ), named_params={}):
#                 pass
#             case _:
#                 return False
#         return self.quantifier == param.quantifier and self.pattern == param.pattern and self.default == param.default
#
#     def __hash__(self):
#         return hash((self.pattern, self.quantifier, self.default))
#
#     def __gt__(self, other):
#         match other:
#             case Parameter():
#                 q = self.compare_quantifier(other)
#                 return q < 0 or q == 0 and self.pattern > other.pattern
#         return NotImplemented
#
#     def __ge__(self, other):
#         match other:
#             case Parameter():
#                 q = self.compare_quantifier(other)
#                 return q < 0 or q == 0 and self.pattern >= other.pattern
#         return NotImplemented
#
#     def __repr__(self):
#         return f"Parameter({self.pattern} {self.binding or ''}{self.quantifier})"

def param_byte_code(self: Parameter):
    vm = self.pattern.vm
    head, tail = vm.head, vm.tail

    match self.quantifier:
        case '?':
            tail.bind(self.binding, Inst())
            tail = tail.next
            head = Inst().split(head, tail)
        case '+':
            head = Inst().save(self.binding, head)
            t = Inst().save(self.binding, Inst())
            tail.split(head, t)
            tail = t.next
        case '*':
            t = Inst()
            h = Inst().split(head, t)
            tail.jump(h)
            head = h
            tail = t
        case '??':
            head = Inst().split(tail, head)
        case '+?':
            t = Inst()
            tail.split(t, head)
            tail = t
        case '*?':
            t = Inst()
            h = Inst().split(t, head)
            tail.jump(h)
            head = h
            tail = t
        case '':
            tail.bind(self.binding, Inst())
            tail = tail.next
        case _:
            assert False

    tail.success()
    return VM(head, tail)


# Parameter.bytecode = param_byte_code

# def union_bytecode(self: Union):
#     tail = Inst()
#     branches = []
#     for patt in self.patterns:
#         h, t = bytecode()
#
# Union.bytecode = union_bytecode

# class ParamSet(Matcher):
#     parameters: tuple
#     named_params: frozendict
#     vm: VM
#     def __init__(self, *parameters: Parameter, **named_params):
#         self.parameters = parameters
#         self.named_params = frozendict(named_params)
#         Matcher.__init__(self)
#
#         if parameters:
#             vm = parameters[0].bytecode()
#             for sub_vm in (p.bytecode() for p in parameters[1:]):
#                 h, t = sub_vm.head, sub_vm.tail
#                 vm.tail.jump(h)
#                 vm.tail = t
#             vm.tail.success()
#             self.vm = vm
#         else:
#             self.vm = VM()
#
#     def match_score(self, *values: Record) -> int | float:
#         return self.match_zip(values)[0]
#
#     def issubset(self, other):
#         return (isinstance(other, ParamSet)
#                 and all(p1.issubset(p2) for (p1, p2) in zip(self.parameters, other.parameters))
#                 and all(self.named_params[k].issubset(other.named_params[k])
#                         for k in set(self.named_params).union(other.named_params)))
#
#     def __len__(self):
#         return len(self.parameters) + len(self.named_params)
#
#     def __getitem__(self, item):
#         return self.named_params.get(item, self.parameters[item])
#
#     def to_tuple(self):
#         if self.named_params:
#             return None
#         key: list[Record] = []
#         for parameter in self.parameters:
#             match parameter.pattern.matchers:
#                 case (ValueMatcher(value=value), ) if value.hashable():
#                     key.append(value)
#                 case _:
#                     return None
#         return tuple(key)
#
#     def to_args(self):
#         pos_args = []
#         names = {}
#         for id, param in self:
#             if (val := param.pattern.value) is None:
#                 return None
#             if isinstance(id, int):
#                 pos_args.append(val)
#             else:
#                 names[id] = val
#             # match param.pattern.matchers:
#             #     case (ValueMatcher(value=value), ) if value.hashable():
#             #         if isinstance(id, int):
#             #             pos_args.append(value)
#             #         else:
#             #             names[id] = value
#             #     case _:
#             #         return None
#         return Args(*pos_args, **names)
#
#     def __iter__(self):
#         yield from enumerate(self.parameters)
#         yield from self.named_params.items()
#
#     @memoize
#     def min_len(self) -> int:
#         count = 0
#         for param in self.parameters:
#             count += not param.optional
#         return count
#
#     @memoize
#     def max_len(self) -> int | float:
#         for param in self.parameters:
#             if param.quantifier in ("+", "*"):
#                 return math.inf
#         return len(self.parameters)
#
#     def match_zip(self, args=None) -> tuple[float | int, dict[str, Record]]:
#         if args is None:
#             return 1, {}
#         if len(args) == 0 == self.min_len():
#             return 1, {}
#         if not self.min_len() <= len(args) <= self.max_len():
#             return 0, {}
#         if isinstance(args, tuple):
#             args = Args(*args)
#
#         return self.vm.run(args.positional_arguments)
#         # return MatchState(self.parameters, args).match_zip()
#
#     @memoize
#     def min_len(self) -> int:
#         count = 0
#         for param in self.parameters:
#             count += not param.optional
#         return count
#
#     @memoize
#     def max_len(self) -> int | float:
#         for param in self.parameters:
#             if param.quantifier in ("+", "*"):
#                 return math.inf
#         return len(self.parameters)
#
#     def __lt__(self, other):
#         if not isinstance(other, ParamSet):
#             return NotImplemented
#         return self.parameters < other.parameters
#
#     def __le__(self, other):
#         if not isinstance(other, ParamSet):
#             return NotImplemented
#         return self.parameters <= other.parameters
#
#     def __eq__(self, other):
#         return (isinstance(other, ParamSet)
#                 and self.parameters == other.parameters
#                 and self.named_params == other.named_params)
#
#     def __hash__(self):
#         return hash((self.parameters, self.named_params))
#
#     def __gt__(self, other):
#         if not isinstance(other, ParamSet):
#             return NotImplemented
#         return self.parameters > other.parameters
#
#     def __ge__(self, other):
#         if not isinstance(other, ParamSet):
#             return NotImplemented
#         return self.parameters >= other.parameters
#
#     def __repr__(self):
#         return f"ParamSet({', '.join(self.parameters)}{'; ' + str(self.named_params) if self.named_params else ''})"


def patternize(val) -> Pattern | Matcher:
    match val:
        case Pattern() | Matcher():
            return val
        case Table():
            return TableMatcher(val)
        case Trait():
            return TraitMatcher(val)
        case Record():
            return ValueMatcher(val)
        case _:
            raise TypeErr(f"Line {Context.line}: Could not patternize {val}")

