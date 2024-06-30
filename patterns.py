import math
from tables import *

print(f"loading module: {__name__} ...")

class Pattern(Record):
    def __init__(self):
        super().__init__(BuiltIns['Pattern'])
    def bytecode(self):
        raise NotImplementedError(self.__class__.__name__)


class Matcher(Pattern):
    guard = None
    invert = False
    def __init__(self, name: str = None, guard: Function | PyFunction = None, inverse=False):
        if name is not None or guard is not None or inverse:
            raise Exception("Check this out.  Can we get rid of these properties entirely?")
        self.name = name
        self.guard = guard
        self.invert = inverse
        super().__init__()

    def match_score(self, arg: Record) -> bool | float:
        return self.basic_score(arg)
        # score = self.basic_score(arg)
        # if self.invert:
        #     score = not score
        # if score and self.guard:
        #     result = self.guard.call(arg)
        #     return score * BuiltIns['bool'].call(result).value
        # return score

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
            case Matcher():
                return self.get_rank() <= other.get_rank()
            # case Parameter(pattern=pattern):
            #     return self <= pattern
            case Intersection(patterns=patterns):
                return all(self < p for p in patterns)
            case Union(patterns=patterns):
                return any(self <= p for p in patterns)
            case _:
                return NotImplemented

    # def __eq__(self, other):
    #     return self.get_rank() == other.get_rank()

class TableMatcher(Matcher):
    _table: Table
    rank = 5, 0

    def __init__(self, table, name=None, guard=None, inverse=False):
        assert isinstance(table, Table)
        self._table = table
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool:
        return arg.table == self._table

    def issubset(self, other):
        match other:
            case TableMatcher(_table=table):
                return table == self._table
            case TraitMatcher(trait=trait):
                return trait in self._table.traits
        return False

    def equivalent(self, other):
        return isinstance(other, TableMatcher) and self._table == other._table

    def __repr__(self):
        return f"TableMatcher({self._table})"

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

    def __repr__(self):
        return f"TraitMatcher({self.trait})"


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

    def __repr__(self):
        return f"ValueMatcher({self.value})"


class FunctionMatcher(Matcher):
    # signature: ArgsMatcher
    # return_type: Matcher
    def __init__(self, signature, return_type, name=None, guard=None, inverse=False):
        self.signature = signature
        self.return_type = return_type
        super().__init__(name, guard, inverse)

    def basic_score(self, arg):
        if not hasattr(arg, 'op_list'):
            return False
        arg: Function

        def options():
            yield from arg.op_list
            yield from arg.op_map.values()

        if all((option.pattern.issubset(self.signature) and option.return_type.issubset(self.return_type)
                for option in options())):
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


class AnyMatcher(Matcher):
    rank = 100, 0
    def basic_score(self, arg: Record) -> True:
        return True

    def issubset(self, other):
        return isinstance(other, AnyMatcher)

    def equivalent(self, other):
        return isinstance(other, AnyMatcher)

    def __repr__(self):
        return f"AnyMatcher()"

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
            case _:
                return False

    def issubset(self, other):
        return isinstance(other, EmptyMatcher)

    def equivalent(self, other):
        return isinstance(other, EmptyMatcher)

    def __repr__(self):
        return f"EmptyMatcher()"


class IterMatcher(Matcher):
    parameters: tuple
    def __init__(self, *params):
        self.parameters = params

    def basic_score(self, arg: Record):
        return self.match_zip(arg)[0]

    def match_zip(self, arg: Record):
        try:
            it = iter(arg)  # noqa
        except TypeError:
            return 0, {}
        state = MatchState(self.parameters, list(it))
        return state.match_zip()


class FieldMatcher(Matcher):
    fields: dict

    def __init__(self, **fields):
        self.fields = fields

    def basic_score(self, arg: Record):
        for name, param in self.fields.items():
            prop = arg.get(name, None)
            if prop is None:
                if not param.optional:
                    return False
                continue
            if not param.pattern.match_score(prop):
                return False
        return True

    def match_zip(self, arg: Record):
        raise NotImplementedError
        # state = MatchState((), Args(**dict(((name, arg.get(name)) for name in self.fields))))
        # return state.match_zip()

    def __repr__(self):
        return f"FieldMatcher{self.fields}"

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

class Intersection(Pattern):
    # I'm confused.  I think I made this inherit from "Pattern" rather than "Matcher" so that you could do intersections of multiple parameters in a row
    # eg foo[(num+) & (int*, ratio*)]: ...
    # but somehow it's getting compared with matchers now.
    patterns: tuple[Pattern, ...]
    def __init__(self, *patterns: Pattern, binding=None):
        if binding is not None:
            raise Exception("This should be a parameter, not an Intersection.")
        self.patterns = patterns
        super().__init__()

    def get_rank(self):
        return "Why is this being called?"

    @property
    def matchers(self) -> tuple[Matcher, ...]:
        assert all(isinstance(patt, Matcher) for patt in self.patterns)
        return self.patterns

    def match_score(self, arg: Record):
        return all(m.match_score(arg) for m in self.matchers)

    def issubset(self, other):
        match other:
            case Matcher() as other_matcher:
                return any(m.issubset(other_matcher) for m in self.matchers)
            case Intersection() as patt:
                return any(matcher.issubset(patt) for matcher in self.matchers)
            case Union(patterns=patterns):
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
        # if any(isinstance(patt, Parameter) and patt.multi or isinstance(patt, ArgsMatcher)
        #        for patt in self.patterns):
        #     raise NotImplementedError("I don't yet know how to do intersection of multi-parameters")

    def __lt__(self, other):
        match other:
            case Intersection(matchers=other_matchers):
                return (len(self.matchers) > len(other_matchers)
                        or len(self.matchers) == len(other_matchers) and self.matchers < other_matchers)
            case Union(patterns=patterns):
                return any(self <= p for p in patterns)
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case Intersection(matchers=other_matchers):
                return (len(self.matchers) > len(other_matchers)
                        or len(self.matchers) == len(other_matchers) and self.matchers <= other_matchers)
            case Union(patterns=patterns):
                return any(self <= p for p in patterns)
            case _:
                raise NotImplementedError

    def __hash__(self):
        return hash(frozenset(self.patterns))

    def __eq__(self, other):
        match other:
            case Intersection(matchers=matchers):
                return matchers == self.matchers
            case Union(patterns=(Pattern() as patt, )):
                return self == patt
            case Matcher() as m:
                return len(self.matchers) == 1 and self.matchers[0] == m
        return False

    def __repr__(self):
        return f"Intersection{self.patterns}"


class Union(Pattern):
    patterns: tuple[Pattern, ...]
    def __init__(self, *patterns, binding=None):
        self.patterns = patterns
        if binding is not None:
            raise Exception("Shoulda been a Parameter!")
        super().__init__()

    def match_score(self, arg: Record):
        return any(p.match_score(arg) for p in self.patterns)

    def issubset(self, other):
        return all(p.issubset(other) for p in self.patterns)

    def bytecode(self):
        machine = VM()
        machine.tail = Inst().success()
        heads = []
        for vm in (patt.bytecode() for patt in self.patterns):
            vm.tail.jump(machine.tail)
            heads.append(vm.head)
        machine.head = Inst().split(*heads)
        return machine

    def __lt__(self, other):
        match other:
            case Intersection():
                return all(p < other for p in self.patterns)
            case Union(patterns=patterns):
                return self.patterns < patterns
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case Union(patterns=patterns):
                return self.patterns <= patterns
            case Matcher() | Intersection():
                return all(p < other for p in self.patterns)
            case _:
                raise NotImplementedError

    def __eq__(self, other):
        match self.patterns:
            case ():
                return isinstance(other, Union) and other.patterns == ()
            case (Pattern() as patt, ):
                return patt == other
        return isinstance(other, Union) and self.patterns == other.patterns

    def __hash__(self):
        return hash(frozenset(self.patterns))

    def __repr__(self):
        return f"Union{self.patterns}"


class Parameter(Pattern):
    pattern: Pattern | None = None
    binding: str = None  # property
    quantifier: str  # "+" | "*" | "?" | "!" | ""
    count: tuple[int, int | float]
    optional: bool
    required: bool
    multi: bool
    default = None

    # @property
    # def binding(self): return self.pattern.binding

    def __init__(self, pattern, binding: str = None, quantifier="", default=None):
        self.pattern = patternize(pattern)
        self.binding = binding
        # self.name = self.pattern.binding
        if default:
            if isinstance(default, Option):
                self.default = default
            else:
                self.default = Option(ArgsMatcher(), default)
            match quantifier:
                case "":
                    quantifier = '?'
                case "+":
                    quantifier = "*"
        self.quantifier = quantifier
        super().__init__()

    def issubset(self, other):
        if not isinstance(other, Parameter):
            raise NotImplementedError(f"Not yet implemented Parameter.issubset({other.__class__})")
        if self.count[1] > other.count[1] or self.count[0] < other.count[0]:
            return False
        return self.pattern.issubset(other.pattern)

    def _get_quantifier(self) -> str:
        return self._quantifier
    def _set_quantifier(self, quantifier: str):
        self._quantifier = quantifier
        match quantifier:
            case "":
                self.count = (1, 1)
            case "?":
                self.count = (0, 1)
            case "+":
                self.count = (1, math.inf)
            case "*":
                self.count = (0, math.inf)
            case "!":
                self.count = (1, 1)
                # union matcher with `nonempty` pattern
        self.optional = quantifier in ("?", "*")
        self.required = quantifier in ("", "+")
        self.multi = quantifier in ("+", "*")
    quantifier = property(_get_quantifier, _set_quantifier)

    def match_score(self, value) -> int | float: ...

    def compare_quantifier(self, other):
        return "_?+*".find(self.quantifier) - "_?+*".find(other.quantifier)

    def bytecode(self):
        vm = self.pattern.bytecode()
        head, tail = vm.head, vm.tail

        match self.quantifier:
            case '?':
                if isinstance(self.pattern, Matcher):
                    bind

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

    def __lt__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern < other.pattern
        return NotImplemented

    def __le__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern <= other.pattern
        return NotImplemented

    def __eq__(self, other):
        match other:
            case Parameter() as param:
                pass
            case Matcher() | Pattern():
                param = Parameter(other)
            case ArgsMatcher(parameters=(param, ), named_params={}):
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
        return NotImplemented

    def __ge__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern >= other.pattern
        return NotImplemented

    def __repr__(self):
        return f"Parameter({self.pattern}{' ' + self.binding if self.binding else ''}{self.quantifier})"

class Inst:
    opcode: str = 'tail'
    next = None  # Inst | None
    matcher: Matcher = None
    matchers: tuple[Matcher, ...] = ()
    i: int = None
    name: str = None
    branches = None
    Match = 'Match'
    MatchName = 'MatchName'
    MatchAll = 'MatchAll'
    Success = 'Success'
    Jump = 'Jump'
    Split = 'Split'
    Save = 'Save'
    Bind = 'Bind'
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

    def match(self, matcher: Matcher, next=None):
        self.opcode = Inst.Match
        self.matcher = matcher
        self.next = next
        return self

    def match_name(self, name: str, matcher: Matcher, next=None):
        self.name = name
        self.matcher = matcher
        self.next = next
        return self

    def match_all(self, *matchers: Matcher, next=None):
        self.opcode = Inst.MatchAll
        self.matchers = matchers
        self.next = next
        return self

    def success(self):
        self.opcode = Inst.Success
        return self

    def jump(self, next=None):
        self.opcode = Inst.Jump
        self.next = next
        return self

    def split(self, *branches):
        self.opcode = Inst.Split
        self.branches = branches
        return self

    def save(self, name: str, next=None):
        self.opcode = Inst.Save
        self.name = name
        self.next = next
        return self

    def bind(self, name: str, next=None):
        self.opcode = Inst.Bind
        self.name = name
        self.next = next
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


class HashableDict(dict):
    def __hash__(self):
        try:
            self.hash = hash(frozenset(self.items()))
        except TypeError:
            self.hash = hash(frozenset(self))
        return self.hash


class Thread:
    step: Inst
    bindings: dict[str, Record]
    saved: dict[str, list[int, int]]
    def __init__(self, step: Inst, saved: dict[str, list[int, int]], bindings: dict[str, Record]):
        self.step = step
        self.saved = saved
        self.bindings = bindings

    def save(self, name: str, idx: int):
        if name in self.saved:
            self.saved[name][1] = idx
        else:
            self.saved[name] = [idx, idx]

    def copy_saves(self):
        return dict((n, s.copy()) for (n, s) in self.saved.items())

    def __repr__(self):
        return f"Thread({self.step, self.saved}"

    def __hash__(self):
        return hash((self.step, self.bindings))


class ThreadStack(list):
    seen: set[Inst]
    def __init__(self, *initial_threads: Thread):
        super().__init__(initial_threads)
        self.seen = {t.step for t in initial_threads}

    def push(self, step: Inst, saved: dict[str, list[int, int]], bindings: dict[str, Record]):
        if step not in self.seen:
            self.append(Thread(step, saved, bindings))
            self.seen.add(step)
        else:
            pass


def virtualmachine(prog: Inst, args: list[Record], kwargs: dict[str, Record]):
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


class ArgsMatcher(Matcher):
    parameters: tuple
    vm: VM
    named_params: frozendict
    def __init__(self, *parameters, named_params: dict = None):
        self.parameters = parameters
        self.named_params = frozendict(named_params or {})
        # self.vm = VM(parameters)
        super().__init__()
        if named_params:
        if parameters:
            vm = parameters[0].bytecode()
            for sub_vm in (p.bytecode() for p in parameters[1:]):
                h, t = sub_vm.head, sub_vm.tail
                vm.tail.jump(h)
                vm.tail = t
            vm.tail.success()
            self.vm = vm
        else:
            self.vm = VM()

    def match_score(self, *values: Record) -> int | float:
        return self.match_zip(values)[0]

    def issubset(self, other):
        return (isinstance(other, ArgsMatcher)
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

    def match_zip(self, args=None) -> tuple[float | int, dict[str, Record]]:
        # if args is None:
        #     return 1, {}
        if len(args) == 0 == self.min_len():
            return 1, {}
        if not self.min_len() <= len(args) <= self.max_len():
            return 0, {}
        if isinstance(args, tuple):
            args = Args(*args)
        if args.flags:
            raise NotImplementedError("My VM can't yet handle flags... sorry.")
        return self.vm.run(args.positional_arguments, args.named_arguments)
        return MatchState(self.parameters, args).match_zip()
        # state = MatchState(self.parameters, args)
        # return self.match_zip_recursive(state)

    def __lt__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters < other.parameters

    def __le__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters <= other.parameters

    def __eq__(self, other):
        return (isinstance(other, ArgsMatcher)
                and self.parameters == other.parameters
                and self.named_params == other.named_params)

    def __hash__(self):
        return hash((self.parameters, self.named_params))

    def __gt__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters > other.parameters

    def __ge__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters >= other.parameters

    def __repr__(self):
        return f"ArgsMatcher({', '.join(map(repr, self.parameters))}{'; ' + str(self.named_params) if self.named_params else ''})"

#
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
#     # pattern: ArgsMatcher
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
#                 self.default = Option(ArgsMatcher(), default)
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
#             case ArgsMatcher(parameters=(param, ), named_params={}):
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

# class ArgsMatcher(Matcher):
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
#         return (isinstance(other, ArgsMatcher)
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
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters < other.parameters
#
#     def __le__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters <= other.parameters
#
#     def __eq__(self, other):
#         return (isinstance(other, ArgsMatcher)
#                 and self.parameters == other.parameters
#                 and self.named_params == other.named_params)
#
#     def __hash__(self):
#         return hash((self.parameters, self.named_params))
#
#     def __gt__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters > other.parameters
#
#     def __ge__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters >= other.parameters
#
#     def __repr__(self):
#         return f"ArgsMatcher({', '.join(self.parameters)}{'; ' + str(self.named_params) if self.named_params else ''})"


def patternize(val):
    match val:
        case Pattern():
            return val
        case Table():
            return TableMatcher(val)
        case Trait():
            return TraitMatcher(val)
        case Record():
            return ValueMatcher(val)
        case _:
            raise TypeErr(f"Line {Context.line}: Could not patternize {val}")

