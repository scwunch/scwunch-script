import re
import timeit


class Inst:
    opcode: str
    next = None  # Inst | None
    matcher: Matcher = None
    i: int = None
    branches = None
    Match = 'Char'
    Success = 'Match'
    Jump = 'Jump'
    Split = 'Split'
    Save = 'Save'
    BackRef = 'BackRef'
    def __init__(self, opcode: str = 'tail', *, ch=None, i=None, next=None, branches=None):
        self.opcode = opcode
        if ch is not None:
            self.ch = ch
        if i is not None:
            self.i = i
        if next is not None:
            self.next = next
        if branches is not None:
            self.branches = branches

    def match(self, ch: str, next=None):
        self.opcode = Inst.Match
        self.ch = ch
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

    def save(self, i: int, next=None):
        self.opcode = Inst.Save
        self.i = i
        self.next = next
        return self

    def back_ref(self, i: int, next=None):
        self.opcode = Inst.BackRef
        self.i = i
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

    def __repr__(self):
        if self.opcode == 'tail':
            return f"tail ({hex(hash(self) % 65536)[-4:]})"
        res = self.str_node().strip()
        return res + self.str_branches()

    def str_node(self):
        args = (f" {el}" for el in (self.ch, self.i) if el is not None)
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


class Thread:
    def __init__(self, step: Inst, saved: dict):
        self.step = step
        self.saved = saved

    def __repr__(self):
        return f"Thread({self.step, self.saved}"


class ThreadList:
    current: list[Thread]
    next: dict[Inst, Thread]
    progs: set[Inst]

    def __init__(self, initial_thread: Thread):
        self.current = [initial_thread]
        self.next = {}
        self.progs = {initial_thread.step}

    def add_next(self, step: Inst, saved: dict):
        if step not in self.next:
            self.next[step] = Thread(step, saved)

    def add_current(self, step: Inst, saved: dict):
        if step not in self.progs:
            self.current.append(Thread(step, saved))
            self.progs.add(step)

    def advance_next_to_current(self):
        self.current = list(reversed(self.next.values()))
        self.progs = set(self.next)
        self.next = {}



def virtualmachine(prog: Inst, args: list):
    saved = {}
    threads = ThreadList(Thread(prog, saved))
    matched = 0

    def loop(args: list):
        yield from enumerate(args)
        # since there are no more args to process, we just let the machine try whatever remaining threads there may be
        threads.add_next = threads.add_current
        yield len(args), None

    for arg_idx, arg in loop(args):
        while threads.current:
            thread = threads.current.pop()
            step: Inst = thread.step
            match step.opcode:
                case Inst.Match:
                    if arg is not None and step.matcher.match(arg):
                        threads.add_next(step.next, thread.saved)
                case Inst.Success:
                    saved = thread.saved
                    matched = 1
                    break
                case Inst.Jump:
                    threads.add_current(step.next, thread.saved)
                case Inst.Split:
                    for branch in step.branches:
                        threads.add_current(branch, thread.saved)
                case Inst.Save:
                    thread.saved = thread.saved.copy()
                    thread.saved[step.i] = arg_idx
                    threads.add_current(step.next, thread.saved)
                case Inst.BackRef:
                    start, stop = thread.saved[step.i], thread.saved[-step.i]
                    matched = args[start:stop]
                    if len(matched) == 0:
                        threads.add_current(step.next, thread.saved)
                    else:
                        head = tail = Inst()
                        for c in matched:
                            tail = tail.Char(c, Inst()).next
                        tail.jump(step.next)
                        threads.add_current(head, thread.saved)
                case _:
                    raise AssertionError("unrecognized opcode: ", step.opcode)
        threads.advance_next_to_current()
    submatches = {}
    for i in saved:
        if i > 0 and i in saved:
            start, end = saved[i], saved[-i]
            submatches[start] = input[start:end]

    return matched, submatches

def pp(d: dict):
    return ', '.join(f"{k}:{repr(v)}" for k, v in d.items())

class RegexNode:
    def __add__(self, other):
        match other:
            case Concat(exprs=exprs):
                return Concat(self, *exprs)
            case EmptyRegex():
                return other
        return Concat(self, other)

    def __or__(self, other):
        match other:
            case Alt(exprs=exprs):
                return Alt(self, *exprs)
        return Alt(self, other)

    def __repr__(self):
        return f"{self.__class__.__name__}({pp(self.__dict__)})"

    def bytecode(self) -> tuple[Inst, Inst]:
        """:return head and tail of bytcode virtual machine"""
        raise NotImplementedError

class CharNode(RegexNode):
    char: str
    def __init__(self, char: str):
        self.char = char

    def bytecode(self):
        step = Inst().Char(self.char, Inst())
        return step, step.next

class CharClass(RegexNode):
    cls: str | set[str] = ()
    classes = dict(
        s=set(' \t\n\v\r'),
        d=set('0123456789'),
        w=set('ABCDEFGHIJKLMNOPQRSTUVWXYSabcdefghijklmnopqrstuvwxys_0123456789')
    )
    def __init__(self, ch: str):
        self.char = ch
        if ch in CharClass.classes:
            self.cls = CharClass.classes[ch]

    def bytecode(self) -> tuple[Inst, Inst]:
        step = Inst().CharClass(self.char, self.cls, Inst())
        return step, step.next

class Quant(RegexNode):
    expr: RegexNode
    quantifier: str
    def __init__(self, expr: RegexNode, quantifier: str):
        self.expr = expr
        self.quantifier = quantifier
    def bytecode(self):
        head, tail = self.expr.bytecode()
        match self.quantifier:
            case '?':
                return Inst().Split(head, tail), tail
            case '+':
                t = Inst()
                tail.Split(head, t)
                return head, t
            case '*':
                t = Inst()
                h = Inst().Split(head, t)
                tail.Jump(h)
                return h, t
            case '??':
                return Inst().Split(tail, head), tail
            case '+?':
                t = Inst()
                tail.Split(t, head)
                return head, t
            case '*?':
                t = Inst()
                h = Inst().Split(t, head)
                tail.Jump(h)
                return h, t

        assert False

class Concat(RegexNode):
    exprs: list[RegexNode]
    def __init__(self, *exprs: RegexNode):
        self.exprs = list(exprs)

    def __add__(self, other):
        match other:
            case Concat(exprs=exprs):
                return Concat(*self.exprs, *exprs)
        return Concat(*self.exprs, other)

    def __iadd__(self, other):
        match other:
            case Concat(exprs=exprs):
                self.exprs.extend(exprs)
            case _:
                self.exprs.append(other)

    def __repr__(self):
        return str(self.exprs)

    def bytecode(self):
        head, tail = self.exprs[0].bytecode()
        for node in self.exprs[1:]:
            h, t = node.bytecode()
            tail.Jump(h)
            tail = t
        return head, tail


class Alt(RegexNode):
    exprs: list[RegexNode]
    def __init__(self, *exprs: RegexNode):
        self.exprs = list(exprs)

    def __or__(self, other):
        match other:
            case Alt(exprs=exprs):
                return Alt(*self.exprs, *exprs)
        return Alt(*self.exprs, other)

    def __ior__(self, other):
        match other:
            case Alt(exprs=exprs):
                self.exprs.extend(exprs)
            case _:
                self.exprs.append(other)

    def bytecode(self):
        alts = []
        tail = Inst()
        for node in self.exprs:
            h, t = node.bytecode()
            alts.append(h)
            t.Jump(tail)

        head = Inst().Split(*alts)
        return head, tail


class GroupNode(RegexNode):
    node: RegexNode
    index: int
    def __init__(self, node: RegexNode, index: int):
        self.node = node
        self.index = index

    def bytecode(self) -> tuple[Inst, Inst]:
        head, tail = Inst(), Inst()
        h, t = self.node.bytecode()
        head.Save(self.index, h)
        t.Save(-self.index, tail)
        return head, tail

class EmptyRegex(RegexNode):
    def __add__(self, other):
        return other

    def __or__(self, other):
        match other:
            case Alt(exprs=exprs):
                return Alt(self, *exprs)
        return Alt(self, other)

    def bytecode(self) -> tuple[Inst, Inst]:
        tail = Inst()
        return Inst().Jump(tail), tail

class BackRef(RegexNode):
    index: int
    def __init__(self, index: int):
        self.index = index

    def bytecode(self) -> tuple[Inst, Inst]:
        tail = Inst()
        return Inst().BackRef(self.index, tail), tail

class TokenState:
    i: int  # string index
    grp_idx: int  # group index
    def __init__(self, i=0, grp_idx=1):
        self.i = i
        self.grp_idx = grp_idx
def tokenize(regex: str, state: TokenState = None):
    nodes: list[RegexNode] = []
    if state is None:
        state = TokenState()

    while state.i < len(regex):
        match regex[state.i]:
            case '?' | '+' | '*' as quantifier:
                modifier = regex[state.i+1:state.i+2]
                modifier *= modifier in '?+'
                nodes[-1] = Quant(nodes[-1], quantifier + modifier)
                state.i += len(modifier)
            case '.':
                nodes.append(CharClass('.'))
            case '|':
                state.i += 1
                nodes = [nodify(nodes) | tokenize(regex, state)]
                if state.i >= len(regex) or regex[state.i] == ')':
                    break
            case '(':
                state.i += 1
                nodes.append(GroupNode(tokenize(regex, state), state.grp_idx))
                if state.i >= len(regex):
                    pass
                assert regex[state.i] == ')'
            case '\\':
                state.i += 1
                assert state.i < len(regex)
                c = regex[state.i]
                match c:
                    case 'n' | 'r' | 't' | '\\':
                        nodes.append(CharNode('\\' + c))
                    case 'd' | 's' | 'w' | 'D' | 'S' | 'W':
                        nodes.append(CharClass(c))
                    case '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9':
                        nodes.append(BackRef(int(c)))
                    case _:
                        raise ValueError('Unrecognized escape character: ' + c)
            case ')':
                break
            case c:
                nodes.append(CharNode(c))
        state.i += 1
    return nodify(nodes)


def nodify(nodes: list[RegexNode]) -> RegexNode:
    match len(nodes):
        case 0:
            return EmptyRegex()
        case 1:
            return nodes[0]
        case _:
            return Concat(*nodes)


def make_vm(regex: str, offset=0) -> Inst:
    node = tokenize(regex)
    head, tail = node.bytecode()
    tail.Match()
    return head

def match(patt: str, text: str):
    program = make_vm(patt)
    return virtualmachine(program, text)


test_patterns = ('a',
                 'b',
                 'ab',
                 'a+b+',
                 'b?a*b+',
                 'a|b',
                 'a|b|c',
                 'b|c',
                 'ab|c',
                 '(a+b+)',
                 '(a+)b',
                 '(a|b)c',
                 '(a*)',
                 '.*',
                 '(.*)',
                 r'(a)\1',
                 r'(a|b)\1',
                 r'(\d+.*)*(.)',
                 r'(a+)b',
                 r'(a+?b)',
                 )

texts = ('ab',
         'aabb',
         'b',
         'c',
         'aaaab',
         'ac',
         'bc',
         'bbbb',
         '123hg78787.8484',
         'abaabaaab',
         )

if 1:
    for regex in test_patterns:
        print('Pattern: '+regex)
        program = make_vm(regex)
        repr(program)
        for text in texts:
            res = virtualmachine(program, text)
            rematch = re.match(regex, text)
            agree = bool(res[0] if rematch else not res[0])
            if not agree:
                pass  # print("MISMATCH FOUND...............................................................................")
            print(f"\t{text}:{' ' * (10 - len(text))}{res} :: {'agree' if agree else 'MISMATCH'}")
        print()

if 1:
    repeats = 1
    for n in range(1):
        patt = 'a*' * n + 'a' * n
        text = 'a' * n

        # patt = '.*a.*b.*c.*d.*e.*f.*g.*h.*i.*j.*k.*l.*m.*n.*o.*p.*q.*r.*s.*t.*u.*v.*w.*x.*y.*z.*'[:3*n]
        # text = 'abcdefghijklmnopqrstuvwxys'
        # text = ''.join(c*n for c in text)

        patt = '.*,' * n + 'x' + '.*,' * n
        text = ('a'*n + ',') * n + 'x' + ('a'*n + ',') * n

        patt = r'(.*,)*(x)'

        patt = r'(x+)+y'
        text = 'x' * n

        patt = '(.*?,)'*n + 'P'
        text = (str(n)+',')*(n+1) + 'P'

        patt = r'<html.*?>.*?<head.*?>.*?<title>.*?</title>.*?</head>.*?<body.*?>(.*?)</body>.*?</html>'
        text = r"""
        <copy paste html here>
        """.strip().replace('\n', '')

        def foo():
            return print(re.match(patt, text))

        def bar():
            return match(patt, text)

        py_time = timeit.timeit(foo, number=repeats) / repeats
        pili_time = timeit.timeit(bar, number=repeats) / repeats
        print(f"n={n}: {py_time} {pili_time}")
        print(bar())
        # print(pili_time)
        # print(f"n={n}: {timeit.timeit(bar, number=repeats) / repeats}")
