from fractions import Fraction
from Syntax import *
from DataStructures import *
from Env import *


def expressionize(nodes: list[Node] | Statement):
    if isinstance(nodes, Statement):
        line = nodes.pos[0]
        src = nodes.source_text
        nodes = nodes.nodes
    else:
        line = None
        src = " ".join(n.source_text for n in nodes)
    if not nodes:
        return EmptyExpr()
    if nodes[0].type == TokenType.Command:
        return Command(nodes[0].source_text, nodes[1:], line, src)
    if len(nodes) == 1:
        return SingleNode(nodes[0], line, src)
    match nodes[0].source_text:
        case 'if':
            return Conditional(nodes, line, src)
        case 'for':
            return ForLoop(nodes, line, src)
        case 'while':
            return WhileLoop(nodes, line, src)

    return Mathological(nodes, line, src)

class Expression:
    line: int = None
    nodes: list[Node] = None
    source: str = ""
    def __init__(self, line: int | None, source: str):
        self.line = line
        self.source = source

    def evaluate(self):
        raise NotImplemented

    def __len__(self):
        return len(self.nodes) if self.nodes else 0

    def __repr__(self):
        if self.line:
            return f"Line {self.line}: {self.source}"
        return self.source

class Mathological(Expression):
    op_idx: int
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)
        self.nodes = nodes
        op_idx = None
        min_precedence = math.inf
        if nodes[0].type == TokenType.Operator:
            pre_op = Op[nodes[0].source_text]
            if pre_op.prefix:
                min_precedence = pre_op.prefix
                op_idx = 0
        if nodes[-1].type == TokenType.Operator:
            post_op = Op[nodes[-1].source_text]
            if post_op.postfix and post_op.postfix < min_precedence:
                min_precedence = post_op.postfix
                op_idx = len(nodes) - 1

        def get_op(idx: int) -> Operator | None:
            if nodes[idx].type != TokenType.Operator:
                return None
            return Op[nodes[idx].source_text]
        prev_op = get_op(0)
        for i in range(1, len(nodes) - 1):
            op = get_op(i)
            if op and op.binop and op.binop <= min_precedence + (op.associativity == 'right'):
                if op.prefix:
                    prev_op = get_op(i-1)
                    if prev_op and prev_op.binop and not prev_op.postfix:
                        continue
                if op.postfix:
                    next_op = get_op(i+1)
                    if next_op and next_op.binop and not next_op.prefix:
                        continue
                op_idx = i
                min_precedence = op.binop

        if op_idx is None:
            raise OperatorError('No operator found on line ' + str(Context.line))
        self.op_idx = op_idx
        self.lhs = nodes[:op_idx]
        self.rhs = nodes[op_idx+1:]
        # operator = Op[nodes[op_idx].source_text]
        # mid = nodes[op_idx + 1:right_idx]
        # rhs = nodes[right_idx + 1:]
        # lhs = Expression(lhs) if lhs else None
        # rhs = Expression(rhs) if rhs else None
        # return lhs, operator, mid, rhs

    def evaluate(self):
        op = Op[self.nodes[self.op_idx].source_text]
        if op.static:
            return op.static(self.lhs, self.rhs)
        args = op.eval_args(self.lhs, self.rhs)
        return op.fn.call(args)


class Conditional(Expression):
    condition: Expression
    consequent: FuncBlock
    alt: list[Node] | FuncBlock
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)
        for i, node in enumerate(nodes):
            if isinstance(node, Block):
                self.condition = expressionize(nodes[1:i])
                self.consequent = FuncBlock(node)
                if i+1 == len(nodes):
                    self.alt = []
                    break
                elif i+2 == len(nodes):
                    raise SyntaxErr("Expected else followed by block.  Got ", nodes[i+1])
                try:
                    assert nodes[i+1].source_text == 'else'  # and isinstance(nodes[i+2], Block)
                    node = nodes[i+2]
                    if isinstance(node, Block):
                        self.alt = FuncBlock(node)
                    elif node.source_text == 'if':
                        self.alt = nodes[i+2:]
                    else:
                        assert False
                except AssertionError:
                    raise SyntaxErr("Expected 'else' block or 'else if' after if block.  Got ", nodes[i+1:])
                break

    def evaluate(self):
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call([condition]).value
        if condition:
            return self.consequent.execute()
        elif isinstance(self.alt, FuncBlock):
            return self.alt.execute()
        else:
            return expressionize(self.alt).evaluate()


class Loop(Expression):
    block: FuncBlock
    alt: list[Node]
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)

class ForLoop(Loop):
    var: Option
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)

class WhileLoop(Loop):
    condition: list[Node]
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)

class Command(Expression):
    command: str
    expr: Expression
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)
        self.command = cmd
        self.expr = expressionize(nodes)

    def evaluate(self):
        match self.command:
            case 'exit':
                print('Exiting now')
                exit()
            case 'debug':
                Context.debug = True
                result = self.expr.evaluate()
                return result
            case 'return':
                result = self.expr.evaluate()
                Context.env.return_value = result
                return result
            case 'print':
                print('!@>', BuiltIns['string'].call([self.expr.evaluate()]).value)
                return Value(None)
            case _:
                raise SyntaxErr(f"Unhandled command {self.command}")

class EmptyExpr(Expression):
    def __init__(self):
        pass
    def evaluate(self):
        return Value(None)

class SingleNode(Expression):
    def __init__(self, node: Node, line: int | None, source: str):
        super().__init__(line, source)
        self.node = node
    def evaluate(self):
        return eval_node(self.node)

def eval_node(node: Node) -> Function:
    match node:
        case Statement() as statement:
            return expressionize(statement).evaluate()
        case Token() as tok:
            return eval_token(tok)
        case Block() as block:
            opt = Option(ListPatt(), FuncBlock(block))
            return opt.resolve(None)
        case List() as node:
            # return Value([eval_node(n) for n in node.nodes])
            return Value(list(map(eval_node, node.nodes)))
    raise ValueError(f'Could not evaluate node {node} at line: {node.pos}')


Context.make_expr = expressionize

def precook_args(op: Operator, lhs, rhs) -> list[Value]:
    if op.binop and lhs and rhs:
        args = [expressionize(lhs).evaluate(), expressionize(rhs).evaluate()]
    elif op.prefix and rhs or op.postfix and lhs:
        args = [expressionize(rhs or lhs).evaluate()]
    else:
        raise ArithmeticError("Mismatch between operator type and operand positions.")
    return args


Operator.eval_args = precook_args


def expr_tree_old(nodes: list[Node]):
    pre_op = Op[nodes[0].source_text] if nodes[0].type in (TokenType.Operator, TokenType.Keyword) else None
    post_op = Op[nodes[-1].source_text] if nodes[-1].type in (TokenType.Operator, TokenType.Keyword) else None
    op_idx = right_idx = None
    min_precedence = math.inf
    if pre_op and pre_op.prefix is not None:
        op_idx = right_idx = 0
        min_precedence = pre_op.prefix
    if post_op and post_op.postfix is not None and post_op.postfix < min_precedence:
        op_idx = right_idx = len(nodes) - 1
        min_precedence = post_op.postfix
    # min_precedence = min(getattr(prefix, 'precedence', math.inf), getattr(postfix, 'precedence', math.inf))
    i_tern = min_tern_prec = ternary_second = ternary_op = None
    for i in range(1, len(nodes)-1):
        if ternary_op and nodes[i].source_text == ternary_op.ternary:
            right_idx, min_precedence = i, ternary_op.binop
            op_idx = i_tern
        elif nodes[i].type in (TokenType.Operator, TokenType.Keyword, TokenType.ListStart):
            # if the previous node represents a binary operator, then this one cannot be binary
            op = Op[nodes[i].source_text]
            if op.binop and op.binop < min_precedence or \
                    op.binop == min_precedence and op.associativity == 'left':
                op_idx, right_idx = i, i
                min_precedence = op.binop
                if min_tern_prec and min_precedence < min_tern_prec:
                    i_tern = min_tern_prec = ternary_op = None
                if op.ternary:
                    i_tern, min_tern_prec = op_idx, op.binop
                    ternary_second = op.ternary
                    ternary_op = op

    if op_idx is None:
        raise OperatorError('No operator found on line '+str(Context.line))

    lhs = nodes[:op_idx]
    operator = Op[nodes[op_idx].source_text]
    mid = nodes[op_idx+1:right_idx]
    rhs = nodes[right_idx+1:]
    # lhs = Expression(lhs) if lhs else None
    # rhs = Expression(rhs) if rhs else None
    return lhs, operator, mid, rhs

class OpSplitter:
    op: Operator
    idx: int
    precedence: int
    fix: str  # binop | prefix | postfix
    def __init__(self, op: Operator, index: int, precedence, fix):
        self.op = op
        self.idx = index
        self.precedence = precedence
        self.fix = fix


def eval_token(tok: Token) -> Function:
    s = tok.source_text
    match tok.type:
        case TokenType.Singleton:
            return Value(singletons[s])
            #     case 'none': return Value(None)
            # t = BasicType.none if s == 'none' else BasicType.Float if s == 'inf' else BasicType.Boolean
            # return Value(singleton_mapper(s), t)
        case TokenType.Number:
            return Value(read_number(s))
        case TokenType.String:
            return Value(string(s))
        # case TokenType.Type:
        #     return Value(type_mapper(s), BasicType.Type)
        case TokenType.Name:
            return Context.env.deref(s)
        case TokenType.PatternName:
            return Value(s)
        case _:
            raise Exception("Could not evaluate token", tok)


def string(text: str):
    q = text[0]
    if q == "`":
        return text[1:-1]
    return text[1:-1]
    # TO IMPLEMENT: "string {formatting}"

def make_param(param_nodes: list[Node]) -> Parameter:
    last = param_nodes[-1]
    if isinstance(last, Token) and last.type in (TokenType.Name, TokenType.PatternName):
        name = last.source_text
        param_nodes = param_nodes[:-1]
    else:
        name = None
    if not param_nodes:
        return Parameter(name)
    expr_val = expressionize(param_nodes).evaluate()
    pattern = patternize(expr_val)
    return Parameter(pattern, name)

def split(nodes: list[Node], splitter: TokenType) -> list[list[Node]]:
    start = 0
    groups: list[list[Node]] = []
    for i, node in enumerate(nodes):
        if isinstance(node, Token) and node.type == splitter:
            groups.append(nodes[start:i])
            start = i+1
    if start < len(nodes):
        groups.append(nodes[start:])
    return groups

def read_option(nodes: list[Node]) -> Option:
    dot_option = nodes[0].source_text == '.'
    match nodes:
        case [Token(source_text='.'), node, _, List() as param_list]:
            fn_nodes = [node]
        case [Token(source_text='.'), *fn_nodes]:
            param_list = []
        case [*fn_nodes, _, List() as param_list]:
            param_list = [item.nodes for item in param_list.nodes]
        case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.PatternName) as name_tok]:
            param_list = [[name_tok]]
        case _:
            param_list = split(nodes, TokenType.Comma)
            fn_nodes = False
    if fn_nodes:
        try:
            fn_val = expressionize(fn_nodes).evaluate()
            if fn_val.type == BuiltIns['str']:
                fn_val = Context.env.deref(fn_val.value)
        except NoMatchingOptionError:
            if dot_option:
                raise NoMatchingOptionError(f"Line {Context.line}: "
                                            f"dot option {' '.join((map(str, fn_nodes)))} not found.")
            opt = read_option(fn_nodes)
            if opt.is_null():
                fn_val = Function()
                opt.value = fn_val
            else:
                fn_val = opt.value
            # how many levels deep should this go?
            # This will recurse infinitely, potentially creating many function
        # if fn_val.type != BasicType.Function:
        #     raise RuntimeErr(f"Line {Context.line}: "
        #                      f"Cannot add option to {fn_val.type.value} {' '.join((map(str, fn_nodes)))}")
        fn = fn_val
        definite_env = True
    else:
        fn = Context.env
        definite_env = False
    params = map(make_param, param_list)
    if dot_option:
        patt = ListPatt(Parameter(Prototype(Context.env)), *params)
    else:
        patt = ListPatt(*params)
    try:
        option = fn.select(patt, walk_prototype_chain=False, ascend_env=not definite_env)
        """critical design decision here: I want to have walk_prototype_chain=False so I don't assign variables from the prototype..."""
    except NoMatchingOptionError:
        option = fn.add_option(patt)
    option.dot_option = dot_option
    return option


def read_number(text: str, base=6) -> int | float | Fraction:
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

def write_number(num: int|float|Fraction, base=6, precision=12) -> str:
    """ take a number and convert to a string of digits, possibly separated by / or . """
    if isinstance(num, Fraction):
        return write_number(num.numerator) + "/" + write_number(num.denominator)
    sign = "-" * (num < 0)
    num = abs(num)
    whole = int(num)
    frac = num - whole

    ls = sign + nat2str(whole, base)
    if frac == 0:
        return ls
    rs = frac_from_base(frac, base, precision)
    return f"{ls}.{''.join(str(d) for d in rs)}"

def nat2str(num: int, base=6):
    if num < base:
        return str(num)
    else:
        return nat2str(num // base, base) + str(num % base)

def frac_from_base(num: float, base=6, precision=12):
    digits = []
    remainders = []
    for i in range(precision):
        tmp = num * base
        itmp = int(tmp)
        num = tmp - itmp
        if 1-num < base ** (i-precision):
            digits += [itmp+1]
            break
        digits += [itmp]
    return digits


if __name__ == "__main__":
    n = read_number("40000.555555555555555")
    while True:
        strinput = input()
        n = read_number(strinput)
        # n = 999999999/100000000
        print("decimal: " + str(n))
        print("senary: " + write_number(n, 6, 25))
        print("via bc:", base(str(n), 10, 6, 15, True, True))

