from fractions import Fraction
from Syntax import *
from DataStructures import *
from Env import *
import numpy


class ExprType(Enum):
    Unknown = '?'
    Empty = 'empty'
    Single = 'single node'
    Option = 'option = statement'
    Conditional = 'conditional'
    Loop = 'loop'
    Command = 'command'
    Mathological = 'mathological'


class Expression:
    line: int = None
    nodes: list[Node]
    type: ExprType = None
    operator: Operator
    condition = None
    consequent = None
    alt = None
    var = None
    block = None
    operator = None
    lhs = None
    mid = None
    rhs = None
    def __init__(self, nodes: list[Node] | Statement):
        # self.type = ExprType.Unknown
        if isinstance(nodes, Statement):
            self.line = nodes.pos[0]
            nodes = nodes.nodes
        self.nodes = nodes
        if not nodes:
            self.type = ExprType.Empty
        elif len(nodes) == 1:
            if nodes[0].type == TokenType.Command:
                self.type = ExprType.Command
            else:
                self.type = ExprType.Single
        else:
            match nodes[0].source_text:
                case 'if':
                    self.type = ExprType.Conditional
                    self.condition, self.consequent, self.alt = conditional(nodes)
                case 'for':
                    self.type = ExprType.Loop
                    self.var, self.iterable, self.block = for_loop(nodes)
                case 'while':
                    self.type = ExprType.Loop
                    self.condition, self.block = while_loop(nodes)
                case _:
                    if nodes[0].type == TokenType.Command:
                        self.type = ExprType.Command
                    else:
                        self.type = ExprType.Mathological
                        self.lhs, self.operator, self.mid, self.rhs = expr_tree(nodes)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, item):
        return self.nodes[item]

    def evaluate(self):
        match self.type:
            case ExprType.Empty:
                return Value(None)
            case ExprType.Single:
                return eval_node(self[0])
            case ExprType.Conditional:
                if self.condition.evaluate().value:
                    return self.consequent.evaluate()
                else:
                    return self.alt.evaluate()
            case ExprType.Loop:
                return 'loop not implemented'
            case ExprType.Command:
                return eval_command(self[0].source_text, Expression(self[1:]))
            case ExprType.Mathological:
                op = self.operator
                if op.static:
                    return op.static(self.lhs, self.mid, self.rhs)
                args = op.prepare_args(self.lhs, self.mid, self.rhs)
                return op.fn.call(args)
            case _:
                return f'expr_type {self.type} not yet implemented'

    def __repr__(self):
        match self.type:
            case ExprType.Empty:
                return '[empty expression]'
            case ExprType.Single:
                return repr(self[0])
            case ExprType.Conditional:
                return f'if {self.condition} then {self.consequent}{" else "+repr(self.alt) if self.alt else ""}'
            case ExprType.Loop:
                return 'loop'
            case ExprType.Command:
                return f"{self[0].source_text} expression"
            case ExprType.Mathological:
                return f"{self.lhs}  {self.operator}  {self.rhs}"


Context.make_expr = Expression

def precook_args(op: Operator, lhs, mid, rhs) -> list[Value]:
    if op.ternary and lhs and mid and rhs:
        args = [Expression(lhs).evaluate(), Expression(mid).evaluate(), Expression(rhs).evaluate()]
    elif op.binop and lhs and rhs:
        args = [Expression(lhs).evaluate(), Expression(rhs).evaluate()]
    elif op.prefix and rhs:
        args = [Expression(rhs).evaluate()]
    elif op.postfix and lhs:
        args = [Expression(lhs).evaluate()]
    else:
        raise ArithmeticError("Mismatch between operator type and operand positions.")
    return args

Operator.prepare_args = precook_args

def conditional(nodes: list[Node]):
    for i, node in enumerate(nodes):
        if isinstance(node, Block):
            condition = Expression(nodes[1:i])
            consequent = Expression([node])
        elif node.source_text == 'else':
            alt = Expression(nodes[i+1:])
            break
    else:
        alt = Expression([])
    return condition, consequent, alt

def for_loop(nodes: list[Node]):
    pass

def while_loop(nodes: list[Node]):
    pass

def eval_command(cmd: str, expr: Expression):
    match cmd:
        case 'exit':
            print('Exiting now')
            exit()
        case 'debug':
            Context.debug = True
            result = expr.evaluate()
            return result
        case 'return':
            result = expr.evaluate()
            Context.env.return_value = result
            return result
        case 'print':
            print('!@>', BuiltIns['string'].call([expr.evaluate()]).value)
            return Value(None)
        case _:
            raise Exception(f"Unhandled command {cmd}")

def expr_tree(nodes: list[Node]):
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
        elif nodes[i].type in (TokenType.Operator, TokenType.OptionSet, TokenType.Keyword, TokenType.ListStart):
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

def eval_node(node: Node) -> Value:
    match node:
        case Statement() as statement:
            return Expression(statement.nodes).evaluate()
        case Token() as tok:
            return eval_token(tok)
        case Block() as block:
            opt = Option(ListPatt(), FuncBlock(block))
            return opt.resolve(None)
        case List() as node:
            # return Value([eval_node(n) for n in node.nodes])
            return Value(list(map(eval_node, node.nodes)), BasicType.List)
    raise ValueError(f'Could not evaluate node {node} at line: {node.pos}')

def eval_token(tok: Token) -> Value:
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
            return Value(string(s), BasicType.String)
        case TokenType.Type:
            return Value(type_mapper(s), BasicType.Type)
        case TokenType.Name:
            return Context.env.deref(s)
        case TokenType.PatternName:
            return Value(s, BasicType.Name)
        case TokenType.PatternName:
            return Value(ListPatt(Parameter(s)), BasicType.Pattern)
        case _:
            raise Exception("Could not evaluate token", tok)

def read_number(text: str) -> int | float | Fraction:
    if isinstance(text, int) or isinstance(text, float) or isinstance(text, Fraction):
        return text
    try:
        return int(text)
    except ValueError:
        parts = text.partition('/')
        return float(text)

    # if text.endswith('d'):
    #     text = text[:-1]
    # else:
    #     try:
    #         text = base(text, 6, string=True, recurring=False)
    #     except ValueError:
    #         pass
    # if '.' in text:
    #     return float(text)
    # else:
    #     return int(text)



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
    expr_val = Expression(param_nodes).evaluate()
    pattern = make_patt(expr_val)
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
            fn_val = Expression(fn_nodes).evaluate()
            if fn_val.type == BasicType.Name:
                fn_val = Context.env.deref(fn_val.value)
        except NoMatchingOptionError:
            if dot_option:
                raise NoMatchingOptionError(f"Line {Context.line}: "
                                            f"dot option {' '.join((map(str, fn_nodes)))} not found.")
            opt = read_option(fn_nodes)
            if opt.is_null():
                fn_val = Value(Function())
                opt.value = fn_val
            else:
                fn_val = opt.value
            # how many levels deep should this go?
            # This will recurse infinitely, potentially creating many function
        if fn_val.type != BasicType.Function:
            raise RuntimeErr(f"Line {Context.line}: "
                             f"Cannot add option to {fn_val.type.value} {' '.join((map(str, fn_nodes)))}")
        fn = fn_val.value
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
    except NoMatchingOptionError:
        option = fn.add_option(patt)
    option.dot_option = dot_option
    return option

""" takes a number in the form of a string of digits, or digits separated by a / or .
    if the number ends with a d, it will be read as a decimal"""
def read_number(text: str, base=6) -> int | float | Fraction:
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


def num2str(num: int|float|Fraction, base=6, precision=12):
    if isinstance(num, Fraction):
        return num2str(num.numerator)+"/"+num2str(num.denominator)
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

def frac2str(num: float, base=6):
    digits = []
    for i in range(10):
        dig = int(num*base)
        num = num*base-dig
        digits += [dig]
    return digits

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

from baseconvert import base
s = str(1 / 5)
n = base(s, 10, 6, 25, True, False)
print(n)



if __name__ == "__main__":
    n = read_number("40000.555555555555555")
    while True:
        strinput = input()
        n = read_number(strinput)
        # n = 999999999/100000000
        print("decimal: " + str(n))
        print("senary: "+num2str(n, 6, 25))
        print("via bc:", base(str(n), 10, 6, 15, True, True))

