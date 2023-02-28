import contextlib

from baseconvert import base
import math
from Syntax import *
from Abstract_Syntax_Tree import Tokenizer, AST
from DataStructures import *
# from BuiltIns import *


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
    def __init__(self, nodes: list[Node]):
        # self.type = ExprType.Unknown
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
            case ExprType.Empty: return Value(None)
            case ExprType.Single: return eval_node(self[0])
            case ExprType.Conditional:
                if self.condition.evaluate().value:
                    return self.consequent.evaluate()
                elif self.alt:
                    return self.alt.evaluate().value
                else:
                    return Value(None)
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
            consequent = node
        elif node.source_text == 'else':
            alt = Expression(nodes[i+1:])
            break
    else:
        alt = None
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
            return Value(None)
        case 'return':
            result = expr.evaluate()
            Context.env.return_value = result
            return result
        case 'print':
            print('!@>', BuiltIns['str'].call([expr.evaluate()]).value)
            return Value(None)
        case _:
            raise Exception(f"Unhandled command {cmd}")

def expr_tree(nodes: list[Node]):
    pre_op = Op[nodes[0].source_text] if nodes[0].type == TokenType.Operator else None
    post_op = Op[nodes[-1].source_text] if nodes[-1].type == TokenType.Operator else None
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
            op = Op[nodes[i].source_text]
            if op.binop < min_precedence or \
                    op.binop == min_precedence and op.associativity == 'left':
                op_idx, right_idx, min_precedence = i, i, op.binop
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
        case Statement() as node:
            return Expression(node.nodes).evaluate()
        case Token() as node:
            return eval_token(node)
        case Block() as node:
            return Value(Function(block=node, env=Context.env), BasicType.Function)
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
            return Value(number(s))
        case TokenType.String:
            return Value(string(s), BasicType.String)
        case TokenType.Name | TokenType.Type:
            return Context.env.deref(s)
        case TokenType.PatternName:
            return Value(s, BasicType.Name)
        case TokenType.PatternName:
            return Value(Pattern(Parameter(name=s)), BasicType.Pattern)
        case _:
            raise Exception("Could not evaluate token", tok)

def number(text: str) -> int | float:
    if isinstance(text, int) or isinstance(text, float):
        return text
    if text.endswith('d'):
        text = text[:-1]
    else:
        try:
            text = base(text, 6, string=True, recurring=False)
        except ValueError:
            pass
    if '.' in text:
        return float(text)
    else:
        return int(text)

def make_param(param_nodes: list[Node]) -> Parameter:
    last = param_nodes[-1]
    if isinstance(last, Token) and last.type in (TokenType.Name, TokenType.PatternName):
        name = last.source_text
        param_nodes = param_nodes[:-1]
    else:
        name = None
    if not param_nodes:
        return Parameter(name)
    if len(param_nodes) == 1 and isinstance(tok := param_nodes[0], Token):
        basic_type = type_mapper(tok.source_text)
        if basic_type:
            return Parameter(name, basic_type=basic_type)
    value = Expression(param_nodes).evaluate()
    if isinstance(value, Pattern):
        value[0].name = name
        return value[0]
    if value.type == BasicType.Type:
        return Parameter(name, basic_type=value.value)
    return Parameter(name, value)

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

def evaluate_pattern(pattern_nodes: list[Node] | List) -> Pattern:
    if isinstance(pattern_nodes, List):
        param_statements = (statement.nodes for statement in pattern_nodes.nodes)
    else:
        param_statements = split(pattern_nodes, TokenType.Comma)
    parameters = (make_param(ps_nodes) for ps_nodes in param_statements)
    return Pattern(*parameters)

def get_option(nodes: list[Node]) -> Option:
    fn = Context.env
    match nodes:
        case [*fn_nodes, _, List() as param_list]:
            if fn_nodes:
                try:
                    fn = Expression(fn_nodes).evaluate()
                    if fn.type != BasicType.Function:
                        raise RuntimeErr(f"Line {Context.line}: Cannot add option to {fn.type.value} {' '.join((map(str, fn_nodes)))}")
                    fn = fn.value
                except NoMatchingOptionError:
                    fn = get_option(fn_nodes).function
                    # how many levels deep should this go?
                    # This will recurse infinitely, potentially creating many functions
            param_list = [item.nodes for item in param_list.nodes]
        case _:
            param_list = split(nodes, TokenType.Comma)
    params = map(make_param, param_list)
    patt = Pattern(*params)
    return fn.select(patt, create_if_not_exists=True)

    fn = patt = name = None
    if last_node.type in (TokenType.Name, TokenType.PatternName):
        name = last_node.source_text
        nodes = nodes[:-1]
    match Expression(nodes).evaluate().value:
        case None:
            pass
        case Pattern() as patt:
            pass
        case Function() as fn:
            pass
        case _:
            pass
    if isinstance(last_node, List):
        if len(nodes) > 1:
            param_pattern = evaluate_pattern(last_node)
            if len(nodes) == 3 and nodes[0].type == TokenType.Name:
                fn = eval_node(nodes[0])
                if fn.is_null():  # if function does not exist, create it
                    fn = Context.env.assign_option(
                        Pattern(Parameter(nodes[0].source_text)),
                        Function(env=Context.env))
            else:
                fn = Expression(nodes[:-2]).evaluate()
            return fn.select(param_pattern)
        else:
            nodes = last_node

    pattern = evaluate_pattern(nodes)
    return Context.env.select(pattern)


def get_option_old(nodes: list[Node]) -> Option:
    last_node = nodes[-1]
    if isinstance(last_node, List):
        if len(nodes) > 1:
            param_pattern = evaluate_pattern(last_node)
            if len(nodes) == 3 and nodes[0].type == TokenType.Name:
                fn = eval_node(nodes[0])
                if fn.is_null():  # if function does not exist, create it
                    fn = Context.env.assign_option(
                        Pattern(Parameter(nodes[0].source_text)),
                        Function(env=Context.env))
            else:
                fn = Expression(nodes[:-2]).evaluate()
            return fn.select(param_pattern)
        else:
            nodes = last_node
    pattern = evaluate_pattern(nodes)
    return Context.env.select(pattern)

def execute(fn: Function, args: list[Value] = None) -> Value:
    # fn = fn.copy()
    # for arg in args:
    #     fn.assign_option(pattern, arg)
    Context.push(fn)
    print(Context.env)
    for statement in fn.block.statements:
        Context.line = statement.pos[0]
        expr = Expression(statement.nodes)
        result = expr.evaluate()
        if fn.return_value:
            Context.pop()
            return fn.return_value
        # if isinstance(result, Action):
        #     match result.action:
        #         case 'return':
        #             Context.pop()
        #             return Value(result.value)
        #         case 'assign':
        #             fn.assign_option(result.pattern, Function(value=Value(result.value)))
    Context.pop()
    fn.return_value = Value(fn)
    return fn.return_value

Function.exec = execute

def string(text: str):
    q = text[0]
    if q == "`":
        return text[1:-1]
    return text[1:-1]
    # TO IMPLEMENT: "string {formatting}"
