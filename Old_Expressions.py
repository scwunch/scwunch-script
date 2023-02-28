from baseconvert import base
import math
from DataStructures import *
from Abstract_Syntax_Tree import Tokenizer, AST
from DataStructures import *
from BuiltIns import *





# def execute_block(block: Block, function: Function):
#     for statement in block.statements:
#         expr = Expression(statement.nodes, function)
#         result = expr.evaluate()
#         if isinstance(result, Return):
#             return result.value


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


class Expression:
    nodes: list[Node]
    type: ExprType
    assignment_operator_index: int | None
    # context: Function | None
    eval: any

    def __init__(self, nodes: list[Node]):  # , context: Function = None):
        self.nodes = nodes
        self.type = self.determine_expr_type()
        # self.context = context

    def determine_expr_type(self):
        nodes = self.nodes
        if not nodes:
            self.eval = lambda expr, context: None
            return ExprType.Empty

        # for i, node in enumerate(nodes):
        #     if isinstance(node, Token) and node.type == TokenType.OptionSet:
        #         # this code switches the `name[list,of,params]: <block>` to `name = [list,of,params] => <block>`
        #         # if node.source_text == ':' and isinstance(nodes[i-1], List):
        #         #     self.nodes = [*nodes[:i-1], Token('=', node.pos), nodes[i-1], Token('=>', node.pos), nodes[i+1:]]
        #         #     self.assignment_operator_index = i-1
        #         # else:
        #         #     self.assignment_operator_index = i
        #         self.assignment_operator_index = i
        #         return ExprType.Option

        tok = nodes[0]
        if isinstance(tok, Token):
            if tok.type == TokenType.Command:
                self.eval = evaluate_command
                return ExprType.Command
            if len(nodes) == 1:
                return ExprType.Token

        self.eval = evaluate_mathological
        return ExprType.Mathological

    def evaluate(self, context: Function):
        # return self.evaluate(self)
        match self.type:
            case ExprType.Unknown:
                raise Exception('Unknown expression type at ', self.nodes[0].pos)
            case ExprType.Empty:
                return Value(None)
            case ExprType.Token:
                return evaluate_token(self.nodes[0], context)
            case _:
                return self.eval(self.nodes, context)
            # case ExprType.Option:
            #     i = self.assignment_operator_index
            #     param_nodes = self.nodes[:i]
            #     expr_nodes = self.nodes[i+1:]
            #     option = make_option(param_nodes, option_type_mapper(self.nodes[i].source_text), expr_nodes)
            #     return Assign(option)
            # case ExprType.Mathological:
            #     return evaluate_mathological(self.nodes)
            # case _:
            #     raise Exception('Unrecognized expression type: ', self.type)

    def make_option(self):
        i = self.assignment_operator_index
        assignment_type = option_type_mapper(self.nodes[i].source_text)
        expr_nodes = self.nodes[i+1:]
        block, value = None, None
        if assignment_type == OptionType.Function:
            if isinstance(expr_nodes[-1], Block):
                block = expr_nodes[-1]
                if len(expr_nodes) == 2:
                    return_type = Expression([expr_nodes[0]]).eval()
            else:
                block = Block([Statement([Token('return'), *expr_nodes])])
            last_param_node = self.nodes[i - 1]
            fn_params = []
            if isinstance(last_param_node, List):
                fn_params = make_params(last_param_node.nodes)
                i -= 1
            value = Function(block=block)
            fn_option = Option(fn_params, value)
            value.assign_option(fn_option, )
        else:
            value = Expression(expr_nodes).eval()
        param_set = split(self.nodes[:i], TokenType.Comma)

        parameters = make_params(self.nodes[:i])

        return Option(parameters, value)


def make_option(param_nodes: list[Node], assignment_type: OptionType, expr_nodes: list[Node]):
    block, value = None, None
    if assignment_type == OptionType.Function:
        last_param_node = param_nodes[-1]
        if isinstance(last_param_node, List):
            fn_option = make_option(last_param_node.nodes, OptionType.Function, expr_nodes)
            value = Function(options=[fn_option])
        else:
            if isinstance(expr_nodes[-1], Block):
                block = expr_nodes[-1]
                if len(expr_nodes) > 1:
                    return_type = Expression(expr_nodes[:-1]).eval()
            else:
                block = Block([Statement([Token('return'), *expr_nodes])])
            value = Function([], block)
    else:
        value = Expression(expr_nodes).eval()

    parameters = make_params(param_nodes)
    return Option(parameters, value)


def make_params(nodes: list[Node]) -> list[Parameter]:
    param_sets = split(nodes, TokenType.Comma)

    def make_param(param_nodes) -> Parameter:
        last = param_nodes[-1]
        if isinstance(last, Token) and last.type == TokenType.Name:
            name = last.source_text
            param_nodes.pop()
        else:
            name = None
        if not param_nodes:
            return Parameter(name)
        if len(param_nodes) == 1 and isinstance(tok := param_nodes[0], Token):
            basic_type = type_mapper(tok.source_text)
            if basic_type:
                return Parameter(name, basic_type=basic_type)
        value = Expression(param_nodes).eval()
        if isinstance(value, Parameter):
            value.name = name
            return value
        return Parameter(name, value)

    return list(map(make_param, param_sets))


def evaluate_node(node: Node, context: Function) -> Value:
    if isinstance(node, Statement):
        return Expression(node.nodes).evaluate(context)
    if isinstance(node, Token):
        return evaluate_token(node, context)
    if isinstance(node, Block):
        return Function(block=node)
    if isinstance(node, List):
        return Value([evaluate_node(n, context) for n in node.nodes])


def execute(fn: Function, args: list[Value] = None) -> Value:
    # fn = fn.copy()
    # for arg in args:
    #     fn.assign_option(pattern, arg)
    Context.push(fn)
    print(Context.env)
    for statement in fn.block.statements:
        expr = Expression(statement.nodes)
        result = expr.evaluate(context=fn)
        if isinstance(result, Action):
            match result.action:
                case 'return':
                    Context.pop()
                    return Value(result.value)
                case 'assign':
                    fn.assign_option(result.pattern, Value(result.value))
    Context.pop()
    return fn


Function.exec = execute


def evaluate_token(tok, context: Function) -> Value:
    s = tok.source_text
    match tok.type:
        case TokenType.Singleton:
            t = BasicType.none if s == 'none' else BasicType.Float if s == 'inf' else BasicType.Boolean
            return Value(singleton_mapper(s), t)
        case TokenType.Number:
            return Value(number(s))
        case TokenType.String:
            return Value(string(s, context))
        case TokenType.Name | TokenType.Type:
            return Context.env.call(s)
        case TokenType.PatternName:
            return Pattern(Parameter(name=s))
        case _:
            raise Exception("Could not evaluate token", tok)


def number(text: str):
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

    total = 0
    for d in int_part:
        total = 6*total + int(d)
    if not frac_part:
        return total
    for p, d in enumerate(text.split('.')):
        total += int(d) / (6 ** (p+1))
    return total


def string(text: str, context):
    return text[1:-1]
    # TO IMPLEMENT: "string {formatting}"


def evaluate_command(nodes: list[Node], context: Function):
    sub_expr = Expression(nodes[1:])
    match nodes[0].source_text:
        case 'exit':
            print('Exiting now')
            exit()
        case 'debug':
            breakpoint()
        case 'return':
            return Action(sub_expr.evaluate(context), 'return')
        case 'print':
            print('!@>', BuiltIns['str'].call([sub_expr.evaluate(context)]))
        case _:
            raise Exception(f"Unhandled command {nodes[0].source_text}")

class ExprTree:
    operator: Operator
    value: Value | None
    def __init__(self, nodes: list[Node], context):
        if not nodes:
            self.value = Value(None)
        elif len(nodes) == 1:
            self.value = evaluate_node(nodes[0], context)
        else:
            self.value = None
            prefix = Op[nodes[0].source_text] if nodes[0].type == TokenType.Operator else None
            postfix = Op[nodes[-1].source_text] if nodes[-1].type == TokenType.Operator else None
            op_idx = right_idx = None
            min_precedence = math.inf
            if prefix:
                op_idx, right_idx = 0, 1
                min_precedence = prefix.precedence
            if postfix and postfix.precedence < min_precedence:
                op_idx, right_idx = len(nodes) - 1, len(nodes)
                min_precedence = postfix.precedence
            # min_precedence = min(getattr(prefix, 'precedence', math.inf), getattr(postfix, 'precedence', math.inf))
            i_tern = min_tern_prec = ternary_second = ternary_op = None
            for i in range(1, len(nodes)-1):
                if nodes[i].type in (TokenType.Operator, TokenType.OptionSet, TokenType.Keyword, TokenType.ListStart):
                    op = Op[nodes[i].source_text]
                    if nodes[i].source_text == ternary_second:
                        right_idx, min_precedence = i, op.precedence
                        op_idx = i_tern
                        ternary_op = True
                        continue
                    if op.precedence < min_precedence or \
                            op.precedence == min_precedence and op.associativity == 'left':
                        op_idx, right_idx, min_precedence = i, i+1, op.precedence
                        ternary_op = None
                        if op.ternary:
                            i_tern, min_tern_prec = op_idx, op.precedence
                            ternary_second = op.ternary

            if op_idx is None:
                raise Exception('no operator found')

            lhs = nodes[:op_idx]
            self.operator = Op[nodes[op_idx].source_text]
            rhs = nodes[right_idx:] if right_idx < len(nodes) else []
            self.lhs = ExprTree(lhs, context) if lhs else None
            self.rhs = ExprTree(rhs, context) if rhs else None

    def evaluate(self):
        if self.value:
            return self.value
        op, args = self.operator, []
        if op.binop and self.lhs and self.rhs:
            args = [self.lhs.evaluate(), self.rhs.evaluate()]
        elif op.prefix and self.rhs:
            args = [self.rhs.evaluate()]
        elif op.postfix and self.lhs:
            args = [self.lhs.evaluate()]
        else:
            raise ArithmeticError("Operator in the wrong spot, or has no arguments")
        return op.fn.call(args)

    def __repr__(self):
        if self.value:
            return repr(self.value)
        else:
            return f"{self.lhs}  {self.operator}  {self.rhs}"

def evaluate_mathological(nodes: list[Node], context: Function):
    return ExprTree(nodes, context).evaluate()
    # assume the first token is operand and the second is operator; initialize stack
    operands: list[Value] = []
    operators: list[Operator] = []

    def collapse():
        op = operators.pop(0)
        args = [operands.pop(0)]
        if op.binop:
            args.insert(0, operands.pop(0))
        operands.insert(0, op.fn.call(args))

    for node in nodes:
        if node.type not in [TokenType.Operator, TokenType.Keyword]:
            operands.insert(0, evaluate_node(node, context))
        else:
            op: Operator = Op[node.source_text]
            if op.prefix:
                # no collapse
                pass
            else:
                while operators and op.precedence <= operators[0].precedence:
                    if len(operands) < 2 and operators[0].binop:
                        break
                    collapse()
            operators.insert(0, op)

    while len(operators):
        collapse()
    assert len(operands) == 1  # else "Operators and Operands don't quite add up properly."
    return operands[0]




if __name__ == "__main__":
    script_path = "test_script.ss"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    ast = AST(tokenizer)

