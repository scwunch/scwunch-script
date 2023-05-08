from Syntax import Node, Token, TokenType, List, Statement, Block
from Env import Context, NoMatchingOptionError, OperatorError, SyntaxErr
from DataStructures import Value, Function, FuncBlock
from BuiltIns import Op, BuiltIns
from Expressions import expressionize, read_option, eval_node

def static_fn(lhs: list[Node], rhs: list[Node]) -> Value:
    return Value("this is just a function signature")


def assign_val(lhs: list[Node], rhs: list[Node]) -> Value:
    value = expressionize(rhs).evaluate()   # .clone()
    option = read_option(lhs, True)
    option.value = value
    if not value.name:
        value.name = option.pattern.name
    return value

def assign_alias(lhs: list[Node], rhs: list[Node]) -> Value:
    value = expressionize(rhs).evaluate()
    option = read_option(lhs)
    option.value = value
    return value

def assign_fn(lhs: list[Node], rhs: list[Node]) -> Value:
    blk_nd = rhs[0]
    if len(rhs) == 1 and isinstance(blk_nd, Block):
        block: Block = blk_nd
    else:
        return_statement = Statement([Token('return')] + rhs)  # noqa
        block = Block([return_statement])
    option = read_option(lhs)
    option.block = FuncBlock(block)
    return Value(None)


# Op['='].static = assign_val
Op[':='].static = assign_alias
Op[':'].static = assign_fn

# Op['+='].static = lambda a, _, b: assign_val(Op['+'].fn.call())
# def assign_with_operator()
#     val = op.assign_op.fn.call([expressionize(self.lhs).evaluate(), expressionize(self.rhs).evaluate()])


def or_fn(lhs: list[Node], rhs: list[Node]) -> Value:
    condition = expressionize(lhs).evaluate()
    return condition if BuiltIns['bool'].call([condition]).value else expressionize(rhs).evaluate()
Op['or'].static = or_fn

def and_fn(lhs: list[Node], rhs: list[Node]) -> Value:
    condition = expressionize(lhs).evaluate()
    return expressionize(rhs).evaluate() if BuiltIns['bool'].call([condition]).value else condition
Op['and'].static = and_fn

# def if_fn(lhs: list[Node], rhs: list[Node]) -> Value:
#     condition = expressionize(mid).evaluate()
#     if BuiltIns['bool'].call([condition]).value:
#         return expressionize(lhs).evaluate()
#     else:
#         return expressionize(rhs).evaluate()
def if_fn(lhs: list[Node], rhs: list[Node]) -> Value:
    for i in reversed(range(len(rhs))):
        if rhs[i].source_text == 'else':
            condition = expressionize(rhs[:i])
            rhs = rhs[i + 1:]
            condition = condition.evaluate()
            break
    else:
        raise SyntaxErr(f"Line {Context.line}: If statement with no else clause")
    if BuiltIns['bool'].call([condition]).value:
        return expressionize(lhs).evaluate()
    else:
        return expressionize(rhs).evaluate()
Op['if'].static = if_fn

def option_exists(lhs: list[Node], rhs: list[Node]) -> Value:
    try:
        last = lhs[-1]
        if isinstance(last, List):
            key = eval_node(last)
        elif last.type == TokenType.Name:
            key = [Value(last.source_text)]
        else:
            raise OperatorError(f"Line {Context.line}: right-most arg of ? operator must be a name or arg-list.")
        if len(lhs) == 1:
            fn = Context.env
        else:
            # assert lhs[-1] is '.' or '.['
            fn = expressionize(lhs[:-2]).evaluate()
            # assert isinstance(fn, Function)
        fn.select_and_bind(key)
        return Value(True)
    except NoMatchingOptionError:
        return Value(False)
Op['?'].static = option_exists

def nullish_or(lhs: list[Node], rhs: list[Node]) -> Value:
    try:
        return expressionize(lhs).evaluate()
    except NoMatchingOptionError:
        return expressionize(rhs).evaluate()
Op['??'].static = nullish_or


