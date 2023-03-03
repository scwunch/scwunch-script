from Syntax import Node, Token, TokenType, List, Statement, Block
from DataStructures import Value, Function, Pattern, Context, Parameter, NoMatchingOptionError
from BuiltIns import Op, BuiltIns
from Expressions import Expression, get_option, eval_node

def static_fn(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    return Value("this is just a function signature")

# def assign_val(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
#     value = Expression(rhs).evaluate()
#     last_node = lhs[-1]
#     if isinstance(last_node, List):
#         if len(lhs) > 1:
#             param_pattern = evaluate_pattern(last_node)
#             fn = Expression(lhs[:-2]).evaluate()
#             assert isinstance(fn, Function)
#             fn.assign_option(param_pattern, value)
#             return value
#         else:
#             lhs = last_node
#     pattern = evaluate_pattern(lhs)
#     Context.env.assign_option(pattern, clone(value))
#     return value
#
# def assign_fn(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
#     if len(rhs) == 1 and isinstance(rhs[0], Block):
#         value = eval_node(rhs[0])
#     else:
#         return_statement = Statement([Token('return')] + rhs)  # noqa
#         value = Function(block=Block([return_statement]), env=Context.env)
#
#     last_node = lhs[-1]
#     if isinstance(last_node, List):
#         if len(lhs) > 1:
#             param_pattern = evaluate_pattern(last_node)
#             if len(lhs) == 3 and lhs[0].type == TokenType.Name:
#                 fn = eval_node(lhs[0])
#                 if fn.is_null():
#                     fn = Context.env.assign_option(
#                         Pattern(Parameter(lhs[0].source_text)),
#                         Function(env=Context.env))
#             else:
#                 fn = Expression(lhs[:-2]).evaluate()
#             # if fn.is_null():
#             #     fn = Context.env.assign_option()
#             assert isinstance(fn, Function)
#             value.env = fn
#             fn.assign_option(param_pattern, value)
#             return value
#         else:
#             lhs = last_node
#     pattern = evaluate_pattern(lhs)
#     Context.env.assign_option(pattern, value)
#     return value
#

# def assign_alias(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
#     value = Expression(rhs).evaluate()
#     last_node = lhs[-1]
#     if isinstance(last_node, List):
#         if len(lhs) > 1:
#             param_pattern = evaluate_pattern(last_node)
#             fn = Expression(lhs[:-2]).evaluate()
#             assert isinstance(fn, Function)
#             fn.assign_option(param_pattern, value)
#             return value
#         else:
#             lhs = last_node
#     pattern = evaluate_pattern(lhs)
#     Context.env.assign_option(pattern, value)
#     return value

def assign_val(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    value = Expression(rhs).evaluate().clone()
    option = get_option(lhs)
    option.value = value
    return value

def assign_alias(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    value = Expression(rhs).evaluate()
    option = get_option(lhs)
    option.value = value
    return value

def assign_fn(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    if len(rhs) == 1 and isinstance(rhs[0], Block):
        block = rhs[0]
    else:
        return_statement = Statement([Token('return')] + rhs)  # noqa
        block = Block([return_statement])
    option = get_option(lhs)
    option.block = block
    return Value(None)


Op['='].static = assign_val
Op[':='].static = assign_alias
Op[':'].static = assign_fn

# Op['+='].static = lambda a, _, b: assign_val(Op['+'].fn.call())
# def assign_with_operator()
#     val = op.assign_op.fn.call([Expression(self.lhs).evaluate(), Expression(self.rhs).evaluate()])


def or_fn(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    condition = Expression(lhs).evaluate()
    return condition if BuiltIns['bool'].call([condition]).value else Expression(rhs).evaluate()
Op['or'].static = or_fn

def and_fn(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    condition = Expression(lhs).evaluate()
    return Expression(rhs).evaluate() if BuiltIns['bool'].call([condition]).value else condition
Op['and'].static = and_fn

def if_fn(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    condition = Expression(mid).evaluate()
    if BuiltIns['bool'].call([condition]).value:
        return Expression(lhs).evaluate()
    else:
        return Expression(rhs).evaluate()
Op['if'].static = if_fn

def option_exists(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    try:
        Expression(lhs).evaluate()
        return Value(True)
    except NoMatchingOptionError:
        return Value(False)
Op['?'].static = option_exists

def nullish_or(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    try:
        return Expression(lhs).evaluate()
    except NoMatchingOptionError:
        return Expression(rhs).evaluate()
Op['??'].static = nullish_or

def dot_op(lhs: list[Node], mid: list[Node], rhs: list[Node]) -> Value:
    assert len(mid) == 1
    fn = Expression(lhs).evaluate()
    name = mid[0].source_text
    if isinstance(fn, Function):
        try:
            option = fn.deref(name)
        except NoMatchingOptionError:
            option = Context.env.deref()


