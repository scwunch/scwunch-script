import re
from fractions import Fraction
from Syntax import Block, Statement
from Env import *
from DataStructures import *
from Expressions import read_option
from BuiltIns import *

def eval_set_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    value = expressionize(rhs).evaluate()  # .clone()
    fn = None
    match lhs:
        case [Token(type=TokenType.Name, source_text=name)]:
            key = Value(name)
        # case [Token(type=TokenType.Number|TokenType.String) as tok]:
        #     key = eval_token(tok)
        case [List(nodes=statements)]:
            key = Value(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
        case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
            key = Value(name)
            fn = expressionize(fn_nodes).evaluate()
        case [*fn_nodes, Token(source_text='.'|'.['), List() as list_node]:
            key = expressionize([list_node]).evaluate()
            fn = expressionize(fn_nodes).evaluate()
        case _:
            raise SyntaxErr(f'Line {Context.line}: Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
    if not value.name and hasattr(key, 'value') and isinstance(key.value, str):
        value.name = key.value
    if fn is None:
        return [key, value]
    return [fn, key, value]

def eval_alias_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    left, right = eval_set_args(lhs, [])[:-1], eval_set_args(rhs, [])
    right_key = right[-2]
    if len(right) == 3:
        right_fn = right[-3]
        ascend = False
    else:
        right_fn = Context.env
        ascend = True
    # right_fn = right[-3] if len(right)==3 else Context.env
    match right_key.value:
        case str() as name:
            option = right_fn.select_by_name(name, ascend)
        case [Value(value=str() as name)]:
            option = right_fn.select_by_name(name, ascend)
        case list() as args:
            option, _ = right_fn.select_and_bind(args, ascend_env=ascend)
        case _:
            raise TypeErr(f"Line {Context.line}: Sorry, I'm not sure how to alias {right_key}")
    # option = right_fn.select_by_name(right_key.value, ascend)
    return [*left, option]

def assign_var(key: Value, val: Function):
    key_value = key.value
    if isinstance(key_value, str):
        name = key_value
        if name in Context.env.named_options:
            option = Context.env.named_options[name]
            # option.nullify()
            option.assign(val)
        else:
            option = Context.env.add_option(name, val)
        # try:
        #     option = Context.env.select_by_name(name, ascend_env=False)
        #     option.nullify()
        #     option.assign(val)
        # except NoMatchingOptionError:
        #     option = Context.env.add_option(name, val)
    elif isinstance(key_value, list):
        patt = ListPatt(*[Parameter(patternize(k)) for k in key_value])
        option = Context.env.select_by_pattern(patt)
        if option is None:
            option = Context.env.add_option(patt, val)
        else:
            # option.nullify()
            option.assign(val)
    else:
        assert(0 == 1)
    return option.value


def augment_assign_fn(op: str):
    def aug_assign(key: Value, val: Function):
        initial = Context.env.deref(key.value)
        new = BuiltIns[op].call([initial, val])
        return assign_var(key, new)
    return aug_assign


# Operator('=', binop=1, static=True, associativity='right')
Operator(';',
         Function(AnyBinopPattern, lambda x, y: y),
         binop=1)
def assign_fn(fn: Function, patt: Value, block: Function, dot_option: Value) -> Value:
    patt = patt.value
    option = fn.select_by_pattern(patt) or fn.add_option(patt)
    option.assign(block.value)
    option.dot_option = dot_option.value
    return Value(None)
Operator(':',
         Function(ListPatt(AnyParam, PatternParam, AnyParam, BoolParam), assign_fn),
         binop=2, associativity='right')
def eval_assign_fn_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    blk_nd = rhs[0]
    if len(rhs) == 1 and isinstance(blk_nd, Block):
        block: Block = blk_nd
    else:
        return_statement = Statement([Token('return')] + rhs)  # noqa
        block = Block([return_statement])
    fn, patt, dot_option = read_option(lhs)
    return [fn, Value(patt), Value(FuncBlock(block)), Value(dot_option)]
Op[':'].eval_args = eval_assign_fn_args

Operator(':=', binop=2, associativity='right')
Operator('=',
         Function(AnyBinopPattern, assign_var,
                  {ListPatt(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(args)}),
         binop=2, associativity='right')
Op['='].eval_args = eval_set_args
Op[':='].fn = Op['='].fn
for op in ('+', '-', '*', '/', '//', '**', '%'):
    Operator(op+'=', Function(AnyBinopPattern, augment_assign_fn(op),
                              {ListPatt(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(list(args))}),
             binop=2, associativity='right')
    Op[op+'='].eval_args = eval_set_args
Op[':='].eval_args = eval_alias_args

def null_assign(key: Value, val: Function):
    try:
        initial = Context.env.deref(key.value)
        if initial.value is not None:
            return initial
    # WARNING: if the initial value calls a function in it's dereference, and that function contains
    # a NoMatchingOptionErrr, this will erroneously trigger
    except NoMatchingOptionError:
        pass
    return assign_var(key, val)
Operator('??=',
         Function(AnyBinopPattern, null_assign,
                  {ListPatt(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(list(args))}),
         binop=2, associativity='right')
Op['??='].eval_args = eval_set_args
Operator(',',
         Function(ListPatt(Parameter(Any, quantifier='+')), lambda *args: Value(tuple(args)),
                  {AnyParam: lambda x: Value((x,))}),
         binop=2, postfix=2, associativity='right')
def eval_tuple_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    left = expressionize(lhs).evaluate()
    if not rhs:
        return [left]
    right_expr = expressionize(rhs)
    if getattr(right_expr, 'op', None) == Op[',']:
        return [left, *eval_tuple_args(right_expr.lhs, right_expr.rhs)]
    return [left, right_expr.evaluate()]
Op[','].eval_args = eval_tuple_args
def eval_if_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    for i in reversed(range(len(rhs))):
        if rhs[i].source_text == 'else':
            condition = expressionize(rhs[:i])
            rhs = rhs[i + 1:]
            condition = condition.evaluate()
            break
    else:
        raise SyntaxErr(f"Line {Context.line}: If statement with no else clause")
    if BuiltIns['bool'].call([condition]).value:
        return [expressionize(lhs).evaluate(), Value(True), Value(None)]
    else:
        return [Value(None), Value(False), expressionize(rhs).evaluate()]
Operator('if',
         Function(ListPatt(AnyParam, AnyParam, AnyParam),
                  lambda consequent, condition, alt: consequent if condition.value else alt),
         binop=3, ternary='else')
Op['if'].eval_args = eval_if_args
Operator('??',
         Function(AnyBinopPattern, lambda x, y: x if x.value is not None else y),
         binop=4)
def eval_nullish_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    first = expressionize(lhs).evaluate()
    if first.not_null():
        return [first, Value(None)]
    return [first, expressionize(rhs).evaluate()]
Op['??'].eval_args = eval_nullish_args

def or_fn(*args: Function) -> Function:
    i = 0
    for i in range(len(args)-1):
        if BuiltIns['bool'].call([args[i]]).value:
            return args[i]
    return args[i]
Operator('or',
         Function(AnyPlusPattern, or_fn),
         binop=5)
def eval_or_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    condition = expressionize(lhs).evaluate()
    return [condition] if BuiltIns['bool'].call([condition]).value else [expressionize(rhs).evaluate()]
Op['or'].eval_args = eval_or_args

def and_fn(*args: Function) -> Function:
    i = 0
    for i in range(len(args)-1):
        if not BuiltIns['bool'].call([args[i]]).value:
            return args[i]
    return args[i]
Operator('and',
         Function(AnyPlusPattern, and_fn),
         binop=6)
def eval_and_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    condition = expressionize(lhs).evaluate()
    return [condition] if not BuiltIns['bool'].call([condition]).value else [expressionize(rhs).evaluate()]
Op['and'].eval_args = eval_and_args

Operator('not',
         Function(ListPatt(AnyParam), lambda a: Value(not BuiltIns['bool'].call([a]).value)),
         prefix=7)
Operator('in',
         Function(AnyBinopPattern, lambda a, b: Value(a in (opt.value for opt in b.options if hasattr(opt, 'value'))),
                  {ListPatt(AnyParam, ListParam): lambda a, b: Value(a in b.value)}),
         binop=8)
Operator('==',
         Function(AnyBinopPattern, lambda a, b: Value(a == b)),
         binop=9)
Operator('!=',
         Function(AnyBinopPattern, lambda a, b: Value(not BuiltIns['=='].call([a, b]).value)),
         binop=9)
Operator('~',
         Function(AnyBinopPattern, lambda a, b: Value(bool(patternize(b).match_score(a)))),
         binop=9, chainable=False)
Operator('!~',
         Function(AnyBinopPattern, lambda a, b: Value(not patternize(b).match_score(a))),
         binop=9, chainable=False)
Operator('|',
         Function(AnyBinopPattern, lambda a, b: Value(Union(patternize(a), patternize(b)))),
         binop=10)
Operator('<',
         Function(NormalBinopPattern, lambda a, b: Value(a.value < b.value)),
         binop=11, chainable=True)
Operator('>',
         Function(NormalBinopPattern, lambda a, b: Value(a.value > b.value)),
         binop=11, chainable=True)
Operator('<=',
         Function(AnyBinopPattern,
                  lambda a, b: Value(BuiltIns['<'].call([a, b]).value or BuiltIns['=='].call([a, b]).value)),
         binop=11, chainable=True)
Operator('>=',
         Function(AnyBinopPattern,
                  lambda a, b: Value(BuiltIns['>'].call([a, b]).value or BuiltIns['=='].call([a, b]).value)),
         binop=11, chainable=True)
Operator('+',
         Function(NormalBinopPattern, lambda a, b: Value(a.value + b.value),
                  options={ListPatt(AnyParam): lambda a: BuiltIns['number'].call([a])}),
         binop=12, prefix=14)
Operator('-',
         Function(NormalBinopPattern, lambda a, b: Value(a.value - b.value),
                  options={ListPatt(AnyParam): lambda a: Value(-BuiltIns['number'].call([a]).value)}),
         binop=12, chainable=False, prefix=14)
Operator('*',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value * b.value),
                  options={ListPatt(StringParam, IntegralParam): lambda a, b: Value(a.value * b.value)}),
         binop=13)
Operator('/',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value / b.value),
                  options={ListPatt(RationalParam, RationalParam): lambda a, b:
                  Value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))}),
         binop=13, chainable=False)
Operator('//',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(int(a.value // b.value))),
         binop=13, chainable=False)
Operator('%',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value % b.value)),
         binop=13, chainable=False)
Operator('**',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value ** b.value)),
         binop=14, chainable=False, associativity='right')
# Operator('?',
#          postfix=15, static=True)
def has_option(fn: Function, arg: Function = None) -> Value:
    if arg is None:
        fn, arg = Context.env, fn
        ascend = True
    else:
        ascend = False
    try:
        if arg.instanceof(BuiltIns['str']):
            return Value(fn.select_by_name(arg.value, ascend_env=ascend) is not None)
        elif arg.instanceof(BuiltIns['list']):
            # fn.select(arg.value, ascend_env=ascend)
            fn.select_and_bind(arg.value, ascend_env=ascend)
        else:
            # this is convenient but slightly dangerous because of possibility of list args
            # eg  if l is a list, and function foo has an option foo[l] = ...,
            # `foo has l` will confusingly return False (but `foo has [l]` => True)
            fn.select_and_bind([arg])
        return Value(True)
    except NoMatchingOptionError:
        return Value(False)
Operator('has',
         Function(ListPatt(AnyParam, ListParam), has_option,
                  {AnyBinopPattern: has_option,
                   ListPatt(NormalParam): has_option}),
         binop=15, prefix=15)
def add_guard_fn(fn: Function, guard: Function):
    patt = patternize(fn)
    patt.guard = guard
    return Value(patt)
def add_guard_expr(fn: Function, expr: Expression):
    patt = patternize(fn)
    patt.exprs.append(expr)
    return Value(patt)
Operator('&',
         Function(ListPatt(AnyParam, AnyParam), add_guard_expr,
                  {ListPatt(FunctionParam, AnyParam): add_guard_expr}),
         binop=15)
# def eval_patt_guard_args(lhs: list[Node], rhs: list[Node]) -> [Function, Expression]:
#     return [expressionize(lhs).evaluate(), expressionize(rhs)]
Op['&'].eval_args = lambda lhs, rhs: [expressionize(lhs).evaluate(), expressionize(rhs)]

def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    if len(rhs) != 1:
        raise SyntaxErr(f'Line {Context.line}: missing args')
    if rhs[0].type == TokenType.Name:
        right_arg = Value(rhs[0].source_text)
    else:
        right_arg = expressionize(rhs).evaluate()
    if not lhs:
        return [right_arg]
    if not right_arg.instanceof(BuiltIns['list']):
        return [expressionize(lhs).evaluate(), right_arg]
    args = right_arg
    if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
        name = lhs[-1].source_text
        a = expressionize(lhs[:-2]).evaluate()
        try:
            fn = a.deref(name, ascend_env=False)
        except NoMatchingOptionError:
            fn = Context.env.deref(name)
            args = Value([a] + args.value)
    else:
        fn = expressionize(lhs).evaluate()
    return [fn, args]
def dot_fn(a: Function, b: Value):
    match b.value:
        case str() as name:
            try:
                return a.deref(name, ascend_env=False)
            except NoMatchingOptionError:
                pass
            fn = Context.env.deref(name)
            # if not fn.instanceof(BuiltIns['fn']):
            #     raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
            # assert isinstance(fn.value, Function)
            return fn.call([a])
        case list() | tuple() as args:
            return a.call(list(args))
        case _ as val:
            print("WARNING: Line {Context.line}: "
                  "right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
            return a.call([b])
    # raise OperatorError(f"Line {Context.line}: "
    #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
def py_dot(a: Value, b: Value):
    obj = a.value
    match b.value:
        case str() as name:
            return piliize(getattr(obj, name))
        case list() | tuple() as args:
            return Value(a.value(*[arg.value for arg in args]))
Operator('.',
         Function(ListPatt(AnyParam, ListParam), dot_fn,
                  {AnyBinopPattern: dot_fn,
                   StringParam: lambda a: Context.env.deref(a.value),
                   ListParam: lambda ls: Context.env.call(ls.value, ascend=True),
                   ListPatt(Parameter(Prototype(BuiltIns['python_object'])),
                            Parameter(Union(Prototype(BuiltIns['str']), Prototype(BuiltIns['list'])))):
                       py_dot}),
         binop=16, prefix=16)
Operator('.?',
         Function(AnyBinopPattern, lambda a, b: BuiltIns['.'].call([a,b]) if BuiltIns['has'].call([a, b]).value else Value(None),
                  {ListPatt(StringParam): lambda a: BuiltIns['.'].call([a]) if BuiltIns['has'].call([a]).value else Value(None),}),
         binop=16, prefix=16)
# map-dot / swizzle operator
Operator('..',
         Function(ListPatt(ListParam, AnyParam), lambda ls, name: Value([dot_fn(el, name) for el in ls.value])),
         binop=16, prefix=16)
Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = eval_call_args


# pattern generator options for int, str, float, etc
def make_lambda_guard(type_name: str):
    if type_name == 'str':
        return lambda a, b: Value(Prototype(BuiltIns[type_name], guard=lambda x: Value(a.value <= len(x.value) <= b.value)))
    else:
        return lambda a, b: Value(Prototype(BuiltIns[type_name], guard=lambda x: Value(a.value <= x.value <= b.value)))

for type_name in ('num', 'ratio', 'float', 'int', 'str'):
    BuiltIns[type_name].add_option(ListPatt(NumericParam, NumericParam), make_lambda_guard(type_name))


BuiltIns['str'].add_option(ListPatt(StringParam),
                           lambda regex: Value(Prototype(BuiltIns['str'],
                                                         guard=lambda s: Value(bool(re.fullmatch(regex.value, s.value))))))
# BuiltIns['num'].add_option(ListPatt(NumericParam, NumericParam), lambda a, b: Value(Prototype(BuiltIns['num'], guard=lambda x: Value(a.value <= x.value <= b.value))))

# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
def number_guard(op_sym: str):
    # assert a.value == b.type
    return lambda t, n: Value(Prototype(t, guard=lambda x: Op[op_sym].fn.call([x, n])))

# generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
def string_guard(op_sym: str):
    # assert a.value == BuiltIns['str'] and b.type in (BuiltIns['int'], BuiltIns['float'])
    return lambda t, n: Value(Prototype(t, guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), n])))
    def guard(x, y):
        return Value(Prototype(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))
    # return guard
    return Value(Prototype(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))

def add_guards(op_sym: str):
    Op[op_sym].fn.add_option(ListPatt(Parameter(ValuePattern(BuiltIns['int'])), NumericParam),
                                number_guard(op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['float'])), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['ratio'])), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['str'])), NumericParam),
                                lambda a, b: string_guard(a, b, op_sym))


for op_sym in ('>', '<', '>=', '<='):
    for type_name in ('int', 'ratio', 'float', 'num'):
        Op[op_sym].fn.add_option(ListPatt(Parameter(ValuePattern(BuiltIns[type_name])), NumericParam),
                                 number_guard(op_sym))
    Op[op_sym].fn.add_option(ListPatt(Parameter(ValuePattern(BuiltIns['str'])), NumericParam),
                             string_guard(op_sym))
    # add_guards(op)
