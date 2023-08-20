import re
from fractions import Fraction
from Syntax import Block, Statement
from Env import *
from DataStructures import *
from Expressions import read_option
from BuiltIns import *


# noinspection PyShadowingNames
def eval_set_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    value = expressionize(rhs).evaluate()  # .clone()
    fn = None
    match lhs:
        case [Token(type=TokenType.Name, source_text=name)]:
            key = py_value(name)
        # case [Token(type=TokenType.Number|TokenType.String) as tok]:
        #     key = eval_token(tok)
        case [List(nodes=statements)]:
            key = piliize(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
        case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
            key = py_value(name)
            fn = expressionize(fn_nodes).evaluate()
        case [*fn_nodes, Token(source_text='.' | '.['), List() as list_node]:
            key = expressionize([list_node]).evaluate()
            fn = expressionize(fn_nodes).evaluate()
        case _:
            raise SyntaxErr(f'Line {Context.line}: Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
    if not value.name and hasattr(key, 'value') and isinstance(key.value, str):
        value.name = key.value
    if fn is None:
        return [key, value]
    return [fn, key, value]

def eval_alias_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    left, right = eval_set_args(lhs, [])[:-1], eval_set_args(rhs, [])
    right_key = right[-2]
    if len(right) == 3:
        right_fn = right[-3]
        ascend = False
    else:
        right_fn = Context.env
        ascend = True
    # right_fn = right[-3] if len(right)==3 else Context.env
    match right_key:  # right_key.value
        case str() as name:
            option = right_fn.select_by_name(name, ascend)
        case [PyValue(value=str() as name)]:
            option = right_fn.select_by_name(name, ascend)
        case list() as args:
            option, _ = right_fn.select_and_bind(args, ascend_env=ascend)
        case _:
            raise TypeErr(f"Line {Context.line}: Sorry, I'm not sure how to alias {right_key}")
    # option = right_fn.select_by_name(right_key.value, ascend)
    return [*left, option]


def assign_var(key: PyValue, val: Record):
    key_value = key.value
    if isinstance(key_value, str):
        name = key_value
        Context.env.names[name] = val
    #     if name in Context.env.named_options:
    #         option = Context.env.named_options[name]
    #         # option.nullify()
    #         option.assign(val)
    #     else:
    #         option = Context.env.add_option(name, val)
    #     # try:
    #     #     option = Context.env.select_by_name(name, ascend_env=False)
    #     #     option.nullify()
    #     #     option.assign(val)
    #     # except NoMatchingOptionError:
    #     #     option = Context.env.add_option(name, val)
    # elif isinstance(key_value, list):
    #     patt = Pattern(*[Parameter(patternize(k)) for k in key_value])  # noqa
    #     option = Context.env.select_by_pattern(patt)
    #     if option is None:
    #         option = Context.env.add_option(patt, val)
    #     else:
    #         # option.nullify()
    #         option.assign(val)
    else:
        assert(0 == 1)
    return val


def augment_assign_fn(op: str):
    def aug_assign(key: PyValue, val: Record):
        initial = Context.deref(key.value)
        new = BuiltIns[op].call([initial, val])
        return assign_var(key, new)
    return aug_assign


# Operator('=', binop=1, static=True, associativity='right')
Operator(';',
         Function({AnyBinopPattern: lambda x, y: y}),
         binop=1)
def assign_fn(fn: Record, patt: PyValue, block: CodeBlock, dot_option: PyValue[bool]) -> PyValue:
    patt = patt.value
    option = fn.select_by_pattern(patt) or fn.add_option(patt)
    option.assign(block)
    option.dot_option = dot_option.value
    return py_value(None)
Operator(':',
         Function({Pattern(AnyParam, PatternParam, AnyParam, BoolParam): assign_fn}),
         binop=2, associativity='right')
def eval_assign_fn_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    blk_nd = rhs[0]
    if len(rhs) == 1 and isinstance(blk_nd, Block):
        block: Block = blk_nd
    else:
        return_statement = Statement([Token('return')] + rhs)  # noqa
        block = Block([return_statement])
    fn, patt, dot_option = read_option(lhs)
    return [fn, patt, CodeBlock(block), dot_option]
Op[':'].eval_args = eval_assign_fn_args

Operator(':=', binop=2, associativity='right')
Operator('=',
         Function({AnyBinopPattern: assign_var,
                  Pattern(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(args)}),
         binop=2, associativity='right')
Op['='].eval_args = eval_set_args
Op[':='].fn = Op['='].fn
for op in ('+', '-', '*', '/', '//', '**', '%'):
    Operator(op+'=', Function({AnyBinopPattern: augment_assign_fn(op),
                              Pattern(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(list(args))}),
             binop=2, associativity='right')
    Op[op+'='].eval_args = eval_set_args
Op[':='].eval_args = eval_alias_args

def null_assign(key: PyValue, val: Record):
    try:
        initial = Context.deref(key.value)
        if initial.value is not None:
            return initial
    # WARNING: if the initial value calls a function in it's dereference, and that function contains
    # a NoMatchingOptionErrr, this will erroneously trigger
    except NoMatchingOptionError:
        pass
    return assign_var(key, val)
Operator('??=',
         Function({AnyBinopPattern: null_assign,
                  Pattern(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(list(args))}),
         binop=2, associativity='right')
Op['??='].eval_args = eval_set_args
Operator(',',
         Function({Pattern(Parameter(AnyMatcher(), quantifier='+')): lambda *args: py_value(tuple(args)),
                  AnyParam: lambda x: py_value((x,))}),
         binop=2, postfix=2, associativity='right')
def eval_tuple_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    left = expressionize(lhs).evaluate()
    if not rhs:
        return [left]
    right_expr = expressionize(rhs)
    if getattr(right_expr, 'op', None) == Op[',']:
        return [left, *eval_tuple_args(right_expr.lhs, right_expr.rhs)]
    return [left, right_expr.evaluate()]
Op[','].eval_args = eval_tuple_args
def eval_if_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    for i in reversed(range(len(rhs))):
        if rhs[i].source_text == 'else':
            condition = expressionize(rhs[:i])
            rhs = rhs[i + 1:]
            condition = condition.evaluate()
            break
    else:
        raise SyntaxErr(f"Line {Context.line}: If statement with no else clause")
    if BuiltIns['bool'].call([condition]).value:
        return [expressionize(lhs).evaluate(), py_value(True), py_value(None)]
    else:
        return [py_value(None), py_value(False), expressionize(rhs).evaluate()]
Operator('if',
         Function({Pattern(AnyParam, AnyParam, AnyParam):
                  lambda consequent, condition, alt: consequent if condition.value else alt}),
         binop=3, ternary='else')
Op['if'].eval_args = eval_if_args
Operator('??',
         Function({AnyBinopPattern: lambda x, y: x if x.value is not None else y}),
         binop=4)
def eval_nullish_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    first = expressionize(lhs).evaluate()
    if first.not_null():
        return [first, py_value(None)]
    return [first, expressionize(rhs).evaluate()]
Op['??'].eval_args = eval_nullish_args

def or_fn(*args: Record) -> Record:
    i = 0
    for i in range(len(args)-1):
        if BuiltIns['bool'].call([args[i]]).value:
            return args[i]
    return args[i]
Operator('or',
         Function({AnyPlusPattern: or_fn}),
         binop=5)
def eval_or_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    condition = expressionize(lhs).evaluate()
    return [condition] if BuiltIns['bool'].call([condition]).value else [expressionize(rhs).evaluate()]
Op['or'].eval_args = eval_or_args

def and_fn(*args: Record) -> Record:
    i = 0
    for i in range(len(args)-1):
        if not BuiltIns['bool'].call([args[i]]).value:
            return args[i]
    return args[i]
Operator('and',
         Function({AnyPlusPattern: and_fn}),
         binop=6)
def eval_and_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    condition = expressionize(lhs).evaluate()
    return [condition] if not BuiltIns['bool'].call([condition]).value else [expressionize(rhs).evaluate()]
Op['and'].eval_args = eval_and_args

Operator('not',
         Function({Pattern(AnyParam): lambda a: py_value(not BuiltIns['bool'].call([a]).value)}),
         prefix=7)
Operator('in',
         Function({AnyBinopPattern: lambda a, b: py_value(a in (opt.value for opt in b.options if hasattr(opt, 'value'))),
                  Pattern(AnyParam, ListParam): lambda a, b: py_value(a in b.value)}),
         binop=8)
Operator('==',
         Function({AnyBinopPattern: lambda a, b: py_value(a == b)}),
         binop=9)
Operator('!=',
         Function({AnyBinopPattern: lambda a, b: py_value(not BuiltIns['=='].call([a, b]).value)}),
         binop=9)
Operator('~',
         Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))}),
         binop=9, chainable=False)
Operator('!~',
         Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))}),
         binop=9, chainable=False)
# def union_patterns(*values_or_patterns: Record):
#     patts = map(patternize, values_or_patterns)
#     params = (param for patt in patts for param in patt.try_get_params())
#     try:
#         matchers = (m for p in params for m in p.try_get_matchers())
#         return Pattern(Parameter(Union(*matchers)))
#     except TypeError:
#         return Pattern(UnionParam(*params))
def union_patterns(*values_or_patterns: Record):
    patterns = map(patternize, values_or_patterns)
    params = []
    for patt in patterns:
        if len(patt.parameters) != 1:
            raise TypeErr(f"Line {Context.line}: Cannot get union of patterns with multiple parameters.")
        param = patt.parameters[0]
        if isinstance(param, UnionParam):
            params.extend(param.parameters)
        else:
            params.append(param)
    matchers = []
    for param in params:
        if param.quantifier or param.name is not None:
            return Pattern(UnionParam(*params))
        if isinstance(param.matcher, Union):
            matchers.extend(param.matcher.matchers)
        else:
            matchers.append(param.matcher)
    return Pattern(Parameter(Union(*matchers)))


Operator('|',
         Function({AnyBinopPattern: union_patterns}),
         binop=10)
Operator('<',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value < b.value)}),
         binop=11, chainable=True)
Operator('>',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value > b.value)}),
         binop=11, chainable=True)
Operator('<=',
         Function({AnyBinopPattern:
                  lambda a, b: py_value(BuiltIns['<'].call([a, b]).value or BuiltIns['=='].call([a, b]).value)}),
         binop=11, chainable=True)
Operator('>=',
         Function({AnyBinopPattern:
                  lambda a, b: py_value(BuiltIns['>'].call([a, b]).value or BuiltIns['=='].call([a, b]).value)}),
         binop=11, chainable=True)
Operator('+',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value + b.value),
                   Pattern(AnyParam): lambda a: BuiltIns['number'].call([a]),
                   Pattern(ListParam, ListParam): lambda a, b: py_value(a.value + b.value)}),
         binop=12, prefix=14)
Operator('-',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
                  Pattern(AnyParam): lambda a: py_value(-BuiltIns['number'].call([a]).value)}),
         binop=12, chainable=False, prefix=14)
Operator('*',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value * b.value),
                  Pattern(StringParam, IntegralParam): lambda a, b: py_value(a.value * b.value)}),
         binop=13)
Operator('/',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value / b.value),
                  Pattern(RationalParam, RationalParam): lambda a, b:
                  py_value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))}),
         binop=13, chainable=False)
Operator('//',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(int(a.value // b.value))}),
         binop=13, chainable=False)
Operator('%',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)}),
         binop=13, chainable=False)
Operator('**',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)}),
         binop=14, chainable=False, associativity='right')
# Operator('?',
#          postfix=15, static=True)
def has_option(fn: Record, arg: Record = None) -> PyValue:
    if arg is None:
        fn, arg = Context.env, fn
        ascend = True
    else:
        ascend = False
    try:
        if arg.instanceof(BuiltIns['str']) and isinstance(arg, PyValue):
            return py_value(fn.select_by_name(arg.value, ascend_env=ascend) is not None)
        elif arg.instanceof(BuiltIns['Table']) and isinstance(arg, PyValue):
            # fn.select(arg.value, ascend_env=ascend)
            fn.select_and_bind(arg.value, ascend_env=ascend)
        else:
            # this is convenient but slightly dangerous because of possibility of list args
            # eg  if l is a list, and function foo has an option foo[l] = ...,
            # `foo has l` will confusingly return False (but `foo has [l]` => True)
            fn.select_and_bind([arg])
        return py_value(True)
    except NoMatchingOptionError:
        return py_value(False)
Operator('has',
         Function({Pattern(AnyParam, ListParam): has_option,
                   AnyBinopPattern: has_option,
                   Pattern(NormalParam): has_option}),
         binop=15, prefix=15)
def add_guard_fn(fn: Record, guard: Function):
    patt = patternize(fn)
    patt.guard = guard
    return patt
def add_guard_expr(fn: Record, expr: Expression):
    patt = patternize(fn)
    patt.exprs.append(expr)  # noqa
    return patt
Operator('&',
         Function({Pattern(AnyParam, AnyParam): add_guard_expr,
                   Pattern(FunctionParam, AnyParam): add_guard_expr}),
         binop=15)
# def eval_patt_guard_args(lhs: list[Node], rhs: list[Node]) -> [Record, Expression]:
#     return [expressionize(lhs).evaluate(), expressionize(rhs)]
Op['&'].eval_args = lambda lhs, rhs: [expressionize(lhs).evaluate(), expressionize(rhs)]

def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    if len(rhs) != 1:
        raise SyntaxErr(f'Line {Context.line}: missing args')
    if rhs[0].type == TokenType.Name:
        right_arg = py_value(rhs[0].source_text)
    else:
        right_arg = expressionize(rhs).evaluate()
    if not lhs:
        return [right_arg]
    if not right_arg.instanceof(BuiltIns['Table']):
        return [expressionize(lhs).evaluate(), right_arg]
    args = right_arg
    if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
        name = lhs[-1].source_text
        a = expressionize(lhs[:-2]).evaluate()
        try:
            fn = a.deref(name, ascend_env=False)
        except NoMatchingOptionError:
            fn = Context.deref(name)
            args = piliize([a] + args.value)
    else:
        fn = expressionize(lhs).evaluate()
    return [fn, args]
def dot_fn(a: Record, b: PyValue):
    match b.value:
        case str() as name:
            try:
                # return a.deref(name, ascend_env=False)
                return a.get(name)
            except SlotErr as e:
                pass
            fn = Context.deref(name)
            # if not fn.instanceof(BuiltIns['fn']):
            #     raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
            # assert isinstance(fn.value, Record)
            return fn.call([a])
        case list() | tuple() as args:
            return a.call(*args)
        case _:
            print("WARNING: Line {Context.line}: "
                  "right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
            return a.call(b)
    # raise OperatorError(f"Line {Context.line}: "
    #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
def py_dot(a: PyObj, b: PyValue):
    obj = a.obj
    match b.value:
        case str() as name:
            return piliize(getattr(obj, name))
        case list() | tuple() as args:
            return piliize(a.obj(*[arg.value for arg in args]))
Operator('.',
         Function({Pattern(AnyParam, ListParam): dot_fn,
                   AnyBinopPattern: dot_fn,
                   StringParam: lambda a: Context.deref(a.value),
                   # ListParam: lambda ls: Context.env.call(ls.value, ascend=True),  # this is meant to execute options of self without the self keyword... I'll probably delte this
                   Pattern(Parameter(TableMatcher(BuiltIns['python_object'])),
                            Parameter(Union(TableMatcher(BuiltIns['str']), TableMatcher(BuiltIns['Table'])))):
                       py_dot}),
         binop=16, prefix=16)
Operator('.?',
         Function({AnyBinopPattern: lambda a, b: BuiltIns['.'].call([a,b]) if BuiltIns['has'].call([a, b]).value else py_value(None),
                  Pattern(StringParam): lambda a: BuiltIns['.'].call([a]) if BuiltIns['has'].call([a]).value else py_value(None),}),
         binop=16, prefix=16)
# map-dot / swizzle operator
Operator('..',
         Function({Pattern(ListParam, AnyParam): lambda ls, name: piliize([dot_fn(el, name) for el in ls.value])}),
         binop=16, prefix=16)
Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = eval_call_args


# pattern generator options for int, str, float, etc
def make_lambda_guard(type_name: str):
    if type_name == 'str':
        return lambda a, b: Pattern(Parameter(TableMatcher(BuiltIns[type_name], guard=lambda x: py_value(a.value <= len(x.value) <= b.value))))
    else:
        return lambda a, b: Pattern(Parameter(TableMatcher(BuiltIns[type_name], guard=lambda x: py_value(a.value <= x.value <= b.value))))


for type_name in ('num', 'ratio', 'float', 'int', 'str'):
    if type_name == 'int':
        pass
    BuiltIns[type_name].add_option(Pattern(NumericParam, NumericParam), make_lambda_guard(type_name))


BuiltIns['str'].add_option(Pattern(StringParam),
                           lambda regex: Pattern(Parameter(TableMatcher(
                               BuiltIns['str'], guard=lambda s: py_value(bool(re.fullmatch(regex.value, s.value)))))))
# BuiltIns['num'].add_option(Pattern(NumericParam, NumericParam), lambda a, b: py_value(TableMatcher(BuiltIns['num'], guard=lambda x: py_value(a.value <= x.value <= b.value))))

# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
def number_guard(op_sym: str):
    # assert a.value == b.type
    return lambda t, n: Pattern(Parameter(TableMatcher(t, guard=lambda x: Op[op_sym].fn.call([x, n]))))

# generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
def string_guard(op_sym: str):
    # assert a.value == BuiltIns['str'] and b.type in (BuiltIns['int'], BuiltIns['float'])
    return lambda t, n: Pattern(Parameter(TableMatcher(t, guard=lambda s: Op[op_sym].fn.call([py_value(len(s.value)), n]))))
    # def guard(x, y):
    #     return Pattern(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call([py_value(len(s.value)), b]))))
    # # return guard
    # return Pattern(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call([py_value(len(s.value)), b]))))

# def add_guards(op_sym: str):
#     Op[op_sym].fn.add_option(Pattern(Parameter(ValueMatcher(BuiltIns['int'])), NumericParam),
#                                 number_guard(op_sym))
#     Op[op_sym].fn.assign_option(Pattern(Parameter(ValueMatcher(BuiltIns['float'])), NumericParam),
#                                 lambda a, b: number_guard(a, b, op_sym))
#     Op[op_sym].fn.assign_option(Pattern(Parameter(ValueMatcher(BuiltIns['ratio'])), NumericParam),
#                                 lambda a, b: number_guard(a, b, op_sym))
#     Op[op_sym].fn.assign_option(Pattern(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
#                                 lambda a, b: string_guard(a, b, op_sym))


for op_sym in ('>', '<', '>=', '<='):
    for type_name in ('int', 'ratio', 'float', 'num'):
        Op[op_sym].fn.add_option(Pattern(Parameter(ValueMatcher(BuiltIns[type_name])), NumericParam),
                                 number_guard(op_sym))
    Op[op_sym].fn.add_option(Pattern(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
                             string_guard(op_sym))
    # add_guards(op)