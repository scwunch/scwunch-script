import contextlib
import re
from fractions import Fraction
from Syntax import Block, Statement
from Env import *
from DataStructures import *
from Expressions import read_option
from BuiltIns import *


# noinspection PyShadowingNames
def eval_set_args(lhs: list[Node], rhs: list[Node]) -> (tuple[PyValue[str], Record]
                                                        | tuple[Function, List, Record]
                                                        | tuple[Record, PyValue[str], Record]):
    rec = None
    name = None
    match lhs:
        case [Token(type=TokenType.Name, source_text=name)]:
            key = py_value(name)
        case [ListNode(nodes=statements)]:
            key = List(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
        case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
            key = py_value(name)
            rec = expressionize(fn_nodes).evaluate()
        case [*fn_nodes, Token(source_text='.' | '.['), ListNode() as list_node]:
            key = expressionize([list_node]).evaluate()
            rec = expressionize(fn_nodes).evaluate()
        case _:
            raise SyntaxErr(f'Line {Context.line}: '
                            f'Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
    match rhs:
        case []:
            raise SyntaxErr(f'Line {Context.line}: Missing right-hand side for assignment.')
        case [Block() as blk]:
            value = CodeBlock(blk).execute(fn=Function(name=name))
        case _:
            value = expressionize(rhs).evaluate()
    if isinstance(getattr(key, 'value', None), str) and isinstance(value, Function) and value.name is None:
        value.name = key.value
    if rec is None:
        return key, value
    return rec, key, value

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


def assign_option(fn: Function, key: PyValue[list], val: Record):
    params = (Parameter(ValueMatcher(rec)) for rec in key.value)
    fn.trait.assign_option(Pattern(*params), val)
    return val


def augment_assign_fn(op: str):
    def aug_assign(key: PyValue[str], val: Record):
        initial = Context.deref(key.value)
        new = BuiltIns[op].call(initial, val)
        # WARNING: this may potentially create a shadow variable
        return Context.env.assign(key.value, new)
    return aug_assign


# Operator('=', binop=1, static=True, associativity='right')
Operator(';',
         Function({AnyBinopPattern: lambda x, y: y}),
         binop=1)
def assign_fn(fn: Function, patt: Pattern, block: CodeBlock, dot_option: PyValue[bool]) -> PyValue:
    option = fn.trait.select_by_pattern(patt) or fn.trait.add_option(patt)
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
    return [fn, patt, CodeBlock(block), py_value(dot_option)]
Op[':'].eval_args = eval_assign_fn_args

Operator(':=', binop=2, associativity='right')
Operator('=',
         Function({Pattern(StringParam, AnyParam): lambda name, val: Context.env.assign(name, val),
                   Pattern(FunctionParam, NonStrSeqParam, AnyParam): assign_option,
                   Pattern(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val)}),
         binop=2, associativity='right')
Op['='].eval_args = eval_set_args
Op[':='].fn = Op['='].fn
for op in ('+', '-', '*', '/', '//', '**', '%'):
    Operator(op+'=', Function({AnyBinopPattern: augment_assign_fn(op),
                              Pattern(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(*args)}),
             binop=2, associativity='right')
    Op[op+'='].eval_args = eval_set_args
Op[':='].eval_args = eval_alias_args

def eval_null_assign_args(lhs: list[Node], rhs: list[Node]) -> tuple[Record, ...]:
    fn = None
    existing = None
    rec = None
    match lhs:
        case [Token(type=TokenType.Name, source_text=name)]:
            key = py_value(name)
            existing = Context.deref(name, None)
        case [ListNode(nodes=statements)]:
            key = List(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
            fn = Context.env.caller
        case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
            key = py_value(name)
            rec = expressionize(fn_nodes).evaluate()
            existing = rec.get(name, None)
        case [*fn_nodes, Token(source_text='.' | '.['), ListNode() as list_node]:
            key = expressionize([list_node]).evaluate()
            rec = expressionize(fn_nodes).evaluate()
            fn = rec
        case _:
            raise SyntaxErr(
                f'Line {Context.line}: Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
    if fn:
        opt, _ = fn.select(*key.value)
        if opt and opt.value and opt.value != BuiltIns['blank']:
            existing = opt.value
    if existing is None or existing == BuiltIns['blank']:
        value = expressionize(rhs).evaluate()
    else:
        return existing,
    if isinstance(getattr(key, 'value', None), str) and isinstance(value, Function) and value.name is None:
        value.name = key.value
    if rec is None:
        return key, value
    return rec, key, value

def null_assign(key: PyValue[str], val: Record):
    initial = Context.deref(key.value, None)
    if initial is not None:
        return initial
    return Context.env.assign(key.value, val)

def null_assign_option(fn: Function, key: PyValue[list], val: Record):
    params = (Parameter(ValueMatcher(rec)) for rec in key.value)
    fn.trait.assign_option(Pattern(*params), val)
    return val


Operator('??=',
         Function({Pattern(StringParam, AnyParam): lambda name, val: Context.env.assign(name, val),
                   Pattern(FunctionParam, NonStrSeqParam, AnyParam): assign_option,
                   Pattern(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val),
                   Pattern(AnyParam): lambda x: x}),
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
    if BuiltIns['bool'].call(condition).value:
        return [expressionize(lhs).evaluate(), py_value(True), py_value(None)]
    else:
        return [py_value(None), py_value(False), expressionize(rhs).evaluate()]
Operator('if',
         Function({Pattern(AnyParam, AnyParam, AnyParam):
                  lambda consequent, condition, alt: consequent if condition.value else alt}),
         binop=3, ternary='else')
Op['if'].eval_args = eval_if_args

def nullish_or(*args: Record):
    for arg in args[:-1]:
        if arg != BuiltIns['blank']:
            return arg
    return args[-1]


Operator('??',
         Function({Pattern(Parameter(AnyMatcher(), "", "+")): nullish_or}),
         binop=4)
def eval_nullish_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    first = expressionize(lhs).evaluate()
    if first != BuiltIns['blank']:
        return [first]
    return [expressionize(rhs).evaluate()]
Op['??'].eval_args = eval_nullish_args

def or_fn(*args: Record) -> Record:
    i = 0
    for i in range(len(args)-1):
        if BuiltIns['bool'].call(args[i]).value:
            return args[i]
    return args[i]
Operator('or',
         Function({AnyPlusPattern: or_fn}),
         binop=5)
def eval_or_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    condition = expressionize(lhs).evaluate()
    return [condition] if BuiltIns['bool'].call(condition).value else [expressionize(rhs).evaluate()]
Op['or'].eval_args = eval_or_args

def and_fn(*args: Record) -> Record:
    i = 0
    for i in range(len(args)-1):
        if not BuiltIns['bool'].call(args[i]).value:
            return args[i]
    return args[i]
Operator('and',
         Function({AnyPlusPattern: and_fn}),
         binop=6)
def eval_and_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    condition = expressionize(lhs).evaluate()
    return [condition] if not BuiltIns['bool'].call(condition).value else [expressionize(rhs).evaluate()]
Op['and'].eval_args = eval_and_args

Operator('not',
         Function({Pattern(AnyParam): lambda a: py_value(not BuiltIns['bool'].call(a).value)}),
         prefix=7)
Operator('in',
         Function({AnyBinopPattern: lambda a, b: py_value(a in (opt.value for opt in b.options if hasattr(opt, 'value'))),
                   Pattern(AnyParam, IterParam): lambda a, b: py_value(a in b.value)}),
         binop=8)
Operator('==',
         Function({AnyBinopPattern: lambda a, b: py_value(a == b)}),
         binop=9)
Operator('!=',
         Function({AnyBinopPattern: lambda a, b: py_value(not BuiltIns['=='].call(a, b).value)}),
         binop=9)
Operator('~',
         Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))}),
         binop=9, chainable=False)
Operator('!~',
         Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))}),
         binop=9, chainable=False)
Operator('is',
         Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))}),
         binop=9, chainable=False)
Operator('is not',
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
        if isinstance(param.matcher, UnionMatcher):
            matchers.extend(param.matcher.matchers)
        else:
            matchers.append(param.matcher)
    return Pattern(Parameter(UnionMatcher(*matchers)))


Operator('|',
         Function({AnyBinopPattern: union_patterns}),
         binop=10, chainable=True)
Operator('<',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value < b.value)}),
         binop=11, chainable=True)
Operator('>',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value > b.value)}),
         binop=11, chainable=True)
Operator('<=',
         Function({AnyBinopPattern:
                  lambda a, b: py_value(BuiltIns['<'].call(a, b).value or BuiltIns['=='].call(a, b).value)}),
         binop=11, chainable=True)
Operator('>=',
         Function({AnyBinopPattern:
                  lambda a, b: py_value(BuiltIns['>'].call(a, b).value or BuiltIns['=='].call(a, b).value)}),
         binop=11, chainable=True)
Operator('+',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value + b.value),
                   Pattern(AnyParam): lambda a: BuiltIns['num'].call(a),
                   Pattern(StringParam, StringParam): lambda a, b: py_value(a.value + b.value),
                   Pattern(ListParam, ListParam): lambda a, b: py_value(a.value + b.value),
                   Pattern(*(Parameter(TraitMatcher(TupTrait)),) * 2): lambda a, b: py_value(a.value + b.value),
                   # Pattern(SeqParam, SeqParam): lambda a, b: py_value(a.value + b.value)
                   }),
         binop=12, prefix=14)
Operator('-',
         Function({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
                  Pattern(AnyParam): lambda a: py_value(-BuiltIns['num'].call(a).value)}),
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
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value // b.value)}),
         binop=13, chainable=False)
Operator('%',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)}),
         binop=13, chainable=False)
Operator('**',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)}),
         binop=14, chainable=False, associativity='right')
Operator('^',
         Function({Pattern(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)}),
         binop=14, chainable=False, associativity='right')
Operator('?',
         Function({AnyParam: lambda p: union_patterns(p, BuiltIns['blank'])}),
         postfix=15, static=True)
def has_option(fn: Record, arg: Record = None) -> PyValue:
    if arg is None:
        fn, arg = None, fn

    match fn, arg:
        case None, PyValue(value=str() as name):
            return py_value(Context.deref(name, None) is not None)
        case None, _:
            raise TypeErr(f"Line {Context.line}: When used as a prefix, "
                          f"the right-hand term of the `has` operator must be a string, not {arg.table}")
        case Record(), PyValue(value=str() as name):
            return py_value(fn.get(name, None) is not None)
        # case Record(), List(records=args) | PyValue(value=tuple() as args):
        case Record(), PyValue(value=tuple() | list() as args):
            args = tuple(args)
            for t in fn.mro:
                option, _ = t.select_and_bind(args)
                if option:
                    return py_value(True)
            return py_value(False)
        case _:
            raise TypeErr(f"Line {Context.line}: "
                          f"The right-hand term of the `has` operator must be a string or sequence of arguments.")
Operator('has',
         Function({Pattern(AnyParam, NonStrSeqParam): has_option,
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
Operator('@',
         Function(),
         prefix=16)

def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
    if len(rhs) != 1:
        raise SyntaxErr(f'Line {Context.line}: missing args')
    if rhs[0].type == TokenType.Name:
        right_arg = py_value(rhs[0].source_text)
    else:
        right_arg = expressionize(rhs).evaluate()
    if not lhs:
        return [right_arg]
    if not TraitMatcher(SeqTrait).match_score(right_arg):
        return [expressionize(lhs).evaluate(), right_arg]
    args = right_arg
    if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
        name = lhs[-1].source_text
        a = expressionize(lhs[:-2]).evaluate()
        fn = a.get(name, None)
        if fn is None:
            fn = Context.deref(name, None)
            if fn is None:
                raise KeyErr(f"Line {Context.line}: {a.table} {a} has no slot '{name}' and no variable with that name "
                             f"found in current scope either.")
            args = List([a] + args.value)
    else:
        fn = expressionize(lhs).evaluate()
    return [fn, args]
def dot_fn(a: Record, b: Record):
    match b:
        case PyValue(value=str() as name):
            try:
                # return a.deref(name, ascend_env=False)
                return a.get(name)
            except SlotErr as e:
                pass
            fn = Context.deref(name)
            # if not fn.instanceof(BuiltIns['fn']):
            #     raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
            # assert isinstance(fn.value, Record)
            return fn.call(a)
        # case List(records=args) | PyValue(value=tuple() as args):
        case PyValue(value=tuple() | list() as args):
            return a.call(*args)
        case _:
            print(f"WARNING: Line {Context.line}: "
                  f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
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
         Function({Pattern(AnyParam, NonStrSeqParam): dot_fn,
                   AnyBinopPattern: dot_fn,
                   StringParam: lambda a: Context.deref(a.value),
                   Pattern(Parameter(TableMatcher(BuiltIns['PythonObject'])),
                           Parameter(UnionMatcher(TraitMatcher(FnTrait), TableMatcher(BuiltIns['Table'])))):
                       py_dot}),
         binop=16, prefix=16)
BuiltIns['call'] = BuiltIns['.']
Operator('.?',
         Function({AnyBinopPattern: lambda a, b: BuiltIns['.'].call(a,b) if BuiltIns['has'].call(a, b).value else py_value(None),
                  Pattern(StringParam): lambda a: BuiltIns['.'].call(a) if BuiltIns['has'].call(a).value else py_value(None),}),
         binop=16, prefix=16)
# map-dot / swizzle operator
Operator('..',
         Function({Pattern(SeqParam, StringParam): lambda ls, name: List([dot_fn(el, name) for el in ls.value]),
                   Pattern(NumericParam, NumericParam): lambda a, b: piliize(range(a.value, b.value))}),
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
    pass # BuiltIns[type_name].trait.add_option(Pattern(NumericParam, NumericParam), make_lambda_guard(type_name))


# BuiltIns['str'].trait.add_option(Pattern(StringParam),
#                            lambda regex: Pattern(Parameter(TraitMatcher(
#                                BuiltIns['str'], guard=lambda s: py_value(bool(re.fullmatch(regex.value, s.value)))))))
# BuiltIns['num'].trait.add_option(Pattern(NumericParam, NumericParam), lambda a, b: py_value(TableMatcher(BuiltIns['num'], guard=lambda x: py_value(a.value <= x.value <= b.value))))

# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
def number_guard(op_sym: str):
    # assert a.value == b.type
    return lambda t, n: Pattern(Parameter(TraitMatcher(t, guard=lambda x: Op[op_sym].fn.call(x, n))))

# generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
def string_guard(op_sym: str):
    # assert a.value == BuiltIns['str'] and b.type in (BuiltIns['int'], BuiltIns['float'])
    return lambda t, n: Pattern(Parameter(TraitMatcher(t, guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), n))))
    # def guard(x, y):
    #     return Pattern(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), b))))
    # # return guard
    # return Pattern(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), b))))

# def add_guards(op_sym: str):
#     Op[op_sym].fn.trait.add_option(Pattern(Parameter(ValueMatcher(BuiltIns['int'])), NumericParam),
#                                 number_guard(op_sym))
#     Op[op_sym].fn.assign_option(Pattern(Parameter(ValueMatcher(BuiltIns['float'])), NumericParam),
#                                 lambda a, b: number_guard(a, b, op_sym))
#     Op[op_sym].fn.assign_option(Pattern(Parameter(ValueMatcher(BuiltIns['ratio'])), NumericParam),
#                                 lambda a, b: number_guard(a, b, op_sym))
#     Op[op_sym].fn.assign_option(Pattern(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
#                                 lambda a, b: string_guard(a, b, op_sym))


for op_sym in ('>', '<', '>=', '<='):
    for type_name in ('int', 'ratio', 'float', 'num'):
        Op[op_sym].fn.trait.add_option(Pattern(Parameter(ValueMatcher(BuiltIns[type_name])), NumericParam),
                                 number_guard(op_sym))
    Op[op_sym].fn.trait.add_option(Pattern(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
                             string_guard(op_sym))
    # add_guards(op)
