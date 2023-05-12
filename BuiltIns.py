import re
from fractions import Fraction
from Syntax import Node, Token, List, TokenType
from Env import *
from DataStructures import *
from Expressions import expressionize, read_number

BuiltIns['_base_prototype'] = Function(name='base_prototype', type=True)  # noqa
BuiltIns['_base_prototype'].type = None
MetaType = Function(name="Type", type=BuiltIns['_base_prototype'])
BuiltIns['BasicType'] = MetaType
BuiltIns['none'] = Function(name='none', type=MetaType)
BuiltIns['num'] = Function(name="num", type=MetaType)
BuiltIns['float'] = Function(name='float', type=BuiltIns['num'])
BuiltIns['ratio'] = Function(name='ratio', type=BuiltIns['num'])
BuiltIns['int'] = Function(name='int', type=BuiltIns['ratio'])
BuiltIns['bool'] = Function(name='bool', type=BuiltIns['int'])
BuiltIns['str'] = Function(name='str', type=MetaType)
BuiltIns['list'] = Function(name='list', type=MetaType)
BuiltIns['pattern'] = Function(name='pattern', type=MetaType)
BuiltIns['value_pattern'] = Function(name='value_pattern', type=BuiltIns['pattern'])
BuiltIns['union'] = Function(name='union', type=BuiltIns['pattern'])
BuiltIns['type_pattern'] = Function(name='type_pattern', type=BuiltIns['pattern'])
BuiltIns['parameters'] = Function(name='parameters', type=BuiltIns['pattern'])
BuiltIns['fn'] = Function(name='fn', type=BuiltIns['_base_prototype'])
TypeMap.update({
    type(None): BuiltIns['none'],
    bool: BuiltIns['bool'],
    int: BuiltIns['int'],
    Fraction: BuiltIns['ratio'],
    float: BuiltIns['float'],
    str: BuiltIns['str'],
    list: BuiltIns['list'],
    Pattern: BuiltIns['pattern'],
    ValuePattern: BuiltIns['value_pattern'],
    Prototype: BuiltIns['type_pattern'],
    Union: BuiltIns['union'],
    ListPatt: BuiltIns['parameters']
})
BuiltIns['any'] = Value(None)
BuiltIns['any'].value, BuiltIns['any'].type = Any, BuiltIns['pattern']
TypeMap[AnyPattern] = BuiltIns['any']

NoneParam = Parameter(Prototype(BuiltIns["none"]))
BoolParam = Parameter(Prototype(BuiltIns["bool"]))
IntegralParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"])))
FloatParam = Parameter(Prototype(BuiltIns["float"]))
RationalParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"])))
NumericParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"]), Prototype(BuiltIns["float"])))
StringParam = Parameter(Prototype(BuiltIns["str"]))
# NormalParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"]),
#                               Prototype(BuiltIns["float"]), Prototype(BuiltIns["str"])))
NormalParam = Parameter(Union(Prototype(BuiltIns['num']), Prototype(BuiltIns['str'])))
ListParam = Param = Parameter(Prototype(BuiltIns["list"]))
# TypeParam = Parameter(Prototype(BuiltIns["Type"]))
PatternParam = Parameter(Prototype(BuiltIns["pattern"]))
FunctionParam = Parameter(Prototype(BuiltIns["fn"]))
AnyParam = Parameter(Any)
NormalBinopPattern = ListPatt(NormalParam, NormalParam)
AnyBinopPattern = ListPatt(AnyParam, AnyParam)
AnyListPatt = ListPatt(Parameter(Any, quantifier="*"))

PositiveIntParam = Parameter(Prototype(BuiltIns["int"], guard=lambda x: Value(x.value > 0)))
NegativeIntParam = Parameter(Prototype(BuiltIns["int"], guard=lambda x: Value(x.value < 0)))
NonZeroIntParam = Parameter(Prototype(BuiltIns["int"], guard=lambda x: Value(x.value != 0)))
OneIndexList = Parameter(Prototype(BuiltIns['list'], guard=lambda x: Value(len(x.value) == 1 and
                                                                           NonZeroIntParam.match_score(x.value[0]))))

def key_to_param_set(key: Value) -> ListPatt:
    if hasattr(key, 'value') and isinstance(key.value, list):
        vals = key.value
    else:
        vals = [key]
    params = (Parameter(ValuePattern(pval, pval.value if isinstance(pval.value, str) else None)) for pval in vals)
    return ListPatt(*params)
BuiltIns['set'] = Function(ListPatt(AnyParam, AnyParam, AnyParam),
                           lambda fn, key, val: fn.assign_option(key_to_param_set(key), val).resolution)

BuiltIns['bool'].add_option(ListPatt(AnyParam), lambda x: Value(bool(x.value)))
BuiltIns['number'] = Function(ListPatt(BoolParam), lambda x: Value(int(x.value)),
                              options={ListPatt(NumericParam): Value.clone,
                                       ListPatt(StringParam): lambda x: Value(read_number(x.value))},
                              name='number')
BuiltIns['integer'] = Function(ListPatt(NormalParam), lambda x: Value(int(BuiltIns['number'].call([x]).value)),
                               name='integer')
BuiltIns['rational'] = Function(ListPatt(NormalParam), lambda x: Value(Fraction(BuiltIns['number'].call([x]).value)),
                                name='rational')
# BuiltIns['float'] = Function(ListPatt(NormalParam), lambda x: Value(float(BuiltIns['number'].call([x]).value)))
BuiltIns['string'] = Function(ListPatt(AnyParam), lambda x: x.to_string(), name='string')
# BuiltIns['string'].add_option(ListPatt(ListParam), lambda l: Value(str(l.value[1:])))
# BuiltIns['string'].add_option(ListPatt(NumberParam),
#                               lambda n: Value('-' * (n.value < 0) +
#                                               base(abs(n.value), 10, 6, string=True, recurring=False)))
# BuiltIns['string'].add_option(ListPatt(Parameter(Prototype(BuiltIns["Type"]))), lambda t: Value(t.value.name))

BuiltIns['type'] = Function(ListPatt(AnyParam), lambda v: v.type)

BuiltIns['len'] = Function(ListPatt(StringParam), lambda s: Value(len(s.value)))
BuiltIns['len'].add_option(ListPatt(FunctionParam), lambda f: Value(len(f.options)))
BuiltIns['len'].add_option(ListPatt(ListParam), lambda l: Value(len(l.value)))
BuiltIns['len'].add_option(ListPatt(Parameter(Prototype(BuiltIns["pattern"]))), lambda p: Value(len(p.value)))

# BuiltIns['prototype'] = Function(ListPatt(FunctionParam), lambda f: Value(f.value.prototype))

# BuiltIns['contains'] = Function(ListPatt(FunctionParam, AnyParam),
#                                 lambda a, b: Value(b in (opt.value for opt in a.options)))
# BuiltIns['List'] = Function(ListPatt(Parameter(Any, quantifier='*')),
#                             lambda *vals: Value(list(*vals)))
BuiltIns['len'].add_option(ListPatt(Parameter(Prototype(BuiltIns['list']))), lambda l: Value(len(l.value)))

def list_get(scope: Function, *args: Value):
    fn = scope.type
    if len(args) == 1:
        if abs(args[0].value) > BuiltIns['len'].call([fn]).value:
            raise IndexError(f'Line {Context.line}: Index {args[0]} out of range')
        length = BuiltIns['len'].call([fn])
        index = args[0].value
        if BuiltIns['>'].call([Value(abs(index)), length]).value:
            raise IndexError(f'Line {Context.line}: Index {args[0]} out of range')
        index -= index > 0
        return fn.value[index]
    else:
        raise NotImplemented
def list_set(ls: Value, index: Value, val: Function):
    i = index.value[0].value
    i -= i > 0
    if i == len(ls.value):
        ls.value.append(val)
    else:
        ls.value[i] = val
    return val
BuiltIns['list'].add_option(ListPatt(PositiveIntParam), FuncBlock(list_get))
BuiltIns['list'].add_option(ListPatt(NegativeIntParam), FuncBlock(list_get))
BuiltIns['set'].add_option(ListPatt(ListParam, OneIndexList, AnyParam), list_set)
BuiltIns['push'] = Function(ListPatt(Parameter(Prototype(BuiltIns['list'])), AnyParam),
                            lambda fn, val: fn.value.append(val) or fn)
BuiltIns['join'] = Function(ListPatt(ListParam, StringParam),
                            lambda ls, sep: Value(sep.value.join(BuiltIns['string'].call([item]).value for item in ls.value)))
BuiltIns['split'] = Function(ListPatt(StringParam, StringParam), lambda txt, sep: Value(txt.value.split(sep.value)))

def convert(name: str) -> Function:
    o = object()
    # py_fn = getattr(__builtins__, name, o)
    py_fn = __builtins__.get(name, o)
    if py_fn is o:
        raise SyntaxErr(f"Name '{name}' not found.")
    # Context.root.add_option(ListPatt(Parameter(name)), lambda *args: Value(py_fn((arg.value for arg in args))))
    # def lambda_fn(*args):
    #     arg_list = list(arg.value for arg in args)
    #     return Value(py_fn(*arg_list))
    # return Function(AnyListPatt, lambda_fn)
    return Function(AnyListPatt, lambda *args:
        Value(py_fn(*(arg.value for arg in args))))

BuiltIns['python'] = Function(ListPatt(StringParam), lambda n: convert(n.value))

#############################################################################################
def eval_set_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    value = expressionize(rhs).evaluate()  # .clone()
    fn = None
    match lhs:
        case [Token(type=TokenType.Name, source_text=name)]:
            key = Value(name)
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

def assign_var(key: Value, val: Function):
    name = key.value
    assert isinstance(name, str)
    try:
        option = Context.env.select_by_name(name)
        option.nullify()
        option.assign(val)
    except NoMatchingOptionError:
        option = Context.env.add_option(name, val)
    return option.value


def augment_assign_fn(op: str):
    def aug_assign(key: Value, val: Function):
        initial = Context.env.deref(key.value)
        new = BuiltIns[op].call([initial, val])
        return assign_var(key, new)
    return aug_assign


# Operator('=', binop=1, static=True, associativity='right')
Operator(':', binop=1, static=True, associativity='right')
Operator(':=', binop=1, static=True, associativity='right')
Operator('=',
         Function(AnyBinopPattern, assign_var,
                  {ListPatt(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(list(args))}),
         binop=1, associativity='right')
Op['='].eval_args = eval_set_args
for op in ('+', '-', '*', '/', '//', '**', '%'):
    Operator(op+'=', Function(AnyBinopPattern, augment_assign_fn(op),
                              {ListPatt(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(list(args))}),
             binop=1, associativity='right')
    Op[op+'='].eval_args = eval_set_args

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
         binop=1, associativity='right')
Op['??='].eval_args = eval_set_args
Operator('if',
         Function(ListPatt(AnyParam), lambda x: x),
         binop=2, static=True, ternary='else')
Operator('??',
         binop=3, static=True)
Operator('or',
         binop=4, static=True)
Operator('and',
         binop=5, static=True)
Operator('|',
         Function(ListPatt(PatternParam, PatternParam), lambda a, b: Value(Union(patternize(a), patternize(b)))),
         binop=4)
Op['|'].fn.add_option(AnyBinopPattern, lambda a, b: Value(Union(patternize(a), patternize(b))))
Operator('not',
         Function(ListPatt(AnyParam), lambda a: Value(not BuiltIns['bool'].call([a]).value)),
         prefix=6)
Operator('in',
         Function(AnyBinopPattern, lambda a, b: Value(a in (opt.value for opt in b.options if hasattr(opt, 'value'))),
                  {ListPatt(AnyParam, ListParam): lambda a, b: Value(a in b.value)}),
         binop=7)
Operator('==',
         Function(AnyBinopPattern, lambda a, b: Value(a == b)),
         binop=8)
Operator('!=',
         Function(AnyBinopPattern, lambda a, b: Value(not BuiltIns['=='].call([a, b]).value)),
         binop=9)
Operator('<',
         Function(NormalBinopPattern, lambda a, b: Value(a.value < b.value)),
         binop=10, chainable=True)
Operator('>',
         Function(NormalBinopPattern, lambda a, b: Value(a.value > b.value)),
         binop=10, chainable=True)
Operator('<=',
         Function(AnyBinopPattern,
                  lambda a, b: Value(BuiltIns['<'].call([a, b]).value or BuiltIns['=='].call([a, b]).value)),
         binop=10, chainable=True)
Operator('>=',
         Function(AnyBinopPattern,
                  lambda a, b: Value(BuiltIns['>'].call([a, b]).value or BuiltIns['=='].call([a, b]).value)),
         binop=10, chainable=True)
Operator('~',
         Function(AnyBinopPattern, lambda a, b: Value(bool(patternize(b).match_score(a)))),
         binop=9, chainable=False)
Operator('!~',
         Function(AnyBinopPattern, lambda a, b: Value(not patternize(b).match_score(a))),
         binop=9, chainable=False)
Operator('+',
         Function(NormalBinopPattern, lambda a, b: Value(a.value + b.value),
                  options={ListPatt(AnyParam): lambda a: BuiltIns['number'].call([a])}),
         binop=11, prefix=13)
Operator('-',
         Function(NormalBinopPattern, lambda a, b: Value(a.value - b.value),
                  options={ListPatt(AnyParam): lambda a: Value(-BuiltIns['number'].call([a]).value)}),
         binop=11, chainable=False, prefix=13)
Operator('*',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value * b.value),
                  options={ListPatt(StringParam, IntegralParam): lambda a, b: Value(a.value * b.value)}),
         binop=12)
Operator('/',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value / b.value),
                  options={ListPatt(RationalParam, RationalParam): lambda a, b:
                  Value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))}),
         binop=12, chainable=False)
Operator('//',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(int(a.value // b.value))),
         binop=12, chainable=False)
Operator('%',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value % b.value)),
         binop=12, chainable=False)
Operator('**',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value ** b.value)),
         binop=13, chainable=False, associativity='right')
Operator('?',
         postfix=14, static=True)
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
            # fn.select([arg])
            fn.select_and_bind([arg])
        return Value(True)
    except NoMatchingOptionError:
        return Value(False)
Operator('has',
         Function(ListPatt(AnyParam, ListParam), has_option,
                  {AnyBinopPattern: has_option,
                   ListPatt(NormalParam): has_option}),
         binop=14, prefix=14)

def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Function]:
    if len(rhs) != 1:
        raise SyntaxErr(f'Line {Context.line}: missing args')
    if rhs[0].type == TokenType.Name:
        right_arg = Value(rhs[0].source_text)
    else:
        right_arg = expressionize(rhs).evaluate()
    if not right_arg.instanceof(BuiltIns['list']):
        return [expressionize(lhs).evaluate(), right_arg]
    args = right_arg
    if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..'):
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
    name = b.value
    try:
        return a.deref(name, ascend_env=False)
    except NoMatchingOptionError:
        pass
    fn = Context.env.deref(name)
    if not fn.instanceof(BuiltIns['fn']):
        raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
    # assert isinstance(fn.value, Function)
    return fn.call([a])

Operator('.',
         Function(ListPatt(AnyParam, ListParam), lambda a, b: a.call(b.value),
                  {AnyBinopPattern: dot_fn,
                   ListPatt(StringParam): lambda a: dot_fn(Context.env, a)}),
         binop=15, prefix=15)
Operator('.?',
         Function(AnyBinopPattern, lambda a, b: BuiltIns['.'].call([a,b]) if BuiltIns['has'].call([a, b]).value else Value(None),
                  {ListPatt(StringParam): lambda a: BuiltIns['.'].call([a]) if BuiltIns['has'].call([a]).value else Value(None),}),
         binop=15, prefix=15)
# map-dot / swizzle operator
Operator('..',
         Function(ListPatt(ListParam, AnyParam), lambda ls, name: Value([dot_fn(el, name) for el in ls.value])),
         binop=15, prefix=15)
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
