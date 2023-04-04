import re
from fractions import Fraction
from Syntax import Node, TokenType
from Env import *
from DataStructures import *
from Expressions import expressionize, read_number

BuiltIns['_base_prototype'] = Function(name='any', type=True)  # noqa
BuiltIns['_base_prototype'].type = None
MetaType = Function(name="Type", type=BuiltIns['_base_prototype'])
BuiltIns['BasicType'] = MetaType
BuiltIns['none'] = Function(name='none', type=MetaType)
BuiltIns['numeric'] = Function(name="numeric", type=MetaType)
BuiltIns['float'] = Function(name='float', type=BuiltIns['numeric'])
BuiltIns['ratio'] = Function(name='ratio', type=BuiltIns['numeric'])
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

NoneParam = Parameter(Prototype(BuiltIns["none"]))
BoolParam = Parameter(Prototype(BuiltIns["bool"]))
IntegralParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"])))
FloatParam = Parameter(Prototype(BuiltIns["float"]))
RationalParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"])))
NumericParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"]), Prototype(BuiltIns["float"])))
StringParam = Parameter(Prototype(BuiltIns["str"]))
# NormalParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"]),
#                               Prototype(BuiltIns["float"]), Prototype(BuiltIns["str"])))
NormalParam = Parameter(Union(Prototype(BuiltIns['numeric']), Prototype(BuiltIns['str'])))
ListParam = Param = Parameter(Prototype(BuiltIns["list"]))
# TypeParam = Parameter(Prototype(BuiltIns["Type"]))
PatternParam = Parameter(Prototype(BuiltIns["pattern"]))
FunctionParam = Parameter(Prototype(BuiltIns["fn"]))
AnyParam = Parameter(Any)
NormalBinopPattern = ListPatt(NormalParam, NormalParam)
AnyBinopPattern = ListPatt(AnyParam, AnyParam)

NegativeRationalParam = Parameter(Prototype(BuiltIns["ratio"], guard=lambda x: Value(x.value < 0)))
# BuiltIns['numeric'] = Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"]), Prototype(BuiltIns["ratio"]), Prototype(BuiltIns["float"]))
BuiltIns['bool'].add_option(ListPatt(AnyParam), lambda x: Value(bool(x.value)))
BuiltIns['number'] = Function(ListPatt(BoolParam), lambda x: Value(int(x.value)),
                              options={ListPatt(NumericParam): Value.clone,
                                       ListPatt(StringParam): lambda x: Value(read_number(x.value))})
BuiltIns['integer'] = Function(ListPatt(NormalParam), lambda x: Value(int(BuiltIns['number'].call([x]).value)))
BuiltIns['rational'] = Function(ListPatt(NormalParam), lambda x: Value(Fraction(BuiltIns['number'].call([x]).value)))
# BuiltIns['float'] = Function(ListPatt(NormalParam), lambda x: Value(float(BuiltIns['number'].call([x]).value)))
BuiltIns['string'] = Function(ListPatt(AnyParam), lambda x: x.to_string())
# BuiltIns['string'].add_option(ListPatt(NumberParam),
#                               lambda n: Value('-' * (n.value < 0) +
#                                               base(abs(n.value), 10, 6, string=True, recurring=False)))
# BuiltIns['string'].add_option(ListPatt(Parameter(Prototype(BuiltIns["Type"]))), lambda t: Value(t.value.name))

BuiltIns['type'] = Function(ListPatt(AnyParam), lambda v: v.type)

BuiltIns['len'] = Function(ListPatt(StringParam), lambda s: Value(len(s.value)))
BuiltIns['len'].add_option(ListPatt(FunctionParam), lambda f: Value(len(f.value.options)))
BuiltIns['len'].add_option(ListPatt(ListParam), lambda l: Value(len(l.value)))
BuiltIns['len'].add_option(ListPatt(Parameter(Prototype(BuiltIns["pattern"]))), lambda p: Value(len(p.value)))

# BuiltIns['prototype'] = Function(ListPatt(FunctionParam), lambda f: Value(f.value.prototype))

BuiltIns['contains'] = Function(ListPatt(FunctionParam, AnyParam),
                                lambda a, b: Value(b in (opt.value for opt in a.options)))
BuiltIns['List'] = Function(ListPatt(Parameter(Any, quantifier='*')),
                            lambda *vals: Value(list(*vals)))
BuiltIns['len'].add_option(ListPatt(Parameter(Prototype(BuiltIns['list']))), lambda l: Value(len(l.value)-1))
def neg_index(scope: Function, *args: Value):
    fn = scope.type
    if len(args) == 1:
        length = BuiltIns['len'].call([Value(fn)]).value
        index = Value(length + 1 + args[0].value)
        return fn.call([index])
    else:
        raise NotImplemented
BuiltIns['List'].add_option(ListPatt(NegativeRationalParam), FuncBlock(neg_index))
BuiltIns['push'] = Function(ListPatt(Parameter(Prototype(BuiltIns['List'])), AnyParam),
                            lambda fn, val: fn.value.add_option(ValuePattern(Value(len(fn.value.array))), val) and fn)

#############################################################################################

Operator('=', binop=1, static=True, associativity='right')
Operator(':', binop=1, static=True, associativity='right')
Operator(':=', binop=1, static=True, associativity='right')
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
         Function(AnyBinopPattern, lambda a, b: BuiltIns['contains'].call([b, a])),
         binop=7)
Operator('==',
         Function(AnyBinopPattern, lambda a, b: Value(a == b)),
         binop=8)
Operator('!=',
         Function(AnyBinopPattern, lambda a, b: Value(a != b)),
         binop=9)
Operator('<',
         Function(NormalBinopPattern, lambda a, b: Value(a.value < b.value)),
         binop=10, chainable=True)
Operator('>',
         Function(NormalBinopPattern, lambda a, b: Value(a.value > b.value)),
         binop=10, chainable=True)
Operator('<=',
         Function(NormalBinopPattern, lambda a, b: Value(a.value <= b.value)),
         binop=10, chainable=True)
Operator('>=',
         Function(NormalBinopPattern, lambda a, b: Value(a.value >= b.value)),
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
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value // b.value)),
         binop=12, chainable=False)
Operator('%',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value % b.value)),
         binop=12, chainable=False)
Operator('**',
         Function(ListPatt(NumericParam, NumericParam), lambda a, b: Value(a.value ** b.value)),
         binop=13, chainable=False, associativity='right')
Operator('?',
         postfix=14, static=True)
# def dot_call(a: Value, b: Value, c: Value = None):
#     name = b.value
#     args: list[Value] = []
#     if c:
#         assert isinstance(c.value, list)
#         args = c.value
#     try:
#         assert a.type == BuiltIns['fn'] and isinstance(a.value, Function)
#         val = a.value.deref(name, ascend_env=False)
#     except (AssertionError, NoMatchingOptionError):
#         fn = Context.env.deref(name)
#         if fn.type != BuiltIns['fn']:
#             raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
#         assert isinstance(fn.value, Function)
#         args = [a] + args
#         return fn.call(args)
#     if c:
#         fn = val.value
#         if not isinstance(fn, Function):
#             raise OperatorError(f"Line {Context.line}: {val.type.value} '{name}' is not function.")
#         return fn.call(args)
#     else:
#         return val

def dot_fn(a: Function, b: Value):
    name = b.value
    try:
        # assert a.type == BuiltIns['fn'] and isinstance(a.value, Function)
        return a.deref(name, ascend_env=False)
    except (AssertionError, NoMatchingOptionError):
        fn = Context.env.deref(name)
        if not fn.instanceof(BuiltIns['fn']):
            raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
        # assert isinstance(fn.value, Function)
        return fn.call([a])

Operator('.',
         Function(ListPatt(AnyParam, StringParam), dot_fn,
                  options={ListPatt(StringParam): lambda a: dot_fn(Value(Context.env), a)}),
                  # options={ListPatt(AnyParam,
                  #                   Parameter(Prototype(BuiltIns["Name"])),
                  #                   Parameter(Prototype(BuiltIns["List"]), quantifier="?")
                  #                   ): dot_call}),
         binop=15, prefix=15, ternary='.[')

def type_guard(t: Function, args: Value) -> Value:
    fn = None
    match t, *args.value:
        case Function(name='int') | Function(name='float') | Function(name='ratio'), \
             Value(value=int() | float() | Fraction() as min), Value(value=int() | float() | Fraction() as max):
            fn = lambda x: Value(min <= x.value < max)
        case Function(name='str'), \
             Value(value=int() | float() | Fraction() as min), Value(value=int() | float() | Fraction() as max):
            fn = lambda s: Value(min <= len(s.value) <= max)
        case Function(name='str'), Value(value=str() as regex):
            fn = lambda s: Value(bool(re.fullmatch(regex, s.value)))
        case Function(name='str'), Value(value=str() as regex), Value(value=str() as flags):
            f = 0
            for c in flags.upper():
                f |= getattr(re, c)
            fn = lambda s: Value(re.fullmatch(regex, s.value, f))
    # function = Function(ListPatt(AnyParam), fn)
    return Value(Prototype(t.value, guard=fn))

Operator('.[',
         Function(ListPatt(FunctionParam, ListParam), lambda a, b: a.call(b.value),
                  options={ListPatt(PatternParam, ListParam): type_guard,
                           ListPatt(ListParam): lambda a: Context.env.call(a.value)}),
         binop=15, prefix=15)

def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Value]:
    args = expressionize(rhs).evaluate()
    if len(lhs) > 2 and lhs[-1].type == TokenType.PatternName and lhs[-2].source_text == '.':
        name = lhs[-1].source_text
        # ((foo.len).max)[2]
        a = expressionize(lhs[:-2]).evaluate()
        try:
            assert a.instanceof(BuiltIns['fn'])
            fn = a.value.deref(name, ascend_env=False)
        except (AssertionError, NoMatchingOptionError):
            fn = Context.env.deref(name)
            if fn.type != BuiltIns['fn']:
                raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
            args = Value([a] + args.value)
    else:
        fn = expressionize(lhs).evaluate()
    return [fn, args]

Op['.['].eval_args = eval_call_args

# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
def number_guard(a: Value, b: Value, op_sym: str):
    # assert a.value == b.type
    return Value(Prototype(a.value, guard=lambda n: Op[op_sym].fn.call([n, b])))

# generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
def string_guard(a: Value, b: Value, op_sym: str):
    assert a.value == BuiltIns['str'] and b.type in (BuiltIns['int'], BuiltIns['float'])
    def guard(x, y):
        return Value(Prototype(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))
    # return guard
    return Value(Prototype(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))

def add_guards(op_sym: str):
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['int'])), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['float'])), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['ratio'])), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(BuiltIns['str'])), NumericParam),
                                lambda a, b: string_guard(a, b, op_sym))


for op in ('>', '<', '>=', '<='):
    add_guards(op)