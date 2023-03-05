from baseconvert import base
from Syntax import BasicType
from DataStructures import *
from Expressions import Expression, number


NoneParam = Parameter(basic_type=BasicType.none)
BoolParam = Parameter(basic_type=BasicType.Boolean)
IntParam = Parameter(basic_type=BasicType.Integer)
FloatParam = Parameter(basic_type=BasicType.Float)
NumberParam = Parameter(basic_type=(BasicType.Integer, BasicType.Float))
LogNumParam = Parameter(basic_type=(BasicType.Boolean, BasicType.Integer, BasicType.Float))
StringParam = Parameter(basic_type=BasicType.String)
NormalParam = Parameter(basic_type=(BasicType.Boolean, BasicType.Integer, BasicType.Float, BasicType.String))
ListParam = Param = Parameter(basic_type=BasicType.List)
TypeParam = Parameter(basic_type=BasicType.Type)
FunctionParam = Parameter(basic_type=BasicType.Function)
# OptionParam = Parameter(basic_type=BasicType.Option)
AnyParam = Parameter(basic_type=BasicType.Any)
NormalBinopPattern = Pattern(NormalParam, NormalParam)
AnyBinopPattern = Pattern(AnyParam, AnyParam)


BuiltIns['bool'] = Function(Pattern(AnyParam), lambda x: Value(bool(x.value), BasicType.Boolean))
BuiltIns['number'] = Function(Pattern(NormalParam),
                              lambda x: Value(int(x.value)) if x.type == BasicType.Boolean else Value(number(x.value)))
BuiltIns['int'] = Function(Pattern(NormalParam), lambda x: Value(int(BuiltIns['number'].call([x]).value)))
BuiltIns['float'] = Function(Pattern(NormalParam), lambda x: Value(float(BuiltIns['number'].call([x]).value)))
ToString = Function()
# ToString.add_option(Pattern(), lambda: Value(BasicType.String))
ToString.add_option(Pattern(NumberParam),
                    lambda n: Value('-' * (n.value < 0) + base(abs(n.value), 10, 6, string=True, recurring=False)))
ToString.add_option(Pattern(AnyParam), lambda x: Value(str(x.value)))
ToString.add_option(Pattern(Parameter(basic_type=BasicType.Type)), lambda t: Value(t.value.name))
BuiltIns['str'] = ToString

BuiltIns['type'] = Function(Pattern(AnyParam), lambda v: Value(v.type, BasicType.Type))

def contains(a: Function, b: Value):
    return Value(b in (opt.value for opt in a.options))
Contains = Function(Pattern(FunctionParam, AnyParam), contains)
BuiltIns['contains'] = Contains

Operator('=', binop=1, static=True, associativity='right')
Operator(':', binop=1, static=True, associativity='right')
Operator(':=', binop=1, static=True, associativity='right')
Operator('if',
         Function(Pattern(AnyParam), lambda x: x),
         binop=2, static=True, ternary='else')
Operator('??',
         binop=3, static=True)
Operator('or',
         binop=4, static=True)
Operator('and',
         binop=5, static=True)
Operator('not',
         Function(Pattern(AnyParam), lambda a: Value(not BuiltIns['bool'].call([a]).value)),
         prefix=6)
Operator('in',
         Function(AnyBinopPattern, lambda a, b: Contains.call([b, a])),
         binop=7)
Operator('==',
         Function(AnyBinopPattern, lambda a, b: Value(a == b)),
         binop=8)
Operator('!=',
         Function(AnyBinopPattern, lambda a, b: Value(a != b)),
         binop=9)
Operator('<',
         Function(NormalBinopPattern, lambda a, b: Value(a.value < b.value)),
         binop=10)
Operator('>',
         Function(NormalBinopPattern, lambda a, b: Value(a.value > b.value)),
         binop=10)
Operator('<=',
         Function(NormalBinopPattern, lambda a, b: Value(a.value <= b.value)),
         binop=10)
Operator('>=',
         Function(NormalBinopPattern, lambda a, b: Value(a.value >= b.value)),
         binop=10)
def matchPattern(a: Value, b: Pattern | Value):
    match b:
        case Pattern():
            assert len(b) == 1
            return Value(bool(match_score(a, b.all_parameters[0])), BasicType.Boolean)
        case Value():
            assert b.type == BasicType.Type
            return Value(a.type == b.value, BasicType.Boolean)
Operator('~',
         Function(AnyBinopPattern, lambda a, b: a.type == b.type,
                  options={Pattern(AnyParam, Parameter(basic_type=(BasicType.Type, BasicType.Pattern))): matchPattern}),
         binop=9)
Operator('+',
         Function(NormalBinopPattern, lambda a, b: Value(a.value + b.value)),
         binop=11)
Operator('-',
         Function(NormalBinopPattern, lambda a, b: Value(a.value - b.value),
                  options={Pattern(LogNumParam): lambda a: Value(-a.value)}),
         binop=11, prefix=13)
Operator('*',
         Function(Pattern(LogNumParam, LogNumParam), lambda a, b: Value(a.value * b.value)),
         binop=12)
Operator('/',
         Function(Pattern(LogNumParam, LogNumParam), lambda a, b: Value(a.value / b.value)),
         binop=12)
Operator('%',
         Function(Pattern(LogNumParam, LogNumParam), lambda a, b: Value(a.value % b.value)),
         binop=12)
Operator('**',
         Function(Pattern(LogNumParam, LogNumParam), lambda a, b: Value(a.value ** b.value)),
         binop=13, associativity='right')
Operator('?',
         postfix=14, static=True)
def dot_call(a: Value, b: Value, c: Value = None):
    name = b.value
    args: list[Value] = []
    if c:
        assert isinstance(c.value, list)
        args = c.value
    try:
        assert a.type == BasicType.Function and isinstance(a.value, Function)
        val = a.value.deref(name, ascend_env=False)
    except (AssertionError, NoMatchingOptionError):
        fn = Context.env.deref(name)
        assert isinstance(fn, Function)
        args = [a] + args
        return fn.call(args)
    if c:
        assert isinstance(val.value, Function)
        return val.value.call(args)
    else:
        return val

Operator('.',
         Function(Pattern(FunctionParam, Parameter(basic_type=BasicType.Name)), lambda a, b: a.call(Value(b[0].name)),
                  options={Pattern(AnyParam,
                                   Parameter(basic_type=BasicType.Name), optional_parameters=(ListParam,)
                                   ): dot_call}),
         binop=15, ternary='[')
Operator('.[',
         Function(Pattern(FunctionParam, ListParam), lambda a, b: a.value.call(b.value)),
         binop=15)


# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
def number_guard(a: Value, b: Value, op_sym: str):
    # assert a.value == b.type
    return Pattern(Parameter(basic_type=a.value,
                             fn=lambda n: Op[op_sym].fn.call([n, b])))

# generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
def string_guard(a: Value, b: Value, op_sym: str):
    assert a.value == BasicType.String and b.type in (BasicType.Integer, BasicType.Float)
    return Pattern(Parameter(basic_type=BasicType.String,
                             fn=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))

for op in ('>', '<', '>=', '<='):
    Op[op].fn.assign_option(Pattern(Parameter(value=Value(BasicType.Integer)), NumberParam),
                            lambda a, b: number_guard(a, b, op))
    Op[op].fn.assign_option(Pattern(Parameter(value=Value(BasicType.Float)), NumberParam),
                            lambda a, b: number_guard(a, b, op))
    Op[op].fn.assign_option(Pattern(Parameter(value=Value(BasicType.String)), NumberParam),
                            lambda a, b: string_guard(a, b, op))



