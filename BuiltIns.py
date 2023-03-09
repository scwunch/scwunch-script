import re
from baseconvert import base
from Syntax import BasicType
from DataStructures import *
from Expressions import Expression, number


# NoneParam = Parameter(Type(BasicType.none))
# BoolParam = Parameter(Type(BasicType.Boolean))
# IntParam = Parameter(Type(BasicType.Integer))
# FloatParam = Parameter(Type(BasicType.Float))
# NumberParam = Parameter(basic_type=(BasicType.Integer, BasicType.Float))
# LogNumParam = Parameter(basic_type=(BasicType.Boolean, BasicType.Integer, BasicType.Float))
# StringParam = Parameter(Type(BasicType.String))
# NormalParam = Parameter(basic_type=(BasicType.Boolean, BasicType.Integer, BasicType.Float, BasicType.String))
# ListParam = Param = Parameter(Type(BasicType.List))
# TypeParam = Parameter(Type(BasicType.Type))
# TypeOrPatternParam = Parameter(basic_type=(BasicType.Type, BasicType.Pattern))
# FunctionParam = Parameter(Type(BasicType.Function))
# # OptionParam = Parameter(Type(BasicType.Option))
# AnyParam = Parameter(Type(BasicType.Any))
# NormalBinopPattern = Pattern(NormalParam, NormalParam)
# AnyBinopPattern = Pattern(AnyParam, AnyParam)

NoneParam = Parameter(Type(BasicType.none))
BoolParam = Parameter(Type(BasicType.Boolean))
IntParam = Parameter(Type(BasicType.Integer))
FloatParam = Parameter(Type(BasicType.Float))
NumberParam = Parameter(Union(Type(BasicType.Integer), Type(BasicType.Float)))
LogNumParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Float)))
StringParam = Parameter(Type(BasicType.String))
NormalParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Float), Type(BasicType.String)))
ListParam = Param = Parameter(Type(BasicType.List))
TypeParam = Parameter(Type(BasicType.Type))
TypeOrPatternParam = Parameter(Union(Type(BasicType.Type), Type(BasicType.Pattern)))
FunctionParam = Parameter(Type(BasicType.Function))
# OptionParam = Parameter(Type(BasicType.Option))
AnyParam = Parameter(Type(BasicType.Any))
NormalBinopPattern = ListPatt(NormalParam, NormalParam)
AnyBinopPattern = ListPatt(AnyParam, AnyParam)


BuiltIns['bool'] = Function(ListPatt(AnyParam), lambda x: Value(bool(x.value), BasicType.Boolean))
BuiltIns['number'] = Function(ListPatt(NormalParam),
                              lambda x: Value(int(x.value)) if x.type == BasicType.Boolean else Value(number(x.value)))
BuiltIns['int'] = Function(ListPatt(NormalParam), lambda x: Value(int(BuiltIns['number'].call([x]).value)))
BuiltIns['float'] = Function(ListPatt(NormalParam), lambda x: Value(float(BuiltIns['number'].call([x]).value)))
ToString = Function()
# ToString.add_option(ListPatt(), lambda: Value(BasicType.String))
ToString.add_option(ListPatt(NumberParam),
                    lambda n: Value('-' * (n.value < 0) + base(abs(n.value), 10, 6, string=True, recurring=False)))
ToString.add_option(ListPatt(AnyParam), lambda x: Value(str(x.value)))
ToString.add_option(ListPatt(Parameter(Type(BasicType.Type))), lambda t: Value(t.value.name))
BuiltIns['str'] = ToString

BuiltIns['type'] = Function(ListPatt(AnyParam), lambda v: Value(v.type, BasicType.Type))

BuiltIns['len'] = Function(ListPatt(StringParam), lambda s: Value(len(s.value)))
BuiltIns['len'].add_option(ListPatt(FunctionParam), lambda f: Value(len(f.value.options)))
BuiltIns['len'].add_option(ListPatt(ListParam), lambda l: Value(len(l.value)))
BuiltIns['len'].add_option(ListPatt(Parameter(Type(BasicType.Pattern))), lambda p: Value(len(p.value)))

BuiltIns['prototype'] = Function(ListPatt(FunctionParam), lambda f: Value(f.value.prototype))

def contains(a: Function, b: Value):
    return Value(b in (opt.value for opt in a.options))
Contains = Function(ListPatt(FunctionParam, AnyParam), contains)
BuiltIns['contains'] = Contains

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
         Function(ListPatt(TypeOrPatternParam, TypeOrPatternParam), lambda a, b: Union(make_patt(a), make_patt(b))),
         binop=4)
Op['|'].fn.add_option(AnyBinopPattern, lambda a, b: Union(make_patt(a), make_patt(b)))
# Operator('&',
#          Function(),
#          binop=5)
Operator('not',
         Function(ListPatt(AnyParam), lambda a: Value(not BuiltIns['bool'].call([a]).value)),
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
                  options={Pattern(AnyParam, TypeOrPatternParam): matchPattern}),
         binop=9)
Operator('+',
         Function(NormalBinopPattern, lambda a, b: Value(a.value + b.value)),
         binop=11)
Operator('-',
         Function(NormalBinopPattern, lambda a, b: Value(a.value - b.value),
                  options={Pattern(LogNumParam): lambda a: Value(-a.value)}),
         binop=11, prefix=13)
Operator('*',
         Function(ListPatt(LogNumParam, LogNumParam), lambda a, b: Value(a.value * b.value)),
         binop=12)
Operator('/',
         Function(ListPatt(LogNumParam, LogNumParam), lambda a, b: Value(a.value / b.value)),
         binop=12)
Operator('%',
         Function(ListPatt(LogNumParam, LogNumParam), lambda a, b: Value(a.value % b.value)),
         binop=12)
Operator('**',
         Function(ListPatt(LogNumParam, LogNumParam), lambda a, b: Value(a.value ** b.value)),
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
         Function(ListPatt(FunctionParam, Parameter(Type(BasicType.Name))), lambda a, b: a.call(Value(b[0].name)),
                  options={ListPatt(AnyParam,
                                    Parameter(Type(BasicType.Name)),
                                    Parameter(Type(BasicType.List), quantifier="?")
                                   ): dot_call}),
         binop=15, ternary='[')
def type_guard(a: Value, b: Value) -> Value:
    fn = None
    match a.value, *b.value:
        case BasicType.Integer | BasicType.Float, \
             Value(value=int() | float() as min), Value(value=int() | float() as max):
            fn = lambda x: min <= x < max
        case BasicType.String, Value(value=int() | float() as min), Value(value=int() | float() as max):
            fn = lambda s: min <= len(s) <= max
        case BasicType.String, Value(value=str() as regex):
            fn = lambda s: re.fullmatch(regex, s)
        case BasicType.String, Value(value=str() as regex), Value(value=str() as flags):
            f = 0
            for c in flags.upper():
                f |= getattr(re, c)
            fn = lambda s: re.fullmatch(regex, s, f)
    return Value(Type(a.value, guard=fn))
Operator('.[',
         Function(ListPatt(FunctionParam, ListParam), lambda a, b: a.value.call(b.value),
                  options={ListPatt(TypeParam, ListParam): type_guard}),
         binop=15)


# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
# def number_guard(a: Value, b: Value, op_sym: str):
#     # assert a.value == b.type
#     return Pattern(Parameter(basic_type=a.value,
#                              fn=lambda n: Op[op_sym].fn.call([n, b])))
#
# # generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
# def string_guard(a: Value, b: Value, op_sym: str):
#     assert a.value == BasicType.String and b.type in (BasicType.Integer, BasicType.Float)
#     return Pattern(Parameter(basic_type=BasicType.String,
#                              fn=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))
#
# for op in ('>', '<', '>=', '<='):
#     Op[op].fn.assign_option(ListPatt(Parameter(value=Value(BasicType.Integer)), NumberParam),
#                             lambda a, b: number_guard(a, b, op))
#     Op[op].fn.assign_option(ListPatt(Parameter(value=Value(BasicType.Float)), NumberParam),
#                             lambda a, b: number_guard(a, b, op))
#     Op[op].fn.assign_option(ListPatt(Parameter(value=Value(BasicType.String)), NumberParam),
#                             lambda a, b: string_guard(a, b, op))
#
#
#
