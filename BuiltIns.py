import re
from fractions import Fraction
from Syntax import BasicType, Node, TokenType
from Env import *
from DataStructures import *
from Expressions import expressionize, read_number, eval_node

NoneParam = Parameter(Type(BasicType.none))
BoolParam = Parameter(Type(BasicType.Boolean))
IntegralParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer)))
FloatParam = Parameter(Type(BasicType.Float))
RationalParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Rational)))
NumericParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Rational), Type(BasicType.Float)))
StringParam = Parameter(Type(BasicType.String))
NormalParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Rational),
                              Type(BasicType.Float), Type(BasicType.String)))
ListParam = Param = Parameter(Type(BasicType.List))
TypeParam = Parameter(Type(BasicType.Type))
TypeOrPatternParam = Parameter(Union(Type(BasicType.Type), Type(BasicType.Pattern)))
FunctionParam = Parameter(Type(BasicType.Function))
AnyParam = Parameter(Type(BasicType.Any))
NormalBinopPattern = ListPatt(NormalParam, NormalParam)
AnyBinopPattern = ListPatt(AnyParam, AnyParam)
NegativeRationalParam = Parameter(Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Rational),
                                        guard=lambda x: Value(x.value < 0)))

BuiltIns['numeric'] = Union(Type(BasicType.Boolean), Type(BasicType.Integer), Type(BasicType.Rational), Type(BasicType.Float))
BuiltIns['boolean'] = Function(ListPatt(AnyParam), lambda x: Value(bool(x.value), BasicType.Boolean))
BuiltIns['number'] = Function(ListPatt(BoolParam), lambda x: Value(int(x.value)),
                              options={ListPatt(NumericParam): Value.clone,
                                       ListPatt(StringParam): lambda x: Value(read_number(x.value))})
BuiltIns['integer'] = Function(ListPatt(NormalParam), lambda x: Value(int(BuiltIns['number'].call([x]).value)))
BuiltIns['rational'] = Function(ListPatt(NormalParam), lambda x: Value(Fraction(BuiltIns['number'].call([x]).value)))
BuiltIns['float'] = Function(ListPatt(NormalParam), lambda x: Value(float(BuiltIns['number'].call([x]).value)))
BuiltIns['string'] = Function(ListPatt(AnyParam), lambda x: Value(str(x.value)))
# BuiltIns['string'].add_option(ListPatt(NumberParam),
#                               lambda n: Value('-' * (n.value < 0) +
#                                               base(abs(n.value), 10, 6, string=True, recurring=False)))
BuiltIns['string'].add_option(ListPatt(Parameter(Type(BasicType.Type))), lambda t: Value(t.value.name))

BuiltIns['type'] = Function(ListPatt(AnyParam), lambda v: Value(v.type, BasicType.Type))

BuiltIns['len'] = Function(ListPatt(StringParam), lambda s: Value(len(s.value)))
BuiltIns['len'].add_option(ListPatt(FunctionParam), lambda f: Value(len(f.value.options)))
BuiltIns['len'].add_option(ListPatt(ListParam), lambda l: Value(len(l.value)))
BuiltIns['len'].add_option(ListPatt(Parameter(Type(BasicType.Pattern))), lambda p: Value(len(p.value)))

BuiltIns['prototype'] = Function(ListPatt(FunctionParam), lambda f: Value(f.value.prototype))

BuiltIns['contains'] = Function(ListPatt(FunctionParam, AnyParam),
                                lambda a, b: Value(b in (opt.value for opt in a.options)))
BuiltIns['List'] = Function(ListPatt(Parameter(Type(BasicType.Any), quantifier='*')),
                            lambda *vals: Value(ListFunc(*vals)))
BuiltIns['len'].add_option(ListPatt(Parameter(Prototype(BuiltIns['List']))), lambda l: Value(len(l.value.array)-1))
def neg_index(scope: ListFunc, *args: Value):
    fn = scope.prototype
    if len(args) == 1:
        length = BuiltIns['len'].call([Value(fn)]).value
        index = Value(length + 1 + args[0].value)
        return fn.call([index])
    else:
        raise NotImplemented
BuiltIns['List'].add_option(ListPatt(NegativeRationalParam), FuncBlock(neg_index))
def push(scope: ListFunc, *args: Value):
    if len(args) == 1:
        val = args[0]
        """ Yikes, the actual parent list object that I wish to push to is not available from scope! """
        scope.env.add_option(ValuePattern(Value(len(scope.env.array))), val)
        return Value(scope.env)
    else:
        raise NotImplemented

Push = Function(ListPatt(AnyParam), FuncBlock(push, env=BuiltIns['List']), env=BuiltIns['List'])
BuiltIns['List'].add_option(named_patt('push'), Value(Push))

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
         Function(ListPatt(TypeOrPatternParam, TypeOrPatternParam), lambda a, b: Value(Union(make_patt(a), make_patt(b)))),
         binop=4)
Op['|'].fn.add_option(AnyBinopPattern, lambda a, b: Value(Union(make_patt(a), make_patt(b))))
Operator('not',
         Function(ListPatt(AnyParam), lambda a: Value(not BuiltIns['boolean'].call([a]).value)),
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
def match_pattern(a: Value, b: Value):
    return Value(bool(make_patt(b).match_score(a)))
Operator('~',
         Function(AnyBinopPattern, lambda a, b: a.type == b.type,
                  options={ListPatt(AnyParam, TypeOrPatternParam): match_pattern}),
         binop=9, chainable=False)
Operator('!~',
         Function(AnyBinopPattern, lambda a, b: a.type != b.type,
                  options={ListPatt(AnyParam, TypeOrPatternParam):
                               lambda a, b: Value(not make_patt(b).match_score(a))}),
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
#         assert a.type == BasicType.Function and isinstance(a.value, Function)
#         val = a.value.deref(name, ascend_env=False)
#     except (AssertionError, NoMatchingOptionError):
#         fn = Context.env.deref(name)
#         if fn.type != BasicType.Function:
#             raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
#         assert isinstance(fn.value, Function)
#         args = [a] + args
#         return fn.value.call(args)
#     if c:
#         fn = val.value
#         if not isinstance(fn, Function):
#             raise OperatorError(f"Line {Context.line}: {val.type.value} '{name}' is not function.")
#         return fn.call(args)
#     else:
#         return val

def dot_fn(a: Value, b: Value):
    name = b.value
    try:
        assert a.type == BasicType.Function and isinstance(a.value, Function)
        return a.value.deref(name, ascend_env=False)
    except (AssertionError, NoMatchingOptionError):
        fn = Context.env.deref(name)
        if fn.type != BasicType.Function:
            raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
        assert isinstance(fn.value, Function)
        return fn.value.call([a])

Operator('.',
         Function(ListPatt(AnyParam, Parameter(Type(BasicType.Name))), dot_fn,
                  options={ListPatt(Parameter(Type(BasicType.Name))): lambda a: dot_fn(Value(Context.env), a)}),
                  # options={ListPatt(AnyParam,
                  #                   Parameter(Type(BasicType.Name)),
                  #                   Parameter(Type(BasicType.List), quantifier="?")
                  #                   ): dot_call}),
         binop=15, prefix=15, ternary='.[')

def type_guard(t: Value, args: Value) -> Value:
    fn = None
    match t.value, *args.value:
        case BasicType.Integer | BasicType.Float | BasicType.Rational, \
             Value(value=int() | float() | Fraction() as min), Value(value=int() | float() | Fraction() as max):
            fn = lambda x: Value(min <= x.value < max)
        case BasicType.String, \
             Value(value=int() | float() | Fraction() as min), Value(value=int() | float() | Fraction() as max):
            fn = lambda s: Value(min <= len(s.value) <= max)
        case BasicType.String, Value(value=str() as regex):
            fn = lambda s: Value(bool(re.fullmatch(regex, s.value)))
        case BasicType.String, Value(value=str() as regex), Value(value=str() as flags):
            f = 0
            for c in flags.upper():
                f |= getattr(re, c)
            fn = lambda s: Value(re.fullmatch(regex, s.value, f))
    # function = Function(ListPatt(AnyParam), fn)
    return Value(Type(t.value, guard=fn))

Operator('.[',
         Function(ListPatt(FunctionParam, ListParam), lambda a, b: a.value.call(b.value),
                  options={ListPatt(TypeParam, ListParam): type_guard,
                           ListPatt(ListParam): lambda a: Context.env.call(a.value)}),
         binop=15, prefix=15)

def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Value]:
    args = expressionize(rhs).evaluate()
    if len(lhs) > 2 and lhs[-1].type == TokenType.PatternName and lhs[-2].source_text == '.':
        name = lhs[-1].source_text
        # ((foo.len).max)[2]
        a = expressionize(lhs[:-2]).evaluate()
        try:
            assert a.type == BasicType.Function
            fn = a.value.deref(name, ascend_env=False)
        except (AssertionError, NoMatchingOptionError):
            fn = Context.env.deref(name)
            if fn.type != BasicType.Function:
                raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
            args = Value([a] + args.value)
    else:
        fn = expressionize(lhs).evaluate()
    return [fn, args]

Op['.['].eval_args = eval_call_args

# Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
def number_guard(a: Value, b: Value, op_sym: str):
    # assert a.value == b.type
    return Value(Type(a.value, guard=lambda n: Op[op_sym].fn.call([n, b])))

# generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
def string_guard(a: Value, b: Value, op_sym: str):
    assert a.value == BasicType.String and b.type in (BasicType.Integer, BasicType.Float)
    def guard(x, y):
        return Value(Type(BasicType.String, guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))
    # return guard
    return Value(Type(BasicType.String, guard=lambda s: Op[op_sym].fn.call([Value(len(s.value)), b])))

def add_guards(op_sym: str):
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(Value(BasicType.Integer))), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(Value(BasicType.Float))), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(Value(BasicType.Rational))), NumericParam),
                                lambda a, b: number_guard(a, b, op_sym))
    Op[op_sym].fn.assign_option(ListPatt(Parameter(ValuePattern(Value(BasicType.String))), NumericParam),
                                lambda a, b: string_guard(a, b, op_sym))


for op in ('>', '<', '>=', '<='):
    add_guards(op)