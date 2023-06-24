import re
from fractions import Fraction
from Syntax import Node, Token, List, TokenType
from Env import *
from DataStructures import *
from Expressions import expressionize, read_number, Expression, py_eval, piliize

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
BuiltIns['iterable'] = Function(name='iterable', type=MetaType)
BuiltIns['str'] = Function(name='str', type=BuiltIns['iterable'])
BuiltIns['list'] = Function(name='list', type=BuiltIns['iterable'])
BuiltIns['tuple'] = Function(name='tuple', type=BuiltIns['iterable'])
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
    tuple: BuiltIns['tuple'],
    Pattern: BuiltIns['pattern'],
    ValuePattern: BuiltIns['value_pattern'],
    Prototype: BuiltIns['type_pattern'],
    Union: BuiltIns['union'],
    ListPatt: BuiltIns['parameters']
})
BuiltIns['python_object'] = Function(type=MetaType)
BuiltIns['any'] = Value(None)
BuiltIns['any'].value, BuiltIns['any'].type = Any, BuiltIns['pattern']
TypeMap[AnyPattern] = BuiltIns['any']

NoneParam = Parameter(Prototype(BuiltIns["none"]))
BoolParam = Parameter(Prototype(BuiltIns["bool"]))
IntegralParam = Parameter(Union(Prototype(BuiltIns["bool"]), Prototype(BuiltIns["int"])))
FloatParam = Parameter(Prototype(BuiltIns["float"]))
RationalParam = Parameter(Prototype(BuiltIns["ratio"]))
NumericParam = Parameter(Prototype(BuiltIns["num"]))
StringParam = Parameter(Prototype(BuiltIns["str"]))
NormalParam = Parameter(Union(Prototype(BuiltIns['num']), Prototype(BuiltIns['str'])))
ListParam = Param = Parameter(Prototype(BuiltIns["list"]))
# TypeParam = Parameter(Prototype(BuiltIns["Type"]))
PatternParam = Parameter(Prototype(BuiltIns["pattern"]))
FunctionParam = Parameter(Prototype(BuiltIns["fn"]))
AnyParam = Parameter(Any)
NormalBinopPattern = ListPatt(NormalParam, NormalParam)
AnyBinopPattern = ListPatt(AnyParam, AnyParam)
AnyPlusPattern = ListPatt(Parameter(Any, quantifier="+"))
AnyListPatt = ListPatt(Parameter(Any, quantifier="*"))

PositiveIntParam = Parameter(Prototype(BuiltIns["int"], guard=lambda x: Value(x.value > 0)))
NegativeIntParam = Parameter(Prototype(BuiltIns["int"], guard=lambda x: Value(x.value < 0)))
NonZeroIntParam = Parameter(Prototype(BuiltIns["int"], guard=lambda x: Value(x.value != 0)))
OneIndexList = Parameter(Prototype(BuiltIns['list'], guard=lambda x: Value(len(x.value) == 1 and
                                                                           NonZeroIntParam.match_score(x.value[0]))))
bases = {'b': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6, 's': 7, 'o': 8, 'n': 9, 'd': 10}
def setting_set(prop: Value, val: Value):
    prop = prop.value
    val = val.value
    if prop == 'base' and isinstance(val, str):
        if not val in bases:
            raise ValueError('Invalid setting for base.  See manual for available settings.')
        val = bases[val[0]]
    Context.settings[prop] = val
    return BuiltIns['settings']
def setting_get(prop: Value):
    if prop.value == 'base':
        match Context.settings['base']:
            case 2:
                return Value('b')
            case 3:
                return Value('t')
            case 4:
                return Value('q')
            case 5:
                return Value('p')
            case 6:
                return Value('h')
            case 7:
                return Value('s')
            case 8:
                return Value('o')
            case 9:
                return Value('n')
            case 10:
                return Value('d')
    return Value(Context.settings[prop.value])
BuiltIns['settings'] = Function('set',
                                Function(ListPatt(StringParam, AnyParam), setting_set),
                                {'get':
                                 Function(ListPatt(StringParam), setting_get)})

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
                              {ListPatt(NumericParam): Value.clone,
                               ListPatt(StringParam): lambda x: Value(read_number(x.value, Context.settings['base'])),
                               ListPatt(StringParam, IntegralParam): lambda x, b: Value(read_number(x.value, b.value))},
                              name='number')
BuiltIns['integer'] = Function(ListPatt(NormalParam), lambda x: Value(int(BuiltIns['number'].call([x]).value)),
                               name='integer')
BuiltIns['rational'] = Function(ListPatt(NormalParam), lambda x: Value(Fraction(BuiltIns['number'].call([x]).value)),
                                name='rational')
# BuiltIns['float'] = Function(ListPatt(NormalParam), lambda x: Value(float(BuiltIns['number'].call([x]).value)))
BuiltIns['string'] = Function(ListPatt(AnyParam), lambda x: x.to_string(), name='string')
BuiltIns['string'].add_option(ListPatt(NumericParam, IntegralParam), lambda n, b: Value(write_number(n.value, b.value)))
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

BuiltIns['options'] = Function(ListPatt(AnyParam), lambda x: Value([Value(lp.pattern) for lp in x.options]))
BuiltIns['names'] = Function(ListPatt(AnyParam), lambda x: Value([Value(k) for k in x.named_options.keys()]))
BuiltIns['keys'] = Function(ListPatt(AnyParam),
                            lambda x: Value([lp.pattern[0].pattern.value for lp in x.options
                                             if len(lp.pattern) == 1 and isinstance(lp.pattern[0].pattern, ValuePattern)]))

BuiltIns['max'] = Function(ListPatt(Parameter(Prototype(BuiltIns["num"]), quantifier='+')),
                           lambda *args: Value(max(*[arg.value for arg in args])),
                           {ListPatt(Parameter(Prototype(BuiltIns["str"]), quantifier='+')):
                                lambda *args: Value(max(*[arg.value for arg in args])),
                            ListParam: lambda ls: Value(max(*[arg.value for arg in ls.value]    ))})
BuiltIns['min'] = Function(ListPatt(Parameter(Prototype(BuiltIns["num"]), quantifier='+')),
                           lambda *args: Value(min(*[arg.value for arg in args])),
                           {ListPatt(Parameter(Prototype(BuiltIns["str"]), quantifier='+')):
                                lambda *args: Value(min(*[arg.value for arg in args])),
                            ListParam: lambda ls: Value(min(*[arg.value for arg in ls.value]))})
BuiltIns['abs'] = Function(NumericParam, lambda n: Value(abs(n.value)))
def pili_round(num, places):
    num, places = num.value, places.value
    power = Context.settings['base'] ** places
    return Value(round(num * power) / power)
BuiltIns['round'] = Function(NumericParam, lambda n: Value(round(n.value)),
                             {ListPatt(NumericParam, IntegralParam): pili_round})

def inclusive_range(*args: Value):
    step = 1
    match len(args):
        case 0:
            return Value([])
        case 1:
            stop = args[0].value
            if stop == 0:
                return Value([])
            elif stop > 0:
                start = 1
            else:
                start = step = -1
        case 2:
            start = args[0].value
            stop = args[1].value
            step = stop > start or -1
        case 3:
            start, stop, step = (a.value for a in args)
            if not step:
                raise RuntimeErr(f"Line {Context.line}: Third argument in range (step) cannot be 0.")
        case _:
            raise RuntimeErr(f"Line {Context.line}: Too many arguments for range")
    i = start
    ls = []
    while step > 0 and i <= stop or step < 0 and i >= stop:
        ls.append(Value(i))
        i += step
    return Value(ls)
BuiltIns['range'] = Function(Parameter(Prototype(BuiltIns['num']), quantifier="*"), inclusive_range)
BuiltIns['map'] = Function(ListPatt(ListParam, FunctionParam), lambda ls, fn: Value([fn.call([val]) for val in ls.value]),
                           {ListPatt(FunctionParam, ListParam): lambda fn, ls: Value([fn.call([val]) for val in ls.value])})
BuiltIns['filter'] = Function(ListPatt(ListParam, FunctionParam),
                        lambda ls, fn: Value([v for v in ls.value if BuiltIns['bool'].call([fn.call([v])]).value]),
                        {ListPatt(FunctionParam, ListParam):
                            lambda fn, ls: Value([v for v in ls.value if BuiltIns['bool'].call([fn.call([v])]).value])})
BuiltIns['sum'] = Function(ListParam, lambda ls: Value(sum(ls.py_vals())))
BuiltIns['trim'] = Function(StringParam, lambda text: Value(text.value.strip()),
                            {ListPatt(StringParam, StringParam): lambda t, c: Value(t.value.strip(c.value))})
BuiltIns['upper'] = Function(StringParam, lambda text: Value(text.value.upper()))
BuiltIns['lower'] = Function(StringParam, lambda text: Value(text.value.lower()))
BuiltIns['self'] = lambda: Context.env.caller or Context.env or Value(None)
def Args(fn: Function):
    arg_list = Value([opt.value for opt in fn.args])
    arg_list.add_option(FunctionParam, lambda fn: Args(fn))
    return arg_list
BuiltIns['args'] = lambda: Args(Context.env)

def list_get(scope: Function, *args: Value):
    fn = scope.type
    if len(args) == 1:
        length = BuiltIns['len'].call([fn])
        index = args[0].value
        if abs(index) > length.value:
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
def list_slice(ls: Value, start: Value, stop: Value):
    ls = ls.type.value  # I don't know why this works, but it does... I need to fix the prototyping and context handling
    start, stop = start.value,  stop.value
    if start == 0:
        start = None
    elif start > 0:
        start -= 1
    if stop == 0:
        stop = None
    elif stop > 0:
        stop -= 1
    return Value(ls[start:stop])


BuiltIns['list'].add_option(ListPatt(PositiveIntParam), FuncBlock(list_get))
BuiltIns['list'].add_option(ListPatt(NegativeIntParam), FuncBlock(list_get))
BuiltIns['list'].add_option(ListPatt(IntegralParam, IntegralParam), FuncBlock(list_slice))
BuiltIns['set'].add_option(ListPatt(ListParam, OneIndexList, AnyParam), list_set)
BuiltIns['push'] = Function(ListPatt(Parameter(Prototype(BuiltIns['list'])), AnyParam),
                            lambda fn, val: fn.value.append(val) or fn)
BuiltIns['join'] = Function(ListPatt(ListParam, StringParam),
                            lambda ls, sep: Value(sep.value.join(BuiltIns['string'].call([item]).value for item in ls.value)))
BuiltIns['split'] = Function(ListPatt(StringParam, StringParam), lambda txt, sep: Value([Value(s) for s in txt.value.split(sep.value)]))

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

# BuiltIns['python'] = Function(ListPatt(StringParam), lambda n: convert(n.value))
BuiltIns['python'] = Function(StringParam, py_eval) # lambda x: Value(eval(x.value)))