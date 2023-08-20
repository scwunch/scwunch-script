import re
from fractions import Fraction
from Syntax import Node, Token, List, TokenType
from Env import *
# from DataStructures import *
from tables import *
from Expressions import expressionize, read_number, Expression, py_eval, piliize

BuiltIns['blank'] = None
BuiltIns['Table'] = TableTable = MetaTable()
# BuiltIns['Table'].records.append(BuiltIns['Table'])

# BuiltIns['RowId'] = Formula('key', AnyMatcher(), 'placeholder for function object')  # Formula('key', TableMatcher(BuiltIns['ratio']))
# BuiltIns['Blank'] = TableTable
BuiltIns['Blank'] = SetTable()
BuiltIns['blank'] = PyValue(BuiltIns['Blank'], None)
print(py_value(None) == BuiltIns['blank'])
BuiltIns['bool'] = SetTable()
BuiltIns['false'] = PyValue(BuiltIns['bool'], False)
BuiltIns['true'] = PyValue(BuiltIns['bool'], True)
print((BuiltIns['false'], BuiltIns['true']) == (py_value(False), py_value(True)))

for t in ('ratio', 'float', 'str', 'Tuple', 'Frozenset'):
    BuiltIns[t] = VirtTable()
    # BuiltIns[t].upsert_field(Slot('key', TableMatcher(BuiltIns[t])))
    # TableTable.append_record(BuiltIns[t])


BuiltIns['Pattern'] = ListTable()
BuiltIns['Block'] = ListTable()

BuiltIns['Function'] = TableTable
BuiltIns['Field'] = ListTable()

def upsert_field_fields():
    BuiltIns['Field'].upsert_field(Slot('name', TableMatcher(BuiltIns['str'])))
    BuiltIns['Field'].upsert_field(Slot('type', TableMatcher(BuiltIns['Pattern'])))
    BuiltIns['Field'].upsert_field(Slot('is_formula', TableMatcher(BuiltIns['bool'])))
    BuiltIns['Field'].upsert_field(Slot('default', Union(TableMatcher(BuiltIns['Function']), AnyMatcher())))
    BuiltIns['Field'].upsert_field(Slot('formula', TableMatcher(BuiltIns['Function'])))
    BuiltIns['Field'].upsert_field(Slot('setter', TableMatcher(BuiltIns['Function'])))
upsert_field_fields()

BuiltIns['Option'] = ListTable()
BuiltIns['Option'].upsert_field(Slot('signature', TableMatcher(BuiltIns['Pattern'])))
BuiltIns['Option'].upsert_field(Slot('code block', TableMatcher(BuiltIns['Block'])))

BuiltIns['Function'] = FnTable = ListTable()
BuiltIns['Function'].upsert_field(Slot('options', TableMatcher(BuiltIns['Table'])))
                                                    # the type should also have a specifier like `list[Option]`
                                                    # and also a default value: []

# now redo the Field fields, since they weren't able to properly initialize while the fields were incomplete
upsert_field_fields()
BuiltIns['Field'].records = BuiltIns['Field'].records[5:]

BuiltIns['python_object'] = ListTable()

print("DONE ADDING BUILTIN TABLES.")

BuiltIns['int'] = FilterSlice(BuiltIns['ratio'], lambda n: n.value.denominator == 1)
BuiltIns['num'] = Pattern(Parameter(Union(TableMatcher(BuiltIns['float']), TableMatcher(BuiltIns['ratio']))))


# class BasePrototype(Function):
#     def __init__(self):
#         self.name = 'base_prototype'
#         self.type = None
#         self.mro = ()
#         self.env = Context.env
#         self.options = []
#         self.args = []
#         self.hashed_options = {}
#
# BuiltIns['_base_prototype'] = BasePrototype()  # Function({name='base_prototype': type={'mro':()})  # noqa
# # BuiltIns['_base_prototype'].type = None
# MetaType = Function({name="Type": type=BuiltIns['_base_prototype'])
# BuiltIns['BasicType'] = MetaType
# BuiltIns['none'] = Function({name='none': type=MetaType)
# BuiltIns['num'] = Function({name="num": type=MetaType)
# BuiltIns['float'] = Function({name='float': type=BuiltIns['num'])
# BuiltIns['ratio'] = Function({name='ratio': type=BuiltIns['num'])
# BuiltIns['int'] = Function({name='int': type=BuiltIns['ratio'])
# BuiltIns['bool'] = Function({name='bool': type=BuiltIns['int'])
# BuiltIns['iterable'] = Function({name='iterable': type=MetaType)
# BuiltIns['str'] = Function({name='str': type=BuiltIns['iterable'])
# BuiltIns['list'] = Function({name='list': type=BuiltIns['iterable'])
# BuiltIns['tuple'] = Function({name='tuple': type=BuiltIns['iterable'])
# BuiltIns['pattern'] = Function({name='pattern': type=MetaType)
# BuiltIns['value_pattern'] = Function({name='value_pattern': type=BuiltIns['pattern'])
# BuiltIns['union'] = Function({name='union': type=BuiltIns['pattern'])
# BuiltIns['type_pattern'] = Function({name='type_pattern': type=BuiltIns['pattern'])
# BuiltIns['parameters'] = Function({name='parameters': type=BuiltIns['pattern'])
# BuiltIns['fn'] = Function({name='fn': type=BuiltIns['_base_prototype'])
# TypeMap.update({
#     type(None): BuiltIns['none'],
#     bool: BuiltIns['bool'],
#     int: BuiltIns['int'],
#     Fraction: BuiltIns['ratio'],
#     float: BuiltIns['float'],
#     str: BuiltIns['str'],
#     list: BuiltIns['list'],
#     tuple: BuiltIns['tuple'],
#     Pattern: BuiltIns['pattern']
#     # ValueMatcher: BuiltIns['value_pattern'],
#     # Prototype: BuiltIns['type_pattern'],
#     # Union: BuiltIns['union'],
#     # Pattern: BuiltIns['parameters']
# })
# BuiltIns['python_object'] = Function(type=MetaType)
# BuiltIns['any'] = py_value(None)
# BuiltIns['any'].value, BuiltIns['any'].type = Any, BuiltIns['pattern']
# TypeMap[AnyPattern] = BuiltIns['any']

NoneParam = Parameter(TableMatcher(BuiltIns["blank"]))
BoolParam = Parameter(TableMatcher(BuiltIns["bool"]))
IntegralParam = Parameter(Union(TableMatcher(BuiltIns["bool"]), TableMatcher(BuiltIns["int"])))
FloatParam = Parameter(TableMatcher(BuiltIns["float"]))
RationalParam = Parameter(TableMatcher(BuiltIns["ratio"]))
NumericParam = Parameter(BuiltIns["num"])
StringParam = Parameter(TableMatcher(BuiltIns["str"]))
NormalParam = Parameter(Union(TableMatcher(BuiltIns['bool']), TableMatcher(BuiltIns['ratio']), TableMatcher(BuiltIns['float']), TableMatcher(BuiltIns['str'])))
ListParam = Param = Parameter(TableMatcher(BuiltIns["Table"]))
# TypeParam = Parameter(TableMatcher(BuiltIns["Type"]))
PatternParam = Parameter(TableMatcher(BuiltIns["Pattern"]))
FunctionParam = Parameter(TableMatcher(BuiltIns["Function"]))
AnyParam = Parameter(AnyMatcher())
NormalBinopPattern = Pattern(NormalParam, NormalParam)
AnyBinopPattern = Pattern(AnyParam, AnyParam)
AnyPlusPattern = Pattern(Parameter(AnyMatcher(), quantifier="+"))
AnyPattern = Pattern(Parameter(AnyMatcher(), quantifier="*"))

PositiveIntParam = Parameter(TableMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value > 0)))
NegativeIntParam = Parameter(TableMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value < 0)))
NonZeroIntParam = Parameter(TableMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value != 0)))
OneIndexList = Parameter(TableMatcher(BuiltIns['Table'],
                                      guard=lambda x: py_value(len(x.value) == 1 and
                                                               NonZeroIntParam.match_score(x.value[0]))))
bases = {'b': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6, 's': 7, 'o': 8, 'n': 9, 'd': 10}
def setting_set(prop: PyValue, val: PyValue):
    prop = prop.value
    val = val.value
    if prop == 'base' and isinstance(val, str):
        if val not in bases:
            raise ValueError('Invalid setting for base.  See manual for available settings.')
        val = bases[val[0]]
    Context.settings[prop] = val
    return BuiltIns['settings']
def setting_get(prop: PyValue):
    if prop.value == 'base':
        match Context.settings['base']:
            case 2:
                return py_value('b')
            case 3:
                return py_value('t')
            case 4:
                return py_value('q')
            case 5:
                return py_value('p')
            case 6:
                return py_value('h')
            case 7:
                return py_value('s')
            case 8:
                return py_value('o')
            case 9:
                return py_value('n')
            case 10:
                return py_value('d')
    return py_value(Context.settings[prop.value])
BuiltIns['settings'] = Function({'set':
                                Function({Pattern(StringParam, AnyParam): setting_set}),
                                'get':
                                 Function({Pattern(StringParam): setting_get})})

def key_to_param_set(key: PyValue) -> Pattern:
    if hasattr(key, 'value') and isinstance(key.value, list):
        vals = key.value
    else:
        vals = [key]
    params = (Parameter(ValueMatcher(pval, pval.value if isinstance(pval.value, str) else None)) for pval in vals)
    return Pattern(*params)
BuiltIns['set'] = Function({Pattern(AnyParam, AnyParam, AnyParam):
                           lambda fn, key, val: fn.assign_option(key_to_param_set(key), val).resolution})

BuiltIns['bool'].add_option(Pattern(AnyParam), lambda x: py_value(bool(x.value)))
BuiltIns['number'] = Function({Pattern(BoolParam): lambda x: py_value(int(x.value)),
                              Pattern(NumericParam): lambda x: py_value(x.value),
                               Pattern(StringParam): lambda x: py_value(read_number(x.value, Context.settings['base'])),
                               Pattern(StringParam, IntegralParam): lambda x, b: py_value(read_number(x.value, b.value))},
                              name='number')
BuiltIns['integer'] = Function({Pattern(NormalParam): lambda x: py_value(int(BuiltIns['number'].call([x]).value))},
                               name='integer')
BuiltIns['rational'] = Function({Pattern(NormalParam): lambda x: py_value(Fraction(BuiltIns['number'].call([x]).value))},
                                name='rational')
# BuiltIns['float'] = Function({Pattern(NormalParam), lambda x: py_value(float(BuiltIns['number'].call([x]).value)))
BuiltIns['string'] = Function({Pattern(AnyParam): lambda x: x.to_string(),
                              Pattern(NumericParam, IntegralParam):
                                  lambda n, b: py_value(write_number(n.value, b.value))},
                              name='string',)
# BuiltIns['string'].add_option(Pattern(ListParam), lambda l: py_value(str(l.value[1:])))
# BuiltIns['string'].add_option(Pattern(NumberParam),
#                               lambda n: py_value('-' * (n.value < 0) +
#                                               base(abs(n.value), 10, 6, string=True, recurring=False)))
# BuiltIns['string'].add_option(Pattern(Parameter(TableMatcher(BuiltIns["Type"]))), lambda t: py_value(t.value.name))

BuiltIns['type'] = Function({AnyParam: lambda v: v.type})

BuiltIns['len'] = Function({StringParam: lambda s: py_value(len(s.value))})
BuiltIns['len'].add_option(FunctionParam, lambda f: py_value(len(f.options)))
BuiltIns['len'].add_option(ListParam, lambda l: py_value(len(l.value)))
BuiltIns['len'].add_option(Parameter(TableMatcher(BuiltIns["Pattern"])), lambda p: py_value(len(p.value)))

# BuiltIns['prototype'] = Function({Pattern(FunctionParam): lambda f: py_value(f.value.prototype))

# BuiltIns['contains'] = Function({Pattern(FunctionParam: AnyParam),
#                                 lambda a, b: py_value(b in (opt.value for opt in a.options)))
# BuiltIns['List'] = Function({Pattern(Parameter(Any: quantifier='*')),
#                             lambda *vals: py_value(list(*vals)))
BuiltIns['len'].add_option(Pattern(Parameter(TableMatcher(BuiltIns['Table']))), lambda l: py_value(len(l.value)))

BuiltIns['options'] = Function({AnyParam: lambda x: piliize([py_value(lp.pattern) for lp in x.options])})
BuiltIns['names'] = Function({AnyParam: lambda x: piliize([py_value(k) for k in x.named_options.keys()])})
BuiltIns['keys'] = Function({AnyParam:
                            lambda x: piliize([lp.pattern[0].pattern.value for lp in x.options
                                             if len(lp.pattern) == 1 and isinstance(lp.pattern[0].pattern, ValueMatcher)])})

BuiltIns['max'] = Function({Parameter(BuiltIns["num"], quantifier='+'):
                                lambda *args: py_value(max(*[arg.value for arg in args])),
                           Parameter(TableMatcher(BuiltIns["str"]), quantifier='+'):
                                lambda *args: py_value(max(*[arg.value for arg in args])),
                            ListParam:
                                lambda ls: py_value(max(*[arg.value for arg in ls.value]))
                            })
BuiltIns['min'] = Function({Parameter(BuiltIns["num"], quantifier='+'):
                                lambda *args: py_value(min(*[arg.value for arg in args])),
                           Parameter(TableMatcher(BuiltIns["str"]), quantifier='+'):
                                lambda *args: py_value(min(*[arg.value for arg in args])),
                            ListParam: lambda ls: py_value(min(*[arg.value for arg in ls.value]))})
BuiltIns['abs'] = Function({NumericParam: lambda n: py_value(abs(n.value))})
def pili_round(num, places):
    num, places = num.value, places.value
    power = Context.settings['base'] ** places
    return py_value(round(num * power) / power)


BuiltIns['round'] = Function({NumericParam: lambda n: py_value(round(n.value)),
                             Pattern(NumericParam, IntegralParam): pili_round})

def inclusive_range(*args: PyValue):
    step = 1
    match len(args):
        case 0:
            return piliize([])
        case 1:
            stop = args[0].value
            if stop == 0:
                return piliize([])
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
        ls.append(py_value(i))
        i += step
    return piliize(ls)


BuiltIns['range'] = Function({Parameter(BuiltIns["num"], quantifier="*"): inclusive_range})
BuiltIns['map'] = Function({Pattern(ListParam, FunctionParam): lambda ls, fn: piliize([fn.call([val]) for val in ls.value]),
                           Pattern(FunctionParam, ListParam): lambda fn, ls: piliize([fn.call([val]) for val in ls.value])})
BuiltIns['filter'] = Function({Pattern(ListParam, FunctionParam):
                                   lambda ls, fn: piliize([v for v in ls.value if BuiltIns['bool'].call([fn.call([v])]).value]),
                               Pattern(FunctionParam, ListParam):
                                   lambda fn, ls: piliize([v for v in ls.value if BuiltIns['bool'].call([fn.call([v])]).value])})
BuiltIns['sum'] = Function({ListParam: lambda ls: py_value(sum(ls.py_vals()))})
BuiltIns['trim'] = Function({StringParam: lambda text: py_value(text.value.strip()),
                            Pattern(StringParam, StringParam): lambda t, c: py_value(t.value.strip(c.value))})
BuiltIns['upper'] = Function({StringParam: lambda text: py_value(text.value.upper())})
BuiltIns['lower'] = Function({StringParam: lambda text: py_value(text.value.lower())})
BuiltIns['self'] = lambda: Context.env.caller or Context.env or py_value(None)
def Args(fn: Function):
    arg_list = piliize([opt.value for opt in fn.args])
    arg_list.add_option(FunctionParam, lambda fn: Args(fn))
    return arg_list
BuiltIns['args'] = lambda: Args(Context.env)

def list_get(scope: ListTable, *args: PyValue):
    # fn = scope.type
    fn = scope
    if len(args) == 1:
        length = BuiltIns['len'].call([fn])
        index = args[0].value
        if abs(index) > length.value:
            raise IndexError(f'Line {Context.line}: Index {args[0]} out of range')
        index -= index > 0
        return fn[index]  # fn.value[index]
    else:
        raise NotImplementedError
def list_set(ls: ListTable, index: PyValue, val: Function):
    i = index.value[0].value
    i -= i > 0
    if i == len(ls.records):
        ls.records.append(val)
    else:
        ls.records[i] = val
    return val
def list_slice(ls: PyValue, start: PyValue, stop: PyValue):
    # ls = ls.type.value  # inexplicably worked before the tables rehash... needs fixing now
    start, stop = start.value,  stop.value
    if start == 0:
        start = None
    elif start > 0:
        start -= 1
    if stop == 0:
        stop = None
    elif stop > 0:
        stop -= 1
    return py_value(ls[start:stop])


# BuiltIns['list'].add_option(Pattern(PositiveIntParam), FuncBlock(list_get))
# BuiltIns['list'].add_option(Pattern(NegativeIntParam), FuncBlock(list_get))
# BuiltIns['list'].add_option(Pattern(IntegralParam, IntegralParam), FuncBlock(list_slice))
# BuiltIns['set'].add_option(Pattern(ListParam, OneIndexList, AnyParam), list_set)
# BuiltIns['push'] = Function({Pattern(Parameter(TableMatcher(BuiltIns['list'])), AnyParam):
#                             lambda fn, val: fn.value.append(val) or fn})
# BuiltIns['join'] = Function({Pattern(ListParam, StringParam):
#                             lambda ls, sep: py_value(sep.value.join(BuiltIns['string'].call([item]).value for item in ls.value))})
# BuiltIns['split'] = Function({Pattern(StringParam, StringParam): lambda txt, sep: piliize([py_value(s) for s in txt.value.split(sep.value)])})

def convert(name: str) -> Function:
    o = object()
    # py_fn = getattr(__builtins__, name, o)
    py_fn = __builtins__.get(name, o)  # noqa
    if py_fn is o:
        raise SyntaxErr(f"Name '{name}' not found.")
    # Context.root.add_option(Pattern(Parameter(name)), lambda *args: py_value(py_fn((arg.value for arg in args))))
    # def lambda_fn(*args):
    #     arg_list = list(arg.value for arg in args)
    #     return py_value(py_fn(*arg_list))
    # return Function({AnyPattern: lambda_fn)
    return Function({AnyPattern: lambda *args: py_value(py_fn(*(arg.value for arg in args)))})


# BuiltIns['python'] = Function({Pattern(StringParam): lambda n: convert(n.value))
BuiltIns['python'] = Function({StringParam: py_eval})  # lambda x: py_value(eval(x.value)))

for k, v in BuiltIns.items():
    match v:
        case Table() | Function():
            if v.name is None:
                v.name = k