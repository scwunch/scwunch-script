import re
from fractions import Fraction
# from Syntax import Node, Token, ListNode, TokenType
from Env import *
from tables import *
from patterns import *
# from Expressions import expressionize, read_number, Expression, py_eval, piliize

print(f"loading module: {__name__} ...")

# BuiltIns['blank'] = None
BuiltIns['Table'] = TableTable = MetaTable()
BuiltIns['Trait'] = TraitTable = BootstrapTable('Trait')
BuiltIns['Pattern'] = IntermediatePatternTable = BootstrapTable('Pattern')
BuiltIns['Option'] = IntermediateOptionTable = BootstrapTable('Option')
BuiltIns['Args'] = IntermediateArgsTable = BootstrapTable('Args')
TableTable.traits = (Trait(),)
TraitTable.traits = (Trait(),)

BuiltIns['Blank'] = SetTable(name='Blank')
BuiltIns['blank'] = PyValue(BuiltIns['Blank'], None)

# Number traits
BuiltIns['num'] = NumTrait = Trait(name='num')
BuiltIns['float'] = Trait(name='float')
BuiltIns['ratio'] = RatioTrait = Trait(name='ratio')
BuiltIns['int'] = IntTrait = Trait(name='int')
BuiltIns['bool'] = Trait(name='bool')

# Collection Traits
BuiltIns['iter'] = IterTrait = Trait(name='iter')
BuiltIns['seq'] = SeqTrait = Trait(name='seq')
BuiltIns['str'] = StrTrait = Trait(name='str')
BuiltIns['tuple'] = TupTrait = Trait(name='tuple')
BuiltIns['set'] = SetTrait = Trait(name='set')
BuiltIns['frozenset'] = FrozenSetTrait = Trait(name='frozenset')
BuiltIns['list'] = ListTrait = Trait(name='list')
BuiltIns['dict'] = DictTrait = Trait(name='dict')

# Numeric Tables
BuiltIns['Bool'] = SetTable(BuiltIns['bool'], IntTrait, RatioTrait, NumTrait, name='Bool')
BuiltIns['false'] = PyValue(BuiltIns['Bool'], False)
BuiltIns['true'] = PyValue(BuiltIns['Bool'], True)

BuiltIns['Integer'] = VirtTable(IntTrait, RatioTrait, NumTrait, name='Integer')
BuiltIns['Fraction'] = VirtTable(RatioTrait, NumTrait, name='Fraction')
BuiltIns['Float'] = VirtTable(BuiltIns['float'], NumTrait, name='Float')

# Collection Tables
BuiltIns['String'] = VirtTable(StrTrait, SeqTrait, IterTrait, name='String')
BuiltIns['Tuple'] = VirtTable(TupTrait, SeqTrait, IterTrait, name='Tuple')
BuiltIns['Set'] = VirtTable(SetTrait, IterTrait, name='Set')
BuiltIns['List'] = VirtTable(ListTrait, SeqTrait, IterTrait, name='List')
BuiltIns['Dictionary'] = VirtTable(DictTrait, IterTrait, name='Dictionary')
BuiltIns['Args'] = VirtTable(SeqTrait, DictTrait, IterTrait, name='Args')
for rec in IntermediateArgsTable.records:
    rec.table = BuiltIns['Args']

BuiltIns['fn'] = FuncTrait = Trait(name='fn')
BuiltIns['Function'] = ListTable(FuncTrait, name='Function')

TableTable.traits += (FuncTrait,)
BuiltIns['Trait'].traits += (FuncTrait,)

BuiltIns['Pattern'] = ListTable(SeqTrait, IterTrait, name='Pattern')
BuiltIns['Pattern'].records = IntermediatePatternTable.records
# BuiltIns["Pattern"] = BuiltIns['ArgsMatcher']
BuiltIns['Block'] = ListTable(name='Block')

BuiltIns['Field'] = ListTable(name='Field')

def upsert_field_fields(fields: list[Field]):
    fields.append(Slot('name', TraitMatcher(BuiltIns['str'])))
    fields.append(Slot('type', TableMatcher(BuiltIns['Pattern'])))
    fields.append(Slot('is_formula', TraitMatcher(BuiltIns['bool'])))
    fields.append(Slot('default', UnionMatcher(TraitMatcher(FuncTrait), AnyMatcher())))
    fields.append(Slot('formula', TraitMatcher(FuncTrait)))
    fields.append(Slot('setter', TraitMatcher(FuncTrait)))
# upsert_field_fields(BuiltIns['Field'].trait.fields)


BuiltIns['Option'] = ListTable(name='Option')
BuiltIns['Option'].records = IntermediateOptionTable.records
BuiltIns['Option'].trait.fields.append(Slot('signature', TableMatcher(BuiltIns['Pattern'])))
BuiltIns['Option'].trait.fields.append(Slot('code block', TableMatcher(BuiltIns['Block'])))

FuncTrait.fields.append(Slot('options', TraitMatcher(BuiltIns['seq'])))
# the type should also have a specifier like `list[Option]` ... and also a default value: []

# now redo the Field fields, since they weren't able to properly initialize while the fields were incomplete
# upsert_field_fields(BuiltIns['Field'].trait.fields)
BuiltIns['Field'].records = BuiltIns['Field'].records[5:]

BuiltIns['PythonObject'] = ListTable(name='PythonObject')

print("DONE ADDING BUILTIN TABLES.")


BuiltIns['any'] = AnyMatcher()
NoneParam = Parameter(ValueMatcher(BuiltIns["blank"]))
BoolParam = Parameter(TraitMatcher(BuiltIns["bool"]))
IntegralParam = Parameter(TraitMatcher(BuiltIns["int"]))
FloatParam = Parameter(TraitMatcher(BuiltIns["float"]))
RationalParam = Parameter(TraitMatcher(BuiltIns["ratio"]))
NumericParam = Parameter(TraitMatcher(BuiltIns['num']))
StringParam = Parameter(TraitMatcher(BuiltIns["str"]))
NormalParam = Parameter(UnionMatcher(TraitMatcher(BuiltIns['num']), TraitMatcher(BuiltIns['str'])))
SeqParam = Parameter(TraitMatcher(SeqTrait))
ListParam = Parameter(TraitMatcher(ListTrait))
NonStr = TraitMatcher(StrTrait)
NonStr.invert = 1
NonStrSeqParam = Parameter(IntersectionMatcher(TraitMatcher(SeqTrait), NonStr))
IterParam = Parameter(TraitMatcher(IterTrait))
# TypeParam = Parameter(TableMatcher(BuiltIns["Type"]))
PatternParam = Parameter(TableMatcher(BuiltIns['Pattern']))
FunctionParam = Parameter(TraitMatcher(BuiltIns["fn"]))
TableParam = Parameter(TableMatcher(BuiltIns['Table']))
AnyParam = Parameter(AnyMatcher())
NormalBinopPattern = ArgsMatcher(NormalParam, NormalParam)
AnyBinopPattern = ArgsMatcher(AnyParam, AnyParam)
AnyPlusPattern = ArgsMatcher(Parameter(AnyMatcher(), quantifier="+"))
AnyPattern = ArgsMatcher(Parameter(AnyMatcher(), quantifier="*"))

# PositiveIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value > 0)))
# NegativeIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value < 0)))
# NonZeroIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value != 0)))
# OneIndexList = Parameter(TableMatcher(BuiltIns['List'],
#                                       guard=lambda x: py_value(len(x.value) == 1 and
#                                                                NonZeroIntParam.match_score(x.value[0]))))



BuiltIns['bool'].assign_option(ArgsMatcher(AnyParam), lambda x: py_value(bool(x.value)))
BuiltIns['num'].assign_option(ArgsMatcher(BoolParam), lambda x: py_value(int(x.value)))
BuiltIns['num'].assign_option(ArgsMatcher(NumericParam), lambda x: py_value(x.value))
BuiltIns['num'].assign_option(ArgsMatcher(StringParam), lambda x: py_value(read_number(x.value, Context.settings['base'])))
BuiltIns['num'].assign_option(ArgsMatcher(StringParam, IntegralParam),
                              lambda x, b: py_value(read_number(x.value, b.value)))
# Function({ArgsMatcher(BoolParam): lambda x: py_value(int(x.value)),
#                               ArgsMatcher(NumericParam): lambda x: py_value(x.value),
#                                ArgsMatcher(StringParam): lambda x: py_value(read_number(x.value, Context.settings['base'])),
#                                ArgsMatcher(StringParam, IntegralParam): lambda x, b: py_value(read_number(x.value, b.value))},
#                               name='number')
BuiltIns['int'].assign_option(ArgsMatcher(NormalParam), lambda x: py_value(int(BuiltIns['num'].call(x).value)))
BuiltIns['ratio'].assign_option(ArgsMatcher(NormalParam), lambda x: py_value(Fraction(BuiltIns['num'].call(x).value)))
BuiltIns['float'].assign_option(ArgsMatcher(NormalParam), lambda x: py_value(float(BuiltIns['num'].call(x).value)))
# BuiltIns['float'] = Function({ArgsMatcher(NormalParam), lambda x: py_value(float(BuiltIns['number'].call(x).value)))
BuiltIns['str'].assign_option(ArgsMatcher(AnyParam), lambda x: x.to_string())
BuiltIns['str'].assign_option(Option(StringParam, lambda x: x))
BuiltIns['str'].assign_option(ArgsMatcher(NumericParam, IntegralParam),
                              lambda n, b: py_value(write_number(n.value, b.value)))
BuiltIns['list'].assign_option(ArgsMatcher(SeqParam), lambda x: py_value(list(x.value)))
BuiltIns['tuple'].assign_option(ArgsMatcher(SeqParam), lambda x: py_value(tuple(x.value)))
BuiltIns['set'].assign_option(ArgsMatcher(SeqParam), lambda x: py_value(set(x.value)))
BuiltIns['iter'].assign_option(Option(UnionMatcher(*(TraitMatcher(BuiltIns[t])
                                                     for t in ('tuple', 'list', 'set', 'frozenset', 'str'))),
                                      lambda x: x))


bases = {'b': 2, 't': 3, 'q': 4, 'p': 5, 'h': 6, 's': 7, 'o': 8, 'n': 9, 'd': 10}
def setting_set(prop: str, val: PyValue):
    val = val.value
    if prop == 'base' and isinstance(val, str):
        if val not in bases:
            raise RuntimeErr(f'Line {Context.line}: {val} is not a valid base.  Valid base symbols are the following:\n'
                             f"b: 2\nt: 3\nq: 4\np: 5\nh: 6\ns: 7\no: 8\nn: 9\nd: 10")
        val = bases[val[0]]
    Context.settings[prop] = val
    return BuiltIns['settings']
# def setting_get(prop: str):
#     if prop == 'base':
#         return py_value("_ubtqphsond"[Context.settings['base']])
#     return py_value(Context.settings[prop])


# SettingsTable = ListTable(fields=[Slot('set', TableMatcher(BuiltIns['Function'])),
#                                   Slot('get', TableMatcher(BuiltIns['Function']))]
#                           )
# BuiltIns['settings'] = Record(SettingsTable,
#                               set=Function({ArgsMatcher(StringParam, AnyParam): setting_set}),
#                               get=Function({ArgsMatcher(StringParam): setting_get}))
get_base_fn = Function({AnyParam: lambda _: py_value("_ubtqphsond"[Context.settings['base']])},
                       name='get_base_fn')
set_base_fn = Function({ArgsMatcher(AnyParam, Parameter(UnionMatcher(TraitMatcher(StrTrait),
                                                                     *(ValueMatcher(py_value(v)) for v in range(1, 11))))):
                        lambda _, val: setting_set('base', val)},
                       name='set_base_fn')
get_sort_options = Function({AnyParam: lambda: py_value(Context.settings['sort_options'])},
                            name='get_sort_options')
set_sort_options = Function({ArgsMatcher(AnyParam, BoolParam):
                                 lambda _, val: setting_set('sort_options', val)},
                            name='set_sort_options')
SettingsSingletonTable = SetTable(Trait({},
                                     Formula('base', TraitMatcher(IntTrait), get_base_fn),
                                     Setter('base', set_base_fn),
                                     Formula('sort_options', TraitMatcher(BuiltIns['bool']), get_sort_options),
                                     Setter('sort_options', set_sort_options)),
                               name="SettingsSingletonTable")
BuiltIns['settings'] = Record(SettingsSingletonTable)
# BuiltIns['settings'] = Function({},
#                                 Formula('base', TraitMatcher(IntTrait), get_base_fn),
#                                 Setter('base', set_base_fn),
#                                 Formula('sort_options', TraitMatcher(BuiltIns['bool']), get_sort_options),
#                                 Setter('sort_options', set_sort_options)
#                                 )

def key_to_param_set(key: PyValue) -> ArgsMatcher:
    if hasattr(key, 'value') and isinstance(key.value, list):
        vals = key.value
    else:
        vals = [key]
    params = (Parameter(ValueMatcher(pval, pval.value if isinstance(pval.value, str) else None))
              for pval in vals)
    return ArgsMatcher(*params)

# def set_value(fn: Record, key: ArgsMatcher, val: Record):
#     match key:
#         case PyValue(value=str() as name):
#             pass
#         case List(records=records) | PyValue(value=tuple() as records):
#             pass


if 'set' in BuiltIns:
    print("set already a defined builtin")
else:
    BuiltIns['set'] = Function({ArgsMatcher(FunctionParam, AnyParam, AnyParam):
                               lambda fn, key, val: fn.assign_option(key_to_param_set(key), val).resolution})
# BuiltIns['string'].add_option(ArgsMatcher(ListParam), lambda l: py_value(str(l.value[1:])))
# BuiltIns['string'].add_option(ArgsMatcher(NumberParam),
#                               lambda n: py_value('-' * (n.value < 0) +
#                                               base(abs(n.value), 10, 6, string=True, recurring=False)))
# BuiltIns['string'].add_option(ArgsMatcher(Parameter(TableMatcher(BuiltIns["Type"]))), lambda t: py_value(t.value.name))

BuiltIns['type'] = Function({AnyParam: lambda v: v.table})

BuiltIns['len'] = Function({SeqParam: lambda s: py_value(len(s.value)),
                            PatternParam: lambda p: py_value(len(p)),
                            ArgsMatcher(Parameter(TableMatcher(BuiltIns['Table']))): lambda t: py_value(len(t.records))
                            })
# BuiltIns['traits'] = Function({Parameter(BuiltIns['Table']): lambda t: py_value(t.traits)})
# # BuiltIns['contains'] = Function({ArgsMatcher(FunctionParam: AnyParam),
# #                                 lambda a, b: py_value(b in (opt.value for opt in a.options)))
# # BuiltIns['options'] = Function({AnyParam: lambda x: piliize([py_value(lp.pattern) for lp in x.options])})
# # BuiltIns['names'] = Function({AnyParam: lambda x: piliize([py_value(k) for k in x.named_options.keys()])})
# # BuiltIns['keys'] = Function({AnyParam:
# #                             lambda x: piliize([lp.pattern[0].pattern.value for lp in x.options
# #                                                if len(lp.pattern) == 1 and isinstance(lp.pattern[0].pattern, ValueMatcher)])})
#
# BuiltIns['max'] = Function({Parameter(BuiltIns["num"], quantifier='+'):
#                                 lambda *args: py_value(max(*[arg.value for arg in args])),
#                             Parameter(TraitMatcher(BuiltIns["str"]), quantifier='+'):
#                                 lambda *args: py_value(max(*[arg.value for arg in args])),
#                             IterParam:
#                                 lambda ls: py_value(max(*[arg.value for arg in ls.value]))
#                             })
# BuiltIns['min'] = Function({Parameter(BuiltIns["num"], quantifier='+'):
#                                 lambda *args: py_value(min(*[arg.value for arg in args])),
#                             Parameter(TraitMatcher(BuiltIns["str"]), quantifier='+'):
#                                 lambda *args: py_value(min(arg.value for arg in args)),
#                             IterParam: lambda ls: py_value(min(arg.value for arg in ls.value))
#                             })
BuiltIns['abs'] = Function({NumericParam: lambda n: py_value(abs(n.value))})
def round_to_rational(num, places):
    num, places = num.value, places.value
    power = Context.settings['base'] ** places
    return py_value(Fraction(round(num * power), power))


BuiltIns['round'] = Function({NumericParam: lambda n: py_value(round(n.value)),
                             ArgsMatcher(NumericParam, IntegralParam): round_to_rational})

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
# BuiltIns['map'] = Function({ArgsMatcher(SeqParam, FunctionParam): lambda ls, fn: piliize([fn.call(val) for val in ls.value]),
#                             ArgsMatcher(FunctionParam, SeqParam): lambda fn, ls: piliize([fn.call(val) for val in ls.value])})
# BuiltIns['filter'] = Function({ArgsMatcher(ListParam, FunctionParam):
#                                    lambda ls, fn: piliize([v for v in ls.value
#                                                            if BuiltIns['bool'].call(fn.call(v)).value]),
#                                ArgsMatcher(FunctionParam, ListParam):
#                                    lambda fn, ls: piliize([v for v in ls.value
#                                                            if BuiltIns['bool'].call(fn.call(v)).value])})
# BuiltIns['sum'] = Function({SeqParam: lambda ls: BuiltIns['+'].call(*ls.value)})
# BuiltIns['trim'] = Function({StringParam: lambda text: py_value(text.value.strip()),
#                             ArgsMatcher(StringParam, StringParam): lambda t, c: py_value(t.value.strip(c.value))})
# BuiltIns['upper'] = Function({StringParam: lambda text: py_value(text.value.upper())})
# BuiltIns['lower'] = Function({StringParam: lambda text: py_value(text.value.lower())})
# BuiltIns['match'] = Function({ArgsMatcher(StringParam, StringParam):
#                                   lambda s, p: py_value(re.match(p.value, s.value)),
#                               ArgsMatcher(StringParam, StringParam, StringParam):
#                                   lambda s, p, f: py_value(re.match(p.value, s.value, f.value))})
# # BuiltIns['self'] = lambda: Context.env.caller or Context.env or py_value(None)
# # def Args(fn: Function):
# #     arg_list = piliize([opt.value for opt in fn.args])
# #     arg_list.add_option(FunctionParam, lambda fn: Args(fn))
# #     return arg_list
# # BuiltIns['args'] = lambda: Args(Context.env)
#
def list_get(args: Args):
    seq = Context.env.caller
    try:
        seq = seq.value  # noqa
    except AttributeError:
        raise TypeErr(f"Line {Context.line}: Could not find sequence value of non PyValue {seq}")
    match args:
        case Args(positional_arguments=(PyValue() as index,)):
            pass
        case Args(named_arguments={'index': PyValue() as index}):
            pass
        case _:
            raise AssertionError
    try:
        if isinstance(seq, str):
            return py_value(seq[index])
        return seq[index]
    except IndexError as e:
        raise KeyErr(f"Line {Context.line}: {e}")
    except TypeError as e:
        if index.value is None:
            raise KeyErr(f"Line {Context.line}: Pili sequence indices start at 1, not 0.")
        raise KeyErr(f"Line {Context.line}: {e}")

def list_set(ls: List, index: PyValue, val: Function):
    i = index.value[0].value
    i -= i > 0
    if i == len(ls.value):
        ls.value.append(val)
    else:
        ls.value[i] = val
    return val

def list_slice(seq: PyValue, start: PyValue[int], end: PyValue[int], step: PyValue[int] = py_value(1)):
    try:
        seq = seq.value  # noqa
    except AttributeError:
        raise TypeErr(f"Line {Context.line}: Could not find sequence value of non PyValue {seq}")
    start = start.__index__() if start.value else None
    step = step.value
    if step > 0:
        end = end.value + (end.value < 0) or None
    else:
        end = end.value - (end.value > 0) or None
    try:
        return py_value(seq[start:end:step])
    except ValueError as e:
        raise KeyErr(f"Line {Context.line}: {e}")

def list_slice(args: Args):
    seq = Context.env.caller
    try:
        seq = seq.value  # noqa
    except AttributeError:
        raise TypeErr(f"Line {Context.line}: Could not find sequence value of non PyValue {seq}")
    match args:
        case Args(positional_arguments=(start, end, step)):
            step = step.value
        case Args(positional_arguments=(start, end)):
            step = 1
        case _:
            raise ValueError("improper slice args")
    start = start.__index__() if start.value else None
    if step > 0:
        end = end.value + (end.value < 0) or None
    else:
        end = end.value - (end.value > 0) or None
    try:
        return py_value(seq[start:end:step])
    except ValueError as e:
        raise KeyErr(f"Line {Context.line}: {e}")


BuiltIns['slice'] = Function({ArgsMatcher(SeqParam, IntegralParam, IntegralParam): list_slice,
                              ArgsMatcher(SeqParam, IntegralParam, IntegralParam, IntegralParam): list_slice})
list_get_option = Option(ArgsMatcher(IntegralParam), Native(list_get))
list_slice_option1 = Option(ArgsMatcher(IntegralParam, IntegralParam), Native(list_slice))
list_slice_option2 = Option(ArgsMatcher(IntegralParam, IntegralParam, IntegralParam), Native(list_slice))

for tbl in ('List', 'Tuple', 'String'):
    BuiltIns[tbl].catalog.assign_option(list_get_option)
    BuiltIns[tbl].catalog.assign_option(list_slice_option1)
    BuiltIns[tbl].catalog.assign_option(list_slice_option2)

BuiltIns['push'] = Function({ArgsMatcher(ListParam, AnyParam):
                             lambda ls, item: ls.value.append(item)})
BuiltIns['join'] = Function({ArgsMatcher(SeqParam, StringParam):
                             lambda ls, sep: py_value(sep.value.join(BuiltIns['str'].call(item).value
                                                                     for item in iter(ls))),
                             ArgsMatcher(StringParam, Parameter(AnyMatcher(), quantifier="+")):
                             lambda sep, items: py_value(sep.value.join(BuiltIns['str'].call(item).value
                                                                        for item in iter(items))),
                             })
BuiltIns['split'] = Function({ArgsMatcher(StringParam, StringParam):
                                  lambda txt, sep: piliize([py_value(s) for s in txt.value.split(sep.value)])})

BuiltIns['new'] = Function({ArgsMatcher(TableParam, AnyPattern[0]): lambda t, *args, **kwargs: Record(t, *args, **kwargs)})
#
# SeqTrait.options.append(Option(IntegralParam, Native(list_get)))
# SeqTrait.options.append(Option(AnyPlusPattern,
#                                Native(lambda args: BuiltIns['slice'](Args(Context.env.caller) + args))))
# BuiltIns['List'].integrate_traits()
# BuiltIns['Tuple'].integrate_traits()
#
# def convert(name: str) -> Function:
#     o = object()
#     # py_fn = getattr(__builtins__, name, o)
#     py_fn = __builtins__.get(name, o)  # noqa
#     if py_fn is o:
#         raise SyntaxErr(f"Name '{name}' not found.")
#     # Context.root.add_option(ArgsMatcher(Parameter(name)), lambda *args: py_value(py_fn((arg.value for arg in args))))
#     # def lambda_fn(*args):
#     #     arg_list = list(arg.value for arg in args)
#     #     return py_value(py_fn(*arg_list))
#     # return Function({AnyPattern: lambda_fn)
#     return Function({AnyPattern: lambda *args: py_value(py_fn(*(arg.value for arg in args)))})
#
#
# # BuiltIns['python'] = Function({ArgsMatcher(StringParam): lambda n: convert(n.value))
# BuiltIns['python'] = Function({StringParam: lambda code: py_value(eval(code.value))})  # lambda x: py_value(eval(x.value)))
BuiltIns['python'] = Function({StringParam: None})
#
#
#
#
#
#
#
#
#
#
#
#
# for k, v in BuiltIns.items():
#     match v:
#         case Table() | Function():
#             if v.name is None:
#                 v.name = k
