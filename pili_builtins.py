# import re
# from fractions import Fraction
# # from Syntax import Node, Token, ListNode, TokenType
# from Env import *
# from tables import *
# from patterns import *
# # from Expressions import expressionize, read_number, Expression, py_eval, piliize
from state import Op
from syntax import default_op_fn
from utils import read_number, write_number, BuiltIns, BASES, KeyErr
from interpreter import *

print(f'loading {__name__}.py')

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
BuiltIns['index'] = IndexTrait = Trait(name='index')

# Collection Traits
BuiltIns['iter'] = IterTrait = Trait(name='iter')
BuiltIns['seq'] = SeqTrait = Trait(name='seq')
BuiltIns['str'] = StrTrait = Trait(name='str')
BuiltIns['tuple'] = TupTrait = Trait(name='tuple')
BuiltIns['set'] = SetTrait = Trait(name='set')
BuiltIns['frozenset'] = FrozenSetTrait = Trait(name='frozenset')
BuiltIns['list'] = ListTrait = Trait(name='list')
BuiltIns['dict'] = DictTrait = Trait(name='dict')
BuiltIns['range'] = RangeTrait = Trait(name='range')

""" 
TABLES 
"""
# Numeric Tables
BuiltIns['Bool'] = SetTable(BuiltIns['bool'], IntTrait, RatioTrait, NumTrait, IndexTrait, name='Bool')
BuiltIns['false'] = PyValue(BuiltIns['Bool'], False)
BuiltIns['true'] = PyValue(BuiltIns['Bool'], True)

BuiltIns['Integer'] = VirtTable(IntTrait, RatioTrait, NumTrait, IndexTrait, name='Integer')
BuiltIns['Fraction'] = VirtTable(RatioTrait, NumTrait, name='Fraction')
BuiltIns['Float'] = VirtTable(BuiltIns['float'], NumTrait, name='Float')

# Collection Tables
BuiltIns['String'] = VirtTable(StrTrait, SeqTrait, IterTrait, name='String')
BuiltIns['Tuple'] = VirtTable(TupTrait, SeqTrait, IterTrait, name='Tuple')
BuiltIns['Set'] = VirtTable(SetTrait, IterTrait, name='Set')
BuiltIns['List'] = VirtTable(ListTrait, SeqTrait, IterTrait, name='List')
BuiltIns['Dictionary'] = VirtTable(DictTrait, IterTrait, name='Dictionary')
BuiltIns['Range'] = VirtTable(RangeTrait, IterTrait, IndexTrait, name='Range')
BuiltIns['Args'] = VirtTable(SeqTrait, DictTrait, IterTrait, name='Args')
for rec in IntermediateArgsTable.records:
    rec.table = BuiltIns['Args']

BuiltIns['fn'] = FuncTrait = Trait(name='fn')
BuiltIns['Function'] = ListTable(FuncTrait, name='Function')

TableTable.traits += (FuncTrait,)
BuiltIns['Trait'].traits += (FuncTrait,)

BuiltIns['Pattern'] = ListTable(SeqTrait, IterTrait, name='Pattern')
BuiltIns['Pattern'].records = IntermediatePatternTable.records
# BuiltIns["Pattern"] = BuiltIns['ParamSet']
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

BuiltIns['range'].fields = [Slot('start', UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank']))),
                            Slot('end', UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank']))),
                            Slot('step', UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank']))),]

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
# NonStr = TraitMatcher(StrTrait)
# NonStr.invert = 1
NonStr = NotMatcher(TraitMatcher(StrTrait))
NonStrSeqParam = Parameter(IntersectionMatcher(TraitMatcher(SeqTrait), NonStr))
IterParam = Parameter(TraitMatcher(IterTrait))
# TypeParam = Parameter(TableMatcher(BuiltIns["Type"]))
PatternParam = Parameter(TableMatcher(BuiltIns['Pattern']))
FunctionParam = Parameter(TraitMatcher(BuiltIns["fn"]))
TableParam = Parameter(TableMatcher(BuiltIns['Table']))
AnyParam = Parameter(AnyMatcher())
ArgsParam = Parameter(ArgsMatcher())
NormalBinopPattern = ParamSet(NormalParam, NormalParam)
AnyBinopPattern = ParamSet(AnyParam, AnyParam)
AnyPlusPattern = ParamSet(Parameter(AnyMatcher(), quantifier="+"))
AnyPattern = ParamSet(Parameter(AnyMatcher(), quantifier="*"))

# PositiveIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value > 0)))
# NegativeIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value < 0)))
# NonZeroIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value != 0)))
# OneIndexList = Parameter(TableMatcher(BuiltIns['List'],
#                                       guard=lambda x: py_value(len(x.value) == 1 and
#                                                                NonZeroIntParam.match_score(x.value[0]))))


BuiltIns['bool'].assign_option(ParamSet(AnyParam), lambda x: py_value(bool(x.value)))
BuiltIns['num'].assign_option(ParamSet(BoolParam), lambda x: py_value(int(x.value)))
BuiltIns['num'].assign_option(ParamSet(NumericParam), lambda x: py_value(x.value))
BuiltIns['num'].assign_option(ParamSet(StringParam), lambda x: py_value(read_number(x.value, state.settings['base'])))
BuiltIns['num'].assign_option(ParamSet(StringParam, IntegralParam),
                              lambda x, b: py_value(read_number(x.value, b.value)))
# Function({ParamSet(BoolParam): lambda x: py_value(int(x.value)),
#                               ParamSet(NumericParam): lambda x: py_value(x.value),
#                                ParamSet(StringParam): lambda x: py_value(read_number(x.value, state.settings['base'])),
#                                ParamSet(StringParam, IntegralParam): lambda x, b: py_value(read_number(x.value, b.value))},
#                               name='number')
BuiltIns['int'].assign_option(ParamSet(NormalParam), lambda x: py_value(int(BuiltIns['num'].call(x).value)))
BuiltIns['ratio'].assign_option(ParamSet(NormalParam), lambda x: py_value(Fraction(BuiltIns['num'].call(x).value)))
BuiltIns['float'].assign_option(ParamSet(NormalParam), lambda x: py_value(float(BuiltIns['num'].call(x).value)))
# BuiltIns['float'] = Function({ParamSet(NormalParam), lambda x: py_value(float(BuiltIns['number'].call(x).value)))
BuiltIns['str'].assign_option(ParamSet(AnyParam), lambda x: x.to_string())
BuiltIns['str'].assign_option(Option(StringParam, lambda x: x))
BuiltIns['str'].assign_option(ParamSet(NumericParam, IntegralParam),
                              lambda n, b: py_value(write_number(n.value, b.value)))
BuiltIns['list'].assign_option(ParamSet(SeqParam), lambda x: py_value(list(x.value)))
BuiltIns['list'].assign_option(ParamSet(IterParam), lambda x: py_value(list(x)))
BuiltIns['tuple'].assign_option(ParamSet(SeqParam), lambda x: py_value(tuple(x.value)))
BuiltIns['set'].assign_option(ParamSet(SeqParam), lambda x: py_value(set(x.value)))
BuiltIns['iter'].assign_option(Option(UnionMatcher(*(TraitMatcher(BuiltIns[t])
                                                     for t in ('tuple', 'list', 'set', 'frozenset', 'str', 'range'))),
                                      lambda x: x))

def make_custom_iter(rec: Record):
    while not dot_fn(rec, py_value('done')).value:
        yield dot_fn(rec, py_value('next'))
BuiltIns['iter'].assign_option(ParamSet(Parameter(FieldMatcher((), dict(done=AnyMatcher(), next=AnyMatcher())))), make_custom_iter)


def setting_set(prop: str, val: PyValue):
    val = val.value
    if prop == 'base' and isinstance(val, str):
        if val not in BASES:
            raise RuntimeErr(f'Line {state.line}: {val} is not a valid base.  Valid base symbols are the following:\n'
                             f"b: 2\nt: 3\nq: 4\np: 5\nh: 6\ns: 7\no: 8\nn: 9\nd: 10")
        val = BASES[val[0]]
    state.settings[prop] = val
    return BuiltIns['settings']
# def setting_get(prop: str):
#     if prop == 'base':
#         return py_value("_ubtqphsond"[state.settings['base']])
#     return py_value(state.settings[prop])


# SettingsTable = ListTable(fields=[Slot('set', TableMatcher(BuiltIns['Function'])),
#                                   Slot('get', TableMatcher(BuiltIns['Function']))]
#                           )
# BuiltIns['settings'] = Record(SettingsTable,
#                               set=Function({ParamSet(StringParam, AnyParam): setting_set}),
#                               get=Function({ParamSet(StringParam): setting_get}))
get_base_fn = Function({AnyParam: lambda _: py_value("_ubtqphsond"[state.settings['base']])},
                       name='get_base_fn')
set_base_fn = Function({ParamSet(AnyParam, Parameter(UnionMatcher(TraitMatcher(StrTrait),
                                                                  *(ValueMatcher(py_value(v)) for v in range(1, 11))))):
                        lambda _, val: setting_set('base', val)},
                       name='set_base_fn')
get_sort_options = Function({AnyParam: lambda: py_value(state.settings['sort_options'])},
                            name='get_sort_options')
set_sort_options = Function({ParamSet(AnyParam, BoolParam):
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

# def key_to_param_set(key: PyValue) -> ParamSet:
#     if hasattr(key, 'value') and isinstance(key.value, list):
#         vals = key.value
#     else:
#         vals = [key]
#     params = (Parameter(ValueMatcher(pval, pval.value if isinstance(pval.value, str) else None))
#               for pval in vals)
#     return ParamSet(*params)


BuiltIns['type'] = Function({AnyParam: lambda v: v.table})

BuiltIns['len'] = Function({SeqParam: lambda s: py_value(len(s.value)),
                            PatternParam: lambda p: py_value(len(p)),
                            FunctionParam: lambda fn: py_value(len(fn.op_map)),
                            ParamSet(Parameter(TableMatcher(BuiltIns['Table']))): lambda t: py_value(len(t.records))
                            })
# BuiltIns['traits'] = Function({Parameter(BuiltIns['Table']): lambda t: py_value(t.traits)})
# # BuiltIns['contains'] = Function({ParamSet(FunctionParam: AnyParam),
# #                                 lambda a, b: py_value(b in (opt.value for opt in a.options)))
# # BuiltIns['options'] = Function({AnyParam: lambda x: piliize([py_value(lp.pattern) for lp in x.options])})
# # BuiltIns['names'] = Function({AnyParam: lambda x: piliize([py_value(k) for k in x.named_options.keys()])})
BuiltIns['keys'] = Function({FunctionParam: lambda fn: py_value({k.positional_arguments for k in fn.op_map.keys()})})
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
    power = state.settings['base'] ** places
    return py_value(Fraction(round(num * power), power))


BuiltIns['round'] = Function({NumericParam: lambda n: py_value(round(n.value)),
                              ParamSet(NumericParam, IntegralParam): round_to_rational})

# def inclusive_range(*args: PyValue):
#     step = 1
#     match len(args):
#         case 0:
#             return piliize([])
#         case 1:
#             stop = args[0].value
#             if stop == 0:
#                 return piliize([])
#             elif stop > 0:
#                 start = 1
#             else:
#                 start = step = -1
#         case 2:
#             start = args[0].value
#             stop = args[1].value
#             step = stop > start or -1
#         case 3:
#             start, stop, step = (a.value for a in args)
#             if not step:
#                 raise RuntimeErr(f"Line {state.line}: Third argument in range (step) cannot be 0.")
#         case _:
#             raise RuntimeErr(f"Line {state.line}: Too many arguments for range")
#     return Range()
#     i = start
#     ls = []
#     while step > 0 and i <= stop or step < 0 and i >= stop:
#         ls.append(py_value(i))
#         i += step
#     return piliize(ls)


RangeTrait.assign_option(Parameter(BuiltIns["num"], quantifier="*"), lambda *args: Range(*args))
# BuiltIns['map'] = Function({ParamSet(SeqParam, FunctionParam): lambda ls, fn: piliize([fn.call(val) for val in ls.value]),
#                             ParamSet(FunctionParam, SeqParam): lambda fn, ls: piliize([fn.call(val) for val in ls.value])})
# BuiltIns['filter'] = Function({ParamSet(ListParam, FunctionParam):
#                                    lambda ls, fn: piliize([v for v in ls.value
#                                                            if BuiltIns['bool'].call(fn.call(v)).value]),
#                                ParamSet(FunctionParam, ListParam):
#                                    lambda fn, ls: piliize([v for v in ls.value
#                                                            if BuiltIns['bool'].call(fn.call(v)).value])})
# BuiltIns['sum'] = Function({SeqParam: lambda ls: BuiltIns['+'].call(*ls.value)})
BuiltIns['trim'] = Function({StringParam: lambda text: py_value(text.value.strip()),
                            ParamSet(StringParam, StringParam): lambda t, c: py_value(t.value.strip(c.value))})
BuiltIns['upper'] = Function({StringParam: lambda text: py_value(text.value.upper())})
BuiltIns['lower'] = Function({StringParam: lambda text: py_value(text.value.lower())})
BuiltIns['match'] = Function({ParamSet(StringParam, StringParam):
                                  lambda s, p: py_value(re.match(p.value, s.value)),
                              ParamSet(StringParam, StringParam, StringParam):
                                  lambda s, p, f: py_value(re.match(p.value, s.value, f.value))})
# # BuiltIns['self'] = lambda: state.env.caller or state.env or py_value(None)
# # def Args(fn: Function):
# #     arg_list = piliize([opt.value for opt in fn.args])
# #     arg_list.add_option(FunctionParam, lambda fn: Args(fn))
# #     return arg_list
# # BuiltIns['args'] = lambda: Args(state.env)
#
def list_get(args: Args):
    seq = state.env.caller
    try:
        seq = seq.value  # noqa
    except AttributeError:
        raise TypeErr(f"Line {state.line}: Could not find sequence value of non PyValue {seq}")
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
        raise KeyErr(f"Line {state.line}: {e}")
    except TypeError as e:
        if index.value is None:
            raise KeyErr(f"Line {state.line}: Pili sequence indices start at 1, not 0.")
        raise KeyErr(f"Line {state.line}: {e}")

# moved to PyValue.assign_option
def list_set(ls: PyValue[list], index: PyValue, val: Record):
    if index.value == len(ls.value) + 1:
        ls.value.append(val)
    else:
        ls.value[index] = val
    return val

# def list_slice(seq: PyValue, start: PyValue[int], end: PyValue[int], step: PyValue[int] = py_value(1)):
#     try:
#         seq = seq.value  # noqa
#     except AttributeError:
#         raise TypeErr(f"Line {state.line}: Could not find sequence value of non PyValue {seq}")
#     start = start.__index__() if start.value else None
#     step = step.value
#     if step > 0:
#         end = end.value + (end.value < 0) or None
#     else:
#         end = end.value - (end.value > 0) or None
#     try:
#         return py_value(seq[start:end:step])
#     except ValueError as e:
#         raise KeyErr(f"Line {state.line}: {e}")

def list_slice(args: Args):
    seq = state.env.caller
    try:
        seq = seq.value  # noqa
    except AttributeError:
        raise TypeErr(f"Line {state.line}: Could not find sequence value of non PyValue {seq}")
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
        raise KeyErr(f"Line {state.line}: {e}")


BuiltIns['slice'] = Function({ParamSet(SeqParam, IntegralParam, IntegralParam): list_slice,
                              ParamSet(SeqParam, IntegralParam, IntegralParam, IntegralParam): list_slice})
list_get_option = Option(ParamSet(IntegralParam), Native(list_get))
list_slice_option1 = Option(ParamSet(IntegralParam, IntegralParam), Native(list_slice))
list_slice_option2 = Option(ParamSet(IntegralParam, IntegralParam, IntegralParam), Native(list_slice))

# for tbl in ('List', 'Tuple', 'String'):
#     BuiltIns[tbl].catalog.assign_option(list_get_option)
#     BuiltIns[tbl].catalog.assign_option(list_slice_option1)
#     BuiltIns[tbl].catalog.assign_option(list_slice_option2)

SeqTrait.assign_option(list_get_option)
SeqTrait.assign_option(list_slice_option1)
SeqTrait.assign_option(list_slice_option2)

BuiltIns['push'] = Function({ParamSet(ListParam, AnyParam):
                             lambda ls, item: ls.value.append(item) or ls})
BuiltIns['pop'] = Function({ParamSet(ListParam, Parameter(IntegralParam, None, '?', py_value(-1))):
                            lambda ls, idx=-1: ls.value.pop(idx - (idx > 0))})
BuiltIns['join'] = Function({ParamSet(SeqParam, StringParam):
                             lambda ls, sep: py_value(sep.value.join(BuiltIns['str'].call(item).value
                                                                     for item in iter(ls))),
                             ParamSet(StringParam, Parameter(AnyMatcher(), quantifier="+")):
                             lambda sep, items: py_value(sep.value.join(BuiltIns['str'].call(item).value
                                                                        for item in iter(items))),
                             })
BuiltIns['split'] = Function({ParamSet(StringParam, StringParam):
                                  lambda txt, sep: py_value([py_value(s) for s in txt.value.split(sep.value)])})

BuiltIns['new'] = Function({ParamSet(TableParam, AnyPattern[0], kwargs='kwargs'):
                                lambda t, *args, **kwargs: Record(t, *args, **kwargs)})

# class File(Record):


# def convert(name: str) -> Function:
#     o = object()
#     # py_fn = getattr(__builtins__, name, o)
#     py_fn = __builtins__.get(name, o)  # noqa
#     if py_fn is o:
#         raise SyntaxErr(f"Name '{name}' not found.")
#     # state.root.add_option(ParamSet(Parameter(name)), lambda *args: py_value(py_fn((arg.value for arg in args))))
#     # def lambda_fn(*args):
#     #     arg_list = list(arg.value for arg in args)
#     #     return py_value(py_fn(*arg_list))
#     # return Function({AnyPattern: lambda_fn)
#     return Function({AnyPattern: lambda *args: py_value(py_fn(*(arg.value for arg in args)))})
#
#


for tbl in TableTable.records:
    tbl: Table
    tbl.integrate_traits()


def make_flags(*names: str) -> dict[str, Parameter]:
    return {n: Parameter(BoolParam, n, '?', BuiltIns['blank']) for n in names}


BuiltIns['python'] = Function({ParamSet(StringParam, named_params=make_flags('direct', 'execute')):
                                   run_python_code}, name='python')  # function filled in in syntax.py

"""
Files
"""
# BuiltIns['file'] = Trait({}, Slot('path', TraitMatcher(StrTrait)), name='file')
# BuiltIns['File'] = ListTable(BuiltIns['file'], name='File')
#
# def read_file(arg: Record, lines=BuiltIns['blank']):
#     with open(arg.get('path').value, 'r') as f:
#         if lines == BuiltIns['true']:
#             return py_value(f.readlines())
#         else:
#             return py_value(f.read())
# BuiltIns['read'] = Function({ParamSet(Parameter(TraitMatcher(BuiltIns['file'])),
#                                       named_params={'lines': Parameter(BoolParam, 'lines', '?')}):
#                                  read_file}, name='read')

""" 
Regular Expressions
"""
def regex_extract(regex: RegEx | PyValue[str], text: PyValue[str], *,
                  a=BuiltIns['blank'], i=BuiltIns['blank'], m=BuiltIns['blank'],
                  s=BuiltIns['blank'], x=BuiltIns['blank'], l=BuiltIns['blank']):
    flags = 0
    if a.truthy:
        flags |= re.RegexFlag.ASCII
    if i.truthy:
        flags |= re.RegexFlag.IGNORECASE
    if m.truthy:
        flags |= re.RegexFlag.MULTILINE
    if s.truthy:
        flags |= re.RegexFlag.DOTALL
    if x.truthy:
        flags |= re.RegexFlag.VERBOSE
    if l.truthy:
        flags |= re.RegexFlag.LOCALE
    if not flags and isinstance(regex, RegEx):
        for f in regex.flags:
            flags |= getattr(re.RegexFlag, f.upper())
    return py_value(re.findall(regex.value, text.value, flags))

regex_constructor = {ParamSet(StringParam, Parameter(StringParam, 'flags', '?')):
                         lambda s, f=py_value(''): RegEx(s.value, f.value)}
BuiltIns['regex'] = Trait(regex_constructor, Slot('flags', TraitMatcher(StrTrait),
                                                  default=Function({ParamSet(Parameter(AnyMatcher(), 'self')):
                                                                        py_value('')})),
                          name='regex')
BuiltIns['RegEx'] = ListTable(BuiltIns['regex'], fn_options=regex_constructor, name='RegEx',)
regex_extract_function = Function({ParamSet(Parameter(TableMatcher(BuiltIns['RegEx'])),
                                                        StringParam,
                                                        named_params=make_flags(*'aimsxl')):
                                                   regex_extract}, name='extract')
BuiltIns['regex'].frame = Frame(None, bindings={'extract': regex_extract_function})












########################################################
# Operators
########################################################
Operator.fn = Function({AnyPattern: default_op_fn})

identity = lambda x: x

Op[';'].fn = Function({AnyPlusPattern: lambda *args: args[-1]})

# def eval_assign_args(lhs: Node, rhs: Node) -> Args:
#     # """
#     # IDEA: make the equals sign simply run the pattern-matching algorithm as if calling a function
#     #       that will also bind names — and allow very complex destructuring assignment!
#     # What about assigning values to names of properties and keys?
#     #     eg, `foo.bar = value` and `foo[key]` = value`
#     #     foo[bar.prop]: ...
#     #     foo[5]
#     #     special case it?  It's not like you're gonna see that in a parameter pattern anyway
#     #     0r could actually integrate that behaviour into pattern matching.
#     #     - standalone dotted names will bind to those locations (not local scope)
#     #     - function calls same thing... foo[key] will bind to that location
#     # btw, if I start making more widespread use of patterns like this, I might have to add in a method
#     # to Node to evaluate specifically to patterns.  Node.patternize or Node.eval_as_pattern
#     # """
#     if isinstance(rhs, Block):
#         val = Closure(rhs)
#     else:
#         val = rhs.evaluate()
#
#     match lhs:
#         case Token(type=TokenType.Name, source_text=name):
#             # name = value
#             if isinstance(val, Closure):
#                 raise SyntaxErr(f"Line {state.line}: "
#                                 f"Cannot assign block to a name.  Blocks are only assignable to options.")
#             return Args(py_value(name), val)  # str, any
#         case ListNode():
#             # [key] = value
#             return Args(lhs.evaluate(), val)
#         case OpExpr('.', [Node() as fn_node, Token(type=TokenType.Name, source_text=name)]):
#             # foo.bar = value
#             if isinstance(val, Closure):
#                 raise SyntaxErr("Line {state.line}: "
#                                 "Cannot assign block to a name.  Blocks are only assignable to options.")
#             location = fn_node.evaluate()
#             return Args(location, py_value(name), val)  # any, str, any
#         case OpExpr('.', [Node() as fn_node, ListNode(list_type=ListType.Args) as args]):
#             # foo[key] = value
#             location = fn_node.evaluate()  # note: if location is not a function or list, a custom option must be added to =
#             key: Args = args.evaluate()
#             return Args(location, key, val)  # fn/list, args, any
#         # case OpExpr(',', keys):
#         #     pass
#         case OpExpr(','):
#             raise SyntaxErr(f"Line {lhs.line}: Invalid LHS for assignment.  If you want to assign to a key, use either "
#                             f"`[key1, key2] = value` or `key1, key2: value`")
#         case _:
#             raise SyntaxErr(f"Line {lhs.line}: Invalid lhs for assignment: {lhs}")

def eval_eq_args(lhs: Node, *val_nodes: Node) -> Args:
    values = (Closure(node) if isinstance(node, Block) else node.evaluate() for node in val_nodes)
    match lhs:
        # case Token(TokenType.Name, text=name):
        #     patt = Parameter(AnyMatcher(), name)
        case OpExpr('.', [loc_node, Token(TokenType.Name, text=name)]):
            patt = BindPropertyParam(loc_node.evaluate(), name)
        case OpExpr('[', [loc_node, args]):
            patt = BindKeyParam(loc_node.evaluate(), args.evaluate())
        case _:
            patt = lhs.eval_pattern(name_as_any=True)
    return Args(patt, *values)

# def set_or_assign_option(*args, operation: Function = None):
#     match args:
#         case Function() as fn, Args() as args, Record() as val:
#             if operation:
#                 val = operation.call(fn.call(args), val)
#             fn.assign_option(args, val)
#         case PyValue(value=list()) as ls, Args(positional_arguments=[index]) as args, Record() as val:
#             if operation:
#                 val = operation.call(ls.call(args), val)
#             list_set(ls, index, val)
#         case Record() as rec, PyValue(value=str() as name), Record() as val:
#             if operation:
#                 val = operation.call(rec.get(name), val)
#             rec.set(name, val)
#         case _:
#             raise ValueError("Incorrect value types for assignment")
#     return val

def set_with_fn(operation: Function = None):
    def inner(patt: Pattern, left: Record, right: Record):
        val = operation.call(Args(left, right))
        return patt.match_and_bind(val)
    return inner


Op['='].eval_args = eval_eq_args
# Op['='].fn = Function({ParamSet(StringParam, AnyParam): lambda name, val: state.env.assign(name.value, val),
#                        ParamSet(FunctionParam, AnyParam, AnyParam): set_or_assign_option,
#                        ParamSet(ListParam, AnyParam, AnyParam): set_or_assign_option,
#                        ParamSet(AnyParam, StringParam, AnyParam): set_or_assign_option
#                        }, name='=')
Op['='].fn = Function({ParamSet(PatternParam, AnyParam):
                           lambda patt, val: patt.match_and_bind(val),
                       AnyParam: identity},
                      name='=')

# def null_assign(rec_or_name: Record | PyValue, name_or_val: PyValue | Record, val_or_none: Record = None):
#     if val_or_none is None:
#         name = rec_or_name.value
#         val = name_or_val
#         get = state.deref
#         set = state.env.assign
#     else:
#         rec = rec_or_name
#         name = name_or_val.value
#         val = val_or_none
#         get = rec.get
#         set = rec.set
#     existing_value = get(name, BuiltIns['blank'])
#     if existing_value is BuiltIns['blank']:
#         set(name, val)
#         return val
#     else:
#         return existing_value
#
# def null_assign_rec(rec, name, val):
#     name = name.value
#     existing_value = rec.get(name, BuiltIns['blank'])
#     if existing_value is BuiltIns['blank']:
#         rec.set(name, val)
#         return val
#     else:
#         return existing_value
#
#
# Op['??='].fn = Function({ParamSet(StringParam, AnyParam): null_assign,
#                          ParamSet(FunctionParam, AnyParam, AnyParam):
#                              lambda fn, args, val: fn.assign_option(args, val, no_clobber=True).value
#                                                 or BuiltIns['blank'],
#                          ParamSet(AnyParam, StringParam, AnyParam): null_assign
#                          }, name='??=')
Op['??='].fn = Op['='].fn
def eval_null_assign_args(lhs: Node, rhs: Node) -> Args:
    match lhs:
        case Token(TokenType.Name, text=name):
            existing = state.deref(name, None)
            if existing is not None and existing != BuiltIns['blank']:
                return Args(existing)
            patt = Parameter(AnyMatcher(), name)
        case OpExpr('.', [loc_node, Token(TokenType.Name, text=name)]):
            rec = loc_node.evaluate()
            existing = rec.get(name, None)
            if existing is not None and existing != BuiltIns['blank']:
                return Args(existing)
            patt = BindPropertyParam(rec, name)
        case OpExpr('[', terms):
            rec, args = [t.evaluate() for t in terms]
            exists = BuiltIns['has'].call(Args(rec, args)).value
            existing = lhs.evaluate() if exists else BuiltIns['blank']
            if existing != BuiltIns['blank']:
                return Args(existing)
            patt = BindKeyParam(rec, args)
        case _:
            raise SyntaxErr(f'Line {lhs.line}: "{lhs.source_text}" is invalid syntax for left-hand-side of "??=".')
    return Args(patt, rhs.evaluate())
Op['??='].eval_args = eval_null_assign_args


# def eval_colon_args(lhs: Node, rhs: Node) -> Args:
#     if isinstance(rhs, Block):
#         resolution = Closure(rhs)
#     else:
#         resolution = rhs.evaluate()
#
#     match lhs:
#         case ParamsNode():
#             """ [params]: ... """
#             return Args(lhs.evaluate(), resolution)
#         case OpExpr('.', terms):
#             match terms:
#                 case [OpExpr('.', [Token(type=TokenType.Name, source_text=name)]),
#                       ParamsNode() as params] \
#                       | [Token(params, type=TokenType.Name, source_text=name)]:
#                     """ .foo[params] """
#                     # TODO: this block is ridiculous.  Both the pattern and the logic are too complicated
#                     # How do I make it better?
#                     # - stop using the dot as the function call operator (requires overhaul of lots of pattern-matching)
#                     # - separate back into two blocks: .foo[params]: ... and .foo: ...
#                     # - capture leading dot in AST and transform into dedicated expr type
#                     # - change the way foo.bar[arg] is called — eg:
#                     #   - `foo.bar[arg]` => `bar[foo, arg]` (just like with records of tables)
#                     #   - `foo.bar[arg]` => `bar[arg, self=foo]`
#                     location = state.env.fn
#                     if location is None:
#                         raise EnvironmentError(f"Line {state.line}: illegal .dot option found")
#                     key: ParamSet = params.evaluate() if params else ParamSet()
#                     key.prepend(Parameter(location, 'self'))
#                     fn = state.deref(name)
#                     # if fn is None:
#                     #     fn = Function(name=name)
#                     #     state.env.locals[name] = fn
#                     #     if not isinstance(location, Table | Trait):
#                     #         print(f"WARNING: {name} not yet defined in current scope.  Newly created function is "
#                     #               f" currently only accessible as `{location}.{name}`.")
#                     return Args(fn, key, resolution)
#                 case [fn_node, ParamsNode() as params]:
#                     """ foo[key]: ... """
#                     match fn_node:
#                         case Token(type=TokenType.Name, source_text=name):
#                             location = state.deref(name, None)
#                             if location is None:
#                                 location = Function(name=name)
#                                 state.env.locals[name] = location
#                         case _:
#                             location = fn_node.evaluate()
#                     key = params.evaluate()
#                     return Args(location, key, resolution)  # fn, args, any
#                 case [Token(type=TokenType.Name, source_text=name)]:
#                     """ .foo: ... """
#                     match state.env.fn:
#                         case Table() | Trait() as location:
#                             key = ParamSet(Parameter(location, 'self'))
#                         case Function():
#                             key = ParamSet()
#                         case _:
#                             raise EnvironmentError(f"Line {state.line}: illegal .dot option found")
#                     fn = state.deref(name)
#                     # if fn is None:
#                     #     fn = Function(name=name)
#                     #     state.env.locals[name] = fn
#                     # maybe this should have no self parameter if in Function context?
#                     return Args(fn, key, resolution)
#                 case [table_or_trait, Token(type=TokenType.Name, source_text=name)]:
#                     ''' Foo.bar: ... '''
#                     t = table_or_trait.evaluate()
#                     fn = t.get(name)
#                     match t:
#                         case Table() | Trait():
#                             pattern = ParamSet(Parameter(t, 'self'))
#                         case Function():
#                             pattern = ParamSet()
#                         case None:
#                             raise RuntimeErr(f"Line {lhs.line}: leftmost container term must be a table, trait, or function."
#                                              f"\nIe, for `foo.bar: ...` then foo must be a table, trait, or function.")
#                     return Args(fn, pattern, resolution)
#                 case _:
#                     raise SyntaxErr(f"Line {state.line}: Unrecognized syntax for LHS of assignment.")
#         case OpExpr(',', keys):
#             """ key1, key2: ... """
#             key = Args(*(n.evaluate() for n in keys))
#             return Args(key, resolution)
#         case _:
#             """ key: ... """
#             key = Args(lhs.evaluate())
#             return Args(key, resolution)

def eval_colon_args(lhs: Node, rhs: Node) -> Args:
    if isinstance(rhs, Block):
        resolution = Closure(rhs)
    else:
        resolution = rhs.evaluate()
    match lhs:
        case ParamsNode() as params:
            """ [params]: ... """
            fn = state.env.fn
            return Args(fn, params.evaluate(), resolution)
        case OpExpr('[', [fn_node, ParamsNode() as params]):
            """ foo[params]: ... """
            match fn_node:
                case Token(TokenType.Name, text=name):
                    fn = state.deref(name, None)
                    if fn is None:
                        fn = Function(name=name)
                        state.env.locals[name] = fn
                case _:
                    fn = fn_node.evaluate()
            return Args(fn, params.evaluate(), resolution)  # fn, args, any
        # case OpExpr('.', terms):
        #     # this was replaced in AST
        #     raise NotImplementedError
        # case OpExpr(',', keys):
        #     """ key1, key2: ... """
        #     key = Args(*(n.evaluate() for n in keys))
        #     return Args(key, resolution)
        case _:
            """ key: ... """
            if isinstance(lhs, ListLiteral):
                print(f"SYNTAX WARNING (line {lhs.line}): You used [brackets] in a key-value expression.  If you meant "
                      f"to define a function option, you should indent the function body underneath this.")
            key = Args(lhs.evaluate())
            return Args(key, resolution)

def assign_option(*args):
    match args:
        case fn, pattern, resolution:
            fn.assign_option(pattern, resolution)
        case Args() | ParamSet() as key, resolution:
            if not isinstance(state.env.fn, Function):
                raise EnvironmentError(f"Line {state.line}: Cannot assign key-value option in this context.  "
                                       f"Must be within a definition of a function, table, or trait.")
            state.env.fn.assign_option(key, resolution)
        case _:
            raise RuntimeErr(f"Line {state.line}: wrong arguments for colon function.")
    return BuiltIns['blank']



Op[':'].eval_args = eval_colon_args
Op[':'].fn = Function({AnyPlusPattern: assign_option})


def eval_dot_args(lhs: Node, rhs: Node) -> Args:
    if rhs.type == TokenType.Name:
        right_arg = py_value(rhs.text)
    else:
        right_arg = rhs.evaluate()
    # match lhs, rhs:
    #     case (OpExpr('.' | '.?' | '..',
    #                  [left_term, Token(type=TokenType.Name, source_text=name)]),
    #           ListNode(list_type=ListType.Args) as args_node):
    #         # case left_term.name[args_node]
    #         left = left_term.evaluate()
    #         # 1. Try to resolve slot/formula in left
    #         prop = left.get(name, None)  # , search_table_frame_too=True)
    #         if prop is not None:
    #             return Args(prop, right_arg)
    #         # 2. Try to find function in table
    #         method = left.table.get(name, None)
    #         if method is not None:
    #             return Args(method, right_arg, caller=left)
    #         # 3. Finally, try  to resolve name in scope
    #         fn = state.deref(name, None)
    #         if fn is None:
    #             raise KeyErr(f"Line {state.line}: {left.table} {left} has no slot '{name}' and no record with that "
    #                          f"name found in current scope either.")
    #         return Args(fn, Args(left) + right_arg)
    #     case _:
    #         pass
    return Args(lhs.evaluate(), right_arg)

    # if not TraitMatcher(SeqTrait).match_score(right_arg):
    #     return Args(lhs.evaluate(), right_arg)
    # args = right_arg
    # if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
    #     name = lhs[-1].source_text
    #     a = expressionize(lhs[:-2]).evaluate()
    #     fn = a.get(name, None)
    #     if fn is None:
    #         fn = state.deref(name, None)
    #         if fn is None:
    #             raise KeyErr(f"Line {state.line}: {a.table} {a} has no slot '{name}' and no variable with that name "
    #                          f"found in current scope either.")
    #         if isinstance(args, Args):
    #             args.positional_arguments = (a, *args.positional_arguments)
    #         else:
    #             args = List([a] + args.value)
    # else:
    #     fn = expressionize(lhs).evaluate()
    # return [fn, args]


# I had to import the dot_fn from patterns.py because it needs to be used there for the field matcher
"""
# def dot_fn(a: Record, b: Record, *, caller=None, suppress_error=False):
#     match b:
#         case Args() as args:
#             return a.call(args, caller=caller)
#         case PyValue(value=str() as name):
#             prop = a.get(name, None)
#             if prop is not None:
#                 return prop
#             fn = a.table.get(name, state.deref(name, None))
#             if fn is None:
#                 if suppress_error:
#                     return  # this is for pattern matching
#                 raise MissingNameErr(f"Line {state.line}: {a.table} {a} has no field \"{name}\", "
#                                      f"and also not found as function name in scope.")
#             return fn.call(a, caller=caller)
#         # case PyValue(value=tuple() | list() as args):
#         #     return a.call(*args)
#         case _:
#             print(f"WARNING: Line {state.line}: "
#                   f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
#             return a.call(b)
#     # raise OperatorError(f"Line {state.line}: "
#     #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
"""

def list_get(seq: PyValue, args: Args):
    try:
        seq = seq.value  # noqa
    except AttributeError:
        raise TypeErr(f"Line {state.line}: Could not find sequence value of non PyValue {seq}")
    match args:
        case Args(positional_arguments=(PyValue() as index,)):
            pass
        case Args(named_arguments={'index': PyValue() as index}):
            pass
        case Args(positional_arguments=[Range() as rng]):
            return py_value(seq[rng.slice])
        case _:
            raise AssertionError
    try:
        if isinstance(seq, str):
            return py_value(seq[index])
        return seq[index]
    except IndexError as e:
        raise KeyErr(f"Line {state.line}: {e}")
    except TypeError as e:
        if index.value is None:
            raise KeyErr(f"Line {state.line}: Pili sequence indices start at 1, not 0.")
        raise KeyErr(f"Line {state.line}: {e}")

def extract_pyvalue(rec: Record):
    match rec:
        case PyValue(value=value) | PyObj(obj=value):
            return value
        case _:
            raise TypeErr(f"Line {state.line}: incompatible type for python function: {rec.table}")
def py_dot(a: PyObj, b: Args | PyValue[str]):
    obj = a.obj
    match b:
        case PyValue(value=str() as name):
            return py_value(getattr(obj, name))
        case Args(positional_arguments=args, named_arguments=kwargs, flags=flags):
            kwargs.update(dict(zip(flags, [BuiltIns['true']] * len(flags))))
            return py_value(obj(*map(extract_pyvalue, args), **{k: extract_pyvalue(v) for k, v in kwargs.items()}))
        case _:
            raise Exception
    kwargs = {**args.named_arguments, **dict(zip(args.flags, [BuiltIns['true']] * len(args.flags)))}
    return fn(*args.positional_arguments, **kwargs)


caller_patt = ParamSet(AnyParam, AnyParam, named_params={'caller': Parameter(AnyMatcher(), 'caller', '?')})
# note: caller_patt should be (FunctionParam, ArgsParam), but I just made it any, any for a slight speed boost

Op['.'].fn = Function({caller_patt: dot_fn,
                       StringParam: lambda a: state.deref(a.value),
                       ParamSet(AnyParam, StringParam): dot_fn,
                       ParamSet(SeqParam, ArgsParam): list_get,
                       ParamSet(Parameter(TableMatcher(BuiltIns['PythonObject'])),
                                Parameter(UnionMatcher(TraitMatcher(FuncTrait), TableMatcher(BuiltIns['Table'])))):
                           py_dot,  # I don't remember why the second parameter for the pydot is func|table ???
                       ParamSet(Parameter(TableMatcher(BuiltIns['PythonObject'])), AnyParam): py_dot
                       })

Op['.?'].fn = Function({caller_patt:
                            lambda a, b: BuiltIns['.'].call(a,b)
                                         if BuiltIns['has'].call(a, b).value else py_value(None),
                        ParamSet(StringParam):
                            lambda a: BuiltIns['.'].call(a)
                                      if BuiltIns['has'].call(a).value else py_value(None),
                        })
# def eval_swizzle_args(lhs: Node, rhs: Node) -> Args:
#     if rhs.type is TokenType.Name:
#         rvalue = py_value(rhs.text)
#     else:
#         rvalue = rhs.evaluate()
#     return Args(lhs.evaluate(), rvalue)
Op['..'].fn = Function({ParamSet(SeqParam, StringParam):
                            lambda ls, name: py_value([dot_fn(el, name) for el in ls.value]),
                        ParamSet(SeqParam, FunctionParam):
                            lambda ls, fn: py_value([fn.call(el) for el in ls.value]),
                        ParamSet(Parameter(TraitMatcher(IterTrait)), FunctionParam):
                            lambda it, fn: py_value([fn.call(el) for el in it]),
                        }, name='..')
Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = eval_dot_args

def eval_call_args(lhs: Node, rhs: Node) -> Args:
    args = rhs.evaluate()
    match lhs:
        case OpExpr('.', [loc_node, Token(TokenType.Name, text=name)]):
            # case location.name[args_node]
            location = loc_node.evaluate()
            # 1. Try to resolve slot/formula in left
            prop = location.get(name, None)  # , search_table_frame_too=True)
            if prop is not None:
                return Args(prop, args)
            # 2. Try to find function in table and traits
            for scope in (location.table, *location.table.traits):
                method = scope.get(name, None)
                if method is not None:
                    return Args(method, Args(location) + args)
            # 3. Finally, try  to resolve name normally
            fn = state.deref(name, None)
            if fn is None:
                raise KeyErr(f"Line {state.line}: {location} has no slot '{name}' and no record with that "
                             f"name found in current scope either.")
            return Args(fn, Args(location) + args)
        case _:
            return Args(lhs.evaluate(), args)

def call_py_obj(obj: PyObj, args: Args):
    """ call a python function on an Args object; args must only contain bool, int, float, Fraction, and str"""
    kwargs = {**{k.value: v.value for k, v in args.named_arguments.items()},
              **dict(zip(args.flags, [True] * len(args.flags)))}
    return py_value(obj.obj(*(arg.value for arg in args.positional_arguments), **kwargs))

Op['['].eval_args = eval_call_args
Op['['].fn = Function({ParamSet(AnyParam, ArgsParam): Record.call,  # lambda rec, args: rec.call(args),
                       ParamSet(SeqParam, ArgsParam): list_get,
                       ParamSet(Parameter(TableMatcher(BuiltIns['PythonObject'])), ArgsParam): call_py_obj,
                       }, name='call')
BuiltIns['call'] = Op['['].fn

def eval_right_arrow_args(lhs: Node, rhs: Node):
    resolution = Closure(rhs)
    match lhs:
        case ParamsNode() as params:
            pass
        case OpExpr(',', terms):
            params = ParamsNode(terms, [])
        case OpExpr(';', [lhs, rhs]):
            match lhs:
                case OpExpr(',', terms):
                    ord_params = terms
                case _:
                    ord_params = [lhs]
            match rhs:
                case OpExpr(',', terms):
                    named_params = terms
                case _:
                    named_params = (rhs,)
            params = ParamsNode(ord_params, list(named_params))
        case _:
            params = ParamsNode([lhs], [])
    return Args(params.evaluate(), resolution)


Op['=>'].eval_args = eval_right_arrow_args
Op['=>'].fn = Function({AnyBinopPattern: lambda params, block: Function({params: block})},
                       name='=>')
def eval_comma_args(*nodes) -> Args:
    return Args(*eval_list_nodes(nodes))
Op[','].eval_args = eval_comma_args
Op[','].fn = Function({AnyPlusPattern: lambda *args: py_value(args)},
                      name=',')

def eval_nullish_args(*nodes: Node):
    for node in nodes:
        match node:
            case Token(TokenType.Name, text=name):
                existing = state.deref(name, BuiltIns['blank'])
                if existing != BuiltIns['blank']:
                    return Args(existing)
            case OpExpr('.', [loc_node, Token(TokenType.Name, text=name)]):
                rec = loc_node.evaluate()
                existing = rec.get(name, BuiltIns['blank'])
                if existing != BuiltIns['blank']:
                    return Args(existing)
            case OpExpr('[', terms):
                rec, args = [t.evaluate() for t in terms]
                exists = BuiltIns['has'].call(Args(rec, args)).value
                existing = node.evaluate() if exists else BuiltIns['blank']
                if existing != BuiltIns['blank']:
                    return Args(existing)
            case _:
                val = node.evaluate()
                if val != BuiltIns['blank']:
                    return Args(val)
    return Args(BuiltIns['blank'])

def nullish(*args: Record):
    for arg in args:
        if arg != BuiltIns['blank']:
            return arg
    return BuiltIns['blank']


Op['??'].eval_args = eval_nullish_args
Op['??'].fn = Function({AnyPlusPattern: nullish})

def make_or_fn(as_nodes: bool):
    def fn(*args: Node | Record):
        val = BuiltIns['false']
        for arg in args:
            val = arg.evaluate() if as_nodes else arg
            if val.truthy:
                break
        if as_nodes:
            return Args(val)
        return val
    return fn


Op['or'].eval_args = make_or_fn(True)
Op['or'].fn = Function({AnyPlusPattern: make_or_fn(False)},
                       name='or')

def make_and_fn(as_nodes: bool):
    def fn(*args: Node | Record):
        val = BuiltIns['true']
        for arg in args:
            val = arg.evaluate() if as_nodes else arg
            if not val.truthy:
                break
        if as_nodes:
            return Args(val)
        return val
    return fn


Op['and'].eval_args = make_and_fn(True)
Op['and'].fn = Function({AnyPlusPattern: make_and_fn(False)},
                        name='and')

Op['not'].fn = Function({AnyParam: lambda x: py_value(not x.truthy)},
                        name='not')

Op['in'].fn = Function({ParamSet(AnyParam, FunctionParam):
                            lambda a, b: py_value(Args(a) in b.op_map),
                        ParamSet(AnyParam, NonStrSeqParam):
                            lambda a, b: py_value(a in b.value),
                        ParamSet(AnyParam, StringParam):
                            lambda a, b: py_value(a.value in b.value)},
                       name='in')

Op['=='].fn = Function({AnyBinopPattern: lambda a, b: py_value(a == b)},
                       name='==')
Op['!='].fn = Function({AnyBinopPattern: lambda a, b: py_value(not BuiltIns['=='].call(a, b).value)},
                       name='!=')
def eval_is_op_args(lhs: Node, rhs: Node) -> Args:
    # if rhs.type is TokenType.Name:
    #     rhs = BindExpr(rhs)
    return Args(lhs.evaluate(), rhs.eval_pattern())
Op['~'].eval_args = Op['!~'].eval_args = Op['is'].eval_args = Op['is not'].eval_args = eval_is_op_args
Op['~'].fn = Function({AnyBinopPattern: lambda a, b: py_value(b.match(a) is not None)},
                      name='~')
Op['!~'].fn = Function({AnyBinopPattern: lambda a, b: py_value(b.match(a) is None)},
                       name='!~')
Op['is'].fn = Function({AnyBinopPattern: lambda a, b: py_value(b.match(a) is not None)},
                       name='is')
Op['is not'].fn = Function({AnyBinopPattern: lambda a, b: py_value(b.match(a) is None)},
                           name='is not')

def eval_args_as_pattern(*nodes: Node) -> Args:
    return Args(*(node.eval_pattern() for node in nodes))
Op['|'].eval_args = Op['&'].eval_args = eval_args_as_pattern
Op['|'].fn = Function({AnyPlusPattern: lambda *args: Parameter(UnionMatcher(*args))},
                      name='|')
Op['<'].fn = Function({NormalBinopPattern: lambda a, b: py_value(a.value < b.value)},
                   name='<')
Op['>'].fn = Function({NormalBinopPattern: lambda a, b: py_value(a.value > b.value)},
                   name='>')
Op['<='].fn = Function({AnyBinopPattern:
                         lambda a, b: py_value(BuiltIns['<'].call(a, b).value or BuiltIns['=='].call(a, b).value)},
                    name='<=')
Op['>='].fn = Function({AnyBinopPattern:
                         lambda a, b: py_value(BuiltIns['>'].call(a, b).value or BuiltIns['=='].call(a, b).value)},
                    name='>=')
Op['to'].fn = Function({ParamSet(*[Parameter(UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank'])))]*2):
                            lambda *args: Range(*args)},
                       name='to')
Op['>>'].fn = Op['to'].fn
Op['>>'].eval_args = Op['to'].eval_args = lambda *nodes: Args(*(n.evaluate() for n in nodes))
Op['by'].fn = Function({ParamSet(Parameter(TraitMatcher(RangeTrait)), NumericParam):
                            lambda r, step: Range(*r.data[:2], step),
                        ParamSet(SeqParam, NumericParam): lambda seq, step: (v for v in seq[::step.value])},
                       name='by')
Op['+'].fn = Function({Parameter(TraitMatcher(NumTrait), quantifier='+'):
                           lambda *args: py_value(sum(n.value for n in args)),
                       Parameter(TraitMatcher(StrTrait), quantifier='+'):
                           lambda *args: py_value(''.join(n.value for n in args)),
                       # ParamSet(AnyParam): lambda a: BuiltIns['num'].call(a),
                       Parameter(TraitMatcher(ListTrait), quantifier='+'):
                           lambda *args: py_value(sum((n.value for n in args), [])),
                       Parameter(TraitMatcher(TupTrait), quantifier='+'):
                           lambda *args: py_value(sum((n.value for n in args), ())),
                       }, name='+')
Op['-'].fn = Function({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
                       ParamSet(AnyParam): lambda a: py_value(-a.value)},
                      name='-')
def product(*args: PyValue):
    acc = args[0].value
    for n in args[1:]:
        if acc == 0:
            return py_value(0)
        acc *= n.value
    return py_value(acc)


Op['*'].fn = Function({Parameter(TraitMatcher(NumTrait), quantifier='+'): product,
                       ParamSet(SeqParam, IntegralParam): lambda a, b: py_value(a.value * b.value)},
                      name='*')
Op['/'].fn = Function({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value / b.value),
                       ParamSet(RationalParam, RationalParam): lambda a, b:
                    py_value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))},
                   name='/')
Op['//'].fn = Function({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value // b.value)},
                       name='//')
Op['%'].fn = Function({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)},
                      name='%')
Op['**'].fn = Function({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)},
                       name='**')
Op['^'].fn = Function({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)},
                      name='^')
Op['?'].fn = Function({AnyParam: lambda p: UnionMatcher(patternize(p), ValueMatcher(BuiltIns['blank']))},
                      name='?')


def has_option(fn: Record, arg: Record = None) -> PyValue:
    if arg is None:
        fn, arg = None, fn

    match fn, arg:
        case None, PyValue(value=str() as name):
            return py_value(state.deref(name, None) is not None)
        case None, _:
            raise TypeErr(f"Line {state.line}: When used as a prefix, "
                          f"the right-hand term of the `has` operator must be a string, not {arg.table}")
        case Record(), PyValue(value=str() as name):
            return py_value(fn.get(name, None) is not None)
        # case Record(), List(records=args) | PyValue(value=tuple() as args):
        case Record(), Args() as args:
            option, _ = fn.select(args)
            return py_value(option is not None)
        case Record(), PyValue(value=tuple() | list() as args):
            args = Args(*args)
            option, _ = fn.select(args)
            return py_value(option is not None)
        case _:
            raise TypeErr(f"Line {state.line}: "
                          f"The right-hand term of the `has` operator must be a string or sequence of arguments.")


Op['has'].fn = Function({ParamSet(AnyParam, NonStrSeqParam): has_option,
                         AnyBinopPattern: has_option,
                         ParamSet(StringParam): lambda s: py_value(state.deref(s, None) is not None),
                         ParamSet(NormalParam): has_option},
                        name='has')

Op['&'].fn = Function({AnyPlusPattern: lambda *args: Parameter(IntersectionMatcher(*map(patternize, args)))},
                      name='&')
Op['@'].fn = Function({AnyParam: lambda rec: Parameter(ValueMatcher(rec))})
def invert_pattern(rec: Record):
    match patternize(rec):
        case Parameter(pattern=Matcher() as patt, binding=b, quantifier=q, default=d):
            if q[0] in "+*":
                raise NotImplementedError
            return Parameter(NotMatcher(patt), b, q, d)
    raise NotImplementedError

Op['!'].fn = Function({AnyParam: lambda rec: Parameter(ValueMatcher(rec))})

def eval_declaration_arg(_, arg: Node) -> PyValue[str]:
    match arg:
        case Token(type=TokenType.Name, text=name):
            return py_value(name)
    raise AssertionError


Op['var'].eval_args = Op['local'].eval_args = eval_declaration_arg
Op['var'].fn = Function({StringParam: lambda x: VarPatt(x.value)})
Op['local'].fn = Function({StringParam: lambda x: LocalPatt(x.value)})

def make_op_equals_functions(sym: str):
    match sym:
        case '&&':
            op_fn = Op['and'].fn
        case '||':
            op_fn = Op['or'].fn
        case _:
            op_fn = Op[sym].fn
    op_name = sym + '='
    Op[op_name].fn = Function({ParamSet(StringParam, AnyParam):
                                   lambda name, val: state.env.assign(name.value,
                                                                        op_fn.call(state.deref(name), val)),
                               ParamSet(FunctionParam, AnyParam, AnyParam):
                                   lambda *args: set_or_assign_option(*args, operation=op_fn),
                               ParamSet(AnyParam, StringParam, AnyParam):
                                   lambda rec, name, val: rec.set(name.value, op_fn.call(rec.get(name.value), val))
                               }, name=op_name)
    Op[op_name].fn = Function({ParamSet(PatternParam, AnyParam, AnyParam): set_with_fn(op_fn)
                               }, name=op_name)
    Op[op_name].eval_args = eval_eq_args


for sym in ('+', '-', '*', '/', '//', '**', '%', '&', '|', '&&', '||'):  # ??= got special treatment
    match sym:
        case '&&':
            op_fn = Op['and'].fn
        case '||':
            op_fn = Op['or'].fn
        case _:
            op_fn = Op[sym].fn
    op_name = sym + '='
    Op[op_name].fn = Function({ParamSet(PatternParam, AnyParam, AnyParam): set_with_fn(op_fn)
                               }, name=op_name)
    Op[op_name].eval_args = lambda lhs, rhs: Op['='].eval_args(lhs, lhs, rhs)


""" This is an option that matches (1) value int (the trait itself) and (2) any int value.  Eg, `int < 2`.
    The resolution of the option is a function that returns a pattern.  The pattern is an int trait matcher, with a
    specification via Lambda Matcher that said int ought to be less than the value in the pattern expression. 
"""
def make_comp_opt(trait: str):
    if trait in ('str', 'list', 'tuple', 'set', 'frozenset'):
        def t(rec):
            return len(rec.value)
    elif trait in ('int', 'ratio', 'float', 'num'):
        def t(rec):
            return rec.value
    Op['<'].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                      Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: t(y) < x.value))
                             )
    Op['<='].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                       Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: t(y) <= x.value))
                              )
    Op['>='].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                       Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: t(y) >= x.value))
                              )
    Op['>'].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                      Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: t(y) > x.value))
                             )


for trait in ('int', 'ratio', 'float', 'num', 'str', 'list', 'tuple', 'set', 'frozenset'):
    make_comp_opt(trait)