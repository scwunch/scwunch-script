import re
from fractions import Fraction
from pili.utils import write_number, BASES, KeyErr, TypeErr, RuntimeErr
from pili.interpreter import *

print(f'loading {__name__}.py')

# BuiltIns['blank'] = None
BuiltIns['Table'] = TableTable = MetaTable()
BuiltIns['Trait'] = TraitTable = BootstrapTable('Trait')
BuiltIns['pattern'] = PattTrait = Trait(name='pattern')
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
BuiltIns['Block'] = ListTable(name='Block')

# BuiltIns['Field'] = ListTable(name='Field')

# def upsert_field_fields(fields: list[Field]):
#     fields.append(Slot('name', TraitMatcher(BuiltIns['str'])))
#     fields.append(Slot('type', TableMatcher(BuiltIns['Pattern'])))
#     fields.append(Slot('is_formula', TraitMatcher(BuiltIns['bool'])))
#     fields.append(Slot('default', UnionMatcher(TraitMatcher(FuncTrait), AnyMatcher())))
#     fields.append(Slot('formula', TraitMatcher(FuncTrait)))
#     fields.append(Slot('setter', TraitMatcher(FuncTrait)))
# upsert_field_fields(BuiltIns['Field'].trait.fields)


BuiltIns['Option'] = ListTable(name='Option')
BuiltIns['Option'].records = IntermediateOptionTable.records
BuiltIns['Option'].trait.fields.append(Slot('signature', TableMatcher(BuiltIns['Pattern'])))
BuiltIns['Option'].trait.fields.append(Slot('code block', TableMatcher(BuiltIns['Block'])))

TableTable.trait.fields = [
    Formula('fields', TableMatcher(BuiltIns['Tuple']), lambda self: py_value(tuple(self.fields.values()))),
    Formula('records', TraitMatcher(IterTrait), lambda self: py_value(self.records))
]
FuncTrait.fields.append(Slot('options', TraitMatcher(BuiltIns['seq'])))

BuiltIns['Field'] = FieldTable = ListTable(name='Field')
FieldTable.trait.fields = [
    Formula('name', TraitMatcher(StrTrait), lambda field: py_value(field.name)),
    Formula('type', UnionMatcher(TraitMatcher(PattTrait), ValueMatcher(BuiltIns['blank'])),
            lambda field: field.type or BuiltIns['blank']),
    Formula('getter', UnionMatcher(TraitMatcher(FuncTrait), ValueMatcher(BuiltIns['true']), ValueMatcher(BuiltIns['blank'])),
            lambda field: py_value(field.getter)),
    Formula('setter', UnionMatcher(TraitMatcher(FuncTrait), ValueMatcher(BuiltIns['true']), ValueMatcher(BuiltIns['blank'])),
            lambda field: py_value(field.setter)),
    Formula('default', UnionMatcher(FieldMatcher((), {'value': AnyMatcher()}), ValueMatcher(BuiltIns['blank'])),
            lambda field: field.wrap_default())
]

# now redo the Field fields, since they weren't able to properly initialize while the fields were incomplete
# upsert_field_fields(BuiltIns['Field'].trait.fields)
# BuiltIns['Field'].records = BuiltIns['Field'].records[5:]

BuiltIns['range'].fields = [Slot('start', UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank']))),
                            Slot('end', UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank']))),
                            Slot('step', UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank']))),]

BuiltIns['PythonObject'] = ListTable(name='PythonObject')

""" DONE ADDING BUILTIN TABLES."""


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
NonStr = NotMatcher(TraitMatcher(StrTrait))
NonStrSeqParam = Parameter(IntersectionMatcher(TraitMatcher(SeqTrait), NonStr))
SetParam = Parameter(TraitMatcher(SetTrait))
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
state.DEFAULT_PATTERN = ParamSet(Parameter(AnyMatcher(), 'args', "*"), kwargs='kwargs')

BuiltIns['bool'].assign_option(ParamSet(AnyParam), lambda x: py_value(x.truthy))
BuiltIns['num'].assign_option(ParamSet(BoolParam), lambda x: py_value(int(x.truthy)))
BuiltIns['num'].assign_option(ParamSet(NumericParam), lambda x: x)
BuiltIns['num'].assign_option(ParamSet(StringParam), lambda x: py_value(read_number(x.value, state.settings['base'])))
BuiltIns['num'].assign_option(ParamSet(StringParam, IntegralParam),
                              lambda x, b: py_value(read_number(x.value, b.value)))
BuiltIns['int'].assign_option(ParamSet(NormalParam), lambda x: py_value(int(BuiltIns['num'].call(x).value)))
BuiltIns['ratio'].assign_option(ParamSet(NormalParam), lambda x: py_value(Fraction(BuiltIns['num'].call(x).value)))
BuiltIns['float'].assign_option(ParamSet(NormalParam), lambda x: py_value(float(BuiltIns['num'].call(x).value)))
BuiltIns['str'].default_option = (
    Option(ParamSet(AnyParam, Parameter(BoolParam, 'info', default=BuiltIns['blank'])),
           lambda x, info=BuiltIns['blank']: x.to_string(info.truthy)))
BuiltIns['str'].assign_option(Option(StringParam, lambda x: x))
BuiltIns['str'].assign_option(ParamSet(NumericParam, IntegralParam),
                              lambda n, b: py_value(write_number(n.value, b.value)))
# BuiltIns['list'].assign_option(ParamSet(SeqParam), lambda x: py_value(list(x.value)))
BuiltIns['list'].default_option = Option(ParamSet(IterParam), lambda x: py_value(list(x)))
BuiltIns['tuple'].default_option = Option(ParamSet(IterParam), lambda x: py_value(tuple(x)))
BuiltIns['set'].default_option = Option(ParamSet(IterParam), lambda x: py_value(set(x)))
# BuiltIns['iter'].assign_option(Option(UnionMatcher(*(TraitMatcher(BuiltIns[t])
#                                                      for t in ('tuple', 'list', 'set', 'frozenset', 'str', 'range'))),
#                                       lambda x: x))
BuiltIns['iter'].default_option = Option(ParamSet(IterParam), lambda x: x)

def make_custom_iter(rec: Record):
    while not dot_call_fn(rec, 'done').value:
        yield dot_call_fn(rec, 'next')
BuiltIns['iter'].assign_option(ParamSet(Parameter(FieldMatcher((), dict(done=AnyMatcher(), next=AnyMatcher())))),
                               make_custom_iter)


def setting_set(prop: str, val: PyValue):
    val = val.value
    if prop == 'base' and isinstance(val, str):
        if val not in BASES:
            raise RuntimeErr(f'Line {state.line}: {val} is not a valid base.  Valid base symbols are the following:\n'
                             f"b: 2\nt: 3\nq: 4\np: 5\nh: 6\ns: 7\no: 8\nn: 9\nd: 10")
        val = BASES[val[0]]
    state.settings[prop] = val
    return BuiltIns['settings']
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

# BuiltIns['repr'] = Function({AnyParam: lambda arg: arg.to_string(True)})

BuiltIns['type'] = Function(default=lambda v: v.table)

BuiltIns['len'] = Function({SeqParam: lambda s: py_value(len(s.value)),
                            PatternParam: lambda p: py_value(len(p)),
                            FunctionParam: lambda fn: py_value(len(fn.op_map) + len(fn.op_list)),
                            # ParamSet(Parameter(TableMatcher(BuiltIns['Table']))): lambda t: py_value(len(t.records))
                            })
# BuiltIns['traits'] = Function({Parameter(BuiltIns['Table']): lambda t: py_value(t.traits)})
# # BuiltIns['contains'] = Function({ParamSet(FunctionParam: AnyParam),
# #                                 lambda a, b: py_value(b in (opt.value for opt in a.options)))
# # BuiltIns['options'] = Function({AnyParam: lambda x: piliize([py_value(lp.pattern) for lp in x.options])})
# # BuiltIns['names'] = Function({AnyParam: lambda x: piliize([py_value(k) for k in x.named_options.keys()])})
def get_keys(fn: Function):
    return py_value({k[0] if len(k) == 1 else py_value(k.positional_arguments)
                     for k in fn.op_map.keys()})
def get_items(fn: Function):
    return py_value([((k[0] if len(k) == 1 else py_value(k.positional_arguments)), v.value)
                     for k, v in fn.op_map.items()])
BuiltIns['keys'] = Function({FunctionParam: get_keys})
BuiltIns['values'] = Function({FunctionParam: lambda fn: py_value([opt.value for opt in fn.op_map.values()])})
BuiltIns['items'] = Function({FunctionParam: get_items})
#'keys
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


RangeTrait.assign_option(Parameter(TraitMatcher(BuiltIns["num"]), quantifier="*"), lambda *args: Range(*args))
# BuiltIns['map'] = Function({ParamSet(SeqParam, FunctionParam): lambda ls, fn: piliize([fn.call(val) for val in ls.value]),
#                             ParamSet(FunctionParam, SeqParam): lambda fn, ls: piliize([fn.call(val) for val in ls.value])})
# BuiltIns['filter'] = Function({ParamSet(ListParam, FunctionParam):
#                                    lambda ls, fn: piliize([v for v in ls.value
#                                                            if BuiltIns['bool'].call(fn.call(v)).value]),
#                                ParamSet(FunctionParam, ListParam):
#                                    lambda fn, ls: piliize([v for v in ls.value
#                                                            if BuiltIns['bool'].call(fn.call(v)).value])})
BuiltIns['sum'] = Function({IterParam: lambda ls: BuiltIns['+'].call(*ls)})
BuiltIns['trim'] = Function({StringParam: lambda text: py_value(text.value.strip()),
                            ParamSet(StringParam, StringParam): lambda t, c: py_value(t.value.strip(c.value))})
BuiltIns['upper'] = Function({StringParam: lambda text: py_value(text.value.upper())})
BuiltIns['lower'] = Function({StringParam: lambda text: py_value(text.value.lower())})
BuiltIns['replace'] = \
    Function({ParamSet(StringParam, StringParam, StringParam, Parameter(TraitMatcher(IntTrait), quantifier='?')):
                  lambda self, old, new, count=-1: py_value(self.value.replace(old.value, new.value, count.value))})

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


BuiltIns['copy'] = Function({AnyParam: lambda rec: transmogrify(Record(None), rec)}, name='copy')

def transmogrify(ref: Record, data: Record):
    if ref.table.name in ('Blank', 'Bool'):
        raise RuntimeErr(f'Cannot transmogrify singleton {ref}.')
    ref.__class__ = data.__class__
    ref.__dict__ = data.__dict__
    return ref
BuiltIns['transmogrify'] = Function({ParamSet(AnyParam, AnyParam): transmogrify})
def make_flags(*names: str) -> dict[str, Parameter]:
    return {n: Parameter(TraitMatcher(BuiltIns['bool']), n, '?', BuiltIns['blank']) for n in names}


BuiltIns['python'] = Function({ParamSet(StringParam,
                                        named_params=make_flags('direct', 'execute')):
                                   run_python_code}, name='python')  # function filled in in syntax.py

for tbl in TableTable.records:
    tbl: Table
    tbl.integrate_traits()
