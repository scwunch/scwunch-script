import re
from fractions import Fraction
from Syntax import Node, Token, ListNode, TokenType
from Env import *
from tables import *
from Expressions import expressionize, read_number, Expression, py_eval, piliize

# BuiltIns['blank'] = None
BuiltIns['Table'] = TableTable = MetaTable()
BuiltIns['Trait'] = TraitTable = BootstrapTable('Trait')
BuiltIns['Pattern'] = IntermediatePatternTable = BootstrapTable('Pattern')
BuiltIns['Option'] = IntermediateOptionTable = BootstrapTable('Option')
BuiltIns['Args'] = IntermediateArgsTable = BootstrapTable('Args')
TableTable.traits = (Trait(),)
TraitTable.traits = (Trait(),)

BuiltIns['Blank'] = SetTable()
BuiltIns['blank'] = PyValue(BuiltIns['Blank'], None)

# Number traits
BuiltIns['num'] = NumTrait = Trait()
BuiltIns['float'] = Trait()
BuiltIns['ratio'] = RatioTrait = Trait()
BuiltIns['int'] = IntTrait = Trait()
BuiltIns['bool'] = Trait()

# Collection Traits
BuiltIns['iter'] = IterTrait = Trait()
BuiltIns['seq'] = SeqTrait = Trait()
BuiltIns['str'] = StrTrait = Trait()
BuiltIns['tuple'] = TupTrait = Trait()
BuiltIns['set'] = SetTrait = Trait()
BuiltIns['frozenset'] = FrozenSetTrait = Trait()
BuiltIns['list'] = ListTrait = Trait()
BuiltIns['dict'] = DictTrait = Trait()

# Numeric Tables
BuiltIns['Bool'] = SetTable(BuiltIns['bool'], IntTrait, RatioTrait, NumTrait)
BuiltIns['false'] = PyValue(BuiltIns['Bool'], False)
BuiltIns['true'] = PyValue(BuiltIns['Bool'], True)

BuiltIns['Integer'] = VirtTable(IntTrait, RatioTrait, NumTrait)
BuiltIns['Fraction'] = VirtTable(RatioTrait, NumTrait)
BuiltIns['Float'] = VirtTable(BuiltIns['float'], NumTrait)

# Collection Tables
BuiltIns['String'] = VirtTable(StrTrait, SeqTrait, IterTrait)
BuiltIns['Tuple'] = VirtTable(TupTrait, SeqTrait, IterTrait)
BuiltIns['Set'] = VirtTable(SetTrait, IterTrait)
BuiltIns['List'] = VirtTable(ListTrait, SeqTrait, IterTrait)
BuiltIns['Dictionary'] = VirtTable(DictTrait, IterTrait)
BuiltIns['Args'] = VirtTable(SeqTrait, DictTrait, IterTrait)
for rec in IntermediateArgsTable.records:
    rec.table = BuiltIns['Args']

BuiltIns['func'] = FuncTrait = Trait()
BuiltIns['Function'] = ListTable(FuncTrait)

TableTable.traits += (FuncTrait,)
BuiltIns['Trait'].traits += (FuncTrait,)

BuiltIns['Pattern'] = ListTable(SeqTrait, IterTrait)
BuiltIns['Pattern'].records = IntermediatePatternTable.records
BuiltIns['Block'] = ListTable()

BuiltIns['Field'] = ListTable()

def upsert_field_fields(fields: list[Field]):
    fields.append(Slot('name', TraitMatcher(BuiltIns['str'])))
    fields.append(Slot('type', TableMatcher(BuiltIns['Pattern'])))
    fields.append(Slot('is_formula', TraitMatcher(BuiltIns['bool'])))
    fields.append(Slot('default', UnionMatcher(TraitMatcher(FuncTrait), AnyMatcher())))
    fields.append(Slot('formula', TraitMatcher(FuncTrait)))
    fields.append(Slot('setter', TraitMatcher(FuncTrait)))
upsert_field_fields(BuiltIns['Field'].trait.fields)

BuiltIns['Option'] = ListTable()
BuiltIns['Option'].records = IntermediateOptionTable.records
BuiltIns['Option'].trait.fields.append(Slot('signature', TableMatcher(BuiltIns['Pattern'])))
BuiltIns['Option'].trait.fields.append(Slot('code block', TableMatcher(BuiltIns['Block'])))

FuncTrait.fields.append(Slot('options', TraitMatcher(BuiltIns['seq'])))
# the type should also have a specifier like `list[Option]` ... and also a default value: []

# now redo the Field fields, since they weren't able to properly initialize while the fields were incomplete
upsert_field_fields(BuiltIns['Field'].trait.fields)
BuiltIns['Field'].records = BuiltIns['Field'].records[5:]

BuiltIns['PythonObject'] = ListTable()

print("DONE ADDING BUILTIN TABLES.")


BuiltIns['any'] = Pattern(Parameter(AnyMatcher()))
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
NonStrSeqParam = Parameter(Intersection(TraitMatcher(SeqTrait), TraitMatcher(StrTrait, inverse=1)))
IterParam = Parameter(TraitMatcher(IterTrait))
# TypeParam = Parameter(TableMatcher(BuiltIns["Type"]))
PatternParam = Parameter(TableMatcher(BuiltIns["Pattern"]))
FunctionParam = Parameter(TraitMatcher(BuiltIns["func"]))
TableParam = Parameter(TableMatcher(BuiltIns['Table']))
AnyParam = Parameter(AnyMatcher())
NormalBinopPattern = Pattern(NormalParam, NormalParam)
AnyBinopPattern = Pattern(AnyParam, AnyParam)
AnyPlusPattern = Pattern(Parameter(AnyMatcher(), quantifier="+"))
AnyPattern = Pattern(Parameter(AnyMatcher(), quantifier="*"))

PositiveIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value > 0)))
NegativeIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value < 0)))
NonZeroIntParam = Parameter(TraitMatcher(BuiltIns["int"], guard=lambda x: py_value(x.value != 0)))
OneIndexList = Parameter(TableMatcher(BuiltIns['List'],
                                      guard=lambda x: py_value(len(x.value) == 1 and
                                                               NonZeroIntParam.match_score(x.value[0]))))

BuiltIns['bool'].assign_option(Pattern(AnyParam), lambda x: py_value(bool(x.value)))
BuiltIns['num'].assign_option(Pattern(BoolParam), lambda x: py_value(int(x.value)))
BuiltIns['num'].assign_option(Pattern(NumericParam), lambda x: py_value(x.value))
BuiltIns['num'].assign_option(Pattern(StringParam), lambda x: py_value(read_number(x.value, Context.settings['base'])))
BuiltIns['num'].assign_option(Pattern(StringParam, IntegralParam),
                              lambda x, b: py_value(read_number(x.value, b.value)))
# Function({Pattern(BoolParam): lambda x: py_value(int(x.value)),
#                               Pattern(NumericParam): lambda x: py_value(x.value),
#                                Pattern(StringParam): lambda x: py_value(read_number(x.value, Context.settings['base'])),
#                                Pattern(StringParam, IntegralParam): lambda x, b: py_value(read_number(x.value, b.value))},
#                               name='number')
BuiltIns['int'].assign_option(Pattern(NormalParam), lambda x: py_value(int(BuiltIns['num'].call(x).value)))
BuiltIns['ratio'].assign_option(Pattern(NormalParam), lambda x: py_value(Fraction(BuiltIns['num'].call(x).value)))
BuiltIns['float'].assign_option(Pattern(NormalParam), lambda x: py_value(float(BuiltIns['num'].call(x).value)))
# BuiltIns['float'] = Function({Pattern(NormalParam), lambda x: py_value(float(BuiltIns['number'].call(x).value)))
BuiltIns['str'].assign_option(Pattern(AnyParam), lambda x: x.to_string())
BuiltIns['str'].assign_option(Option(StringParam, lambda x: x))
BuiltIns['str'].assign_option(Pattern(NumericParam, IntegralParam),
                              lambda n, b: py_value(write_number(n.value, b.value)))
BuiltIns['list'].assign_option(Pattern(SeqParam), lambda x: py_value(list(x.value)))
BuiltIns['tuple'].assign_option(Pattern(SeqParam), lambda x: py_value(tuple(x.value)))
BuiltIns['set'].assign_option(Pattern(SeqParam), lambda x: py_value(set(x.value)))
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
#                               set=Function({Pattern(StringParam, AnyParam): setting_set}),
#                               get=Function({Pattern(StringParam): setting_get}))
get_base_fn = Function({AnyParam: lambda _: py_value("_ubtqphsond"[Context.settings['base']])})
set_base_fn = Function({Pattern(AnyParam, Parameter(UnionMatcher(TraitMatcher(StrTrait),
                                                                 *(ValueMatcher(py_value(v)) for v in range(1, 11))))):
                        lambda _, val: setting_set('base', val)})
get_sort_options = Function({AnyParam: lambda: py_value(Context.settings['sort_options'])})
set_sort_options = Function({Pattern(AnyParam, BoolParam): lambda _, val: setting_set('sort_options', val)})
BuiltIns['settings'] = Function({},
                                Formula('base', TraitMatcher(IntTrait), get_base_fn),
                                Setter('base', set_base_fn),
                                Formula('sort_options', TraitMatcher(BuiltIns['bool']), get_sort_options),
                                Setter('sort_options', set_sort_options)
                                )

def key_to_param_set(key: PyValue) -> Pattern:
    if hasattr(key, 'value') and isinstance(key.value, list):
        vals = key.value
    else:
        vals = [key]
    params = (Parameter(ValueMatcher(pval, pval.value if isinstance(pval.value, str) else None))
              for pval in vals)
    return Pattern(*params)

# def set_value(fn: Record, key: Pattern, val: Record):
#     match key:
#         case PyValue(value=str() as name):
#             pass
#         case List(records=records) | PyValue(value=tuple() as records):
#             pass


BuiltIns['set'] = Function({Pattern(FunctionParam, AnyParam, AnyParam):
                           lambda fn, key, val: fn.assign_option(key_to_param_set(key), val).resolution})
# BuiltIns['string'].add_option(Pattern(ListParam), lambda l: py_value(str(l.value[1:])))
# BuiltIns['string'].add_option(Pattern(NumberParam),
#                               lambda n: py_value('-' * (n.value < 0) +
#                                               base(abs(n.value), 10, 6, string=True, recurring=False)))
# BuiltIns['string'].add_option(Pattern(Parameter(TableMatcher(BuiltIns["Type"]))), lambda t: py_value(t.value.name))

BuiltIns['type'] = Function({AnyParam: lambda v: v.table})

BuiltIns['len'] = Function({SeqParam: lambda s: py_value(len(s.value)),
                            FunctionParam: lambda f: py_value(len(f.options)),
                            PatternParam: lambda p: py_value(len(p)),
                            Pattern(Parameter(TableMatcher(BuiltIns['Table']))): lambda t: py_value(len(t.records))
                            })
BuiltIns['traits'] = Function({Parameter(BuiltIns['Table']): lambda t: py_value(t.traits)})
# BuiltIns['contains'] = Function({Pattern(FunctionParam: AnyParam),
#                                 lambda a, b: py_value(b in (opt.value for opt in a.options)))
# BuiltIns['options'] = Function({AnyParam: lambda x: piliize([py_value(lp.pattern) for lp in x.options])})
# BuiltIns['names'] = Function({AnyParam: lambda x: piliize([py_value(k) for k in x.named_options.keys()])})
# BuiltIns['keys'] = Function({AnyParam:
#                             lambda x: piliize([lp.pattern[0].pattern.value for lp in x.options
#                                                if len(lp.pattern) == 1 and isinstance(lp.pattern[0].pattern, ValueMatcher)])})

BuiltIns['max'] = Function({Parameter(BuiltIns["num"], quantifier='+'):
                                lambda *args: py_value(max(*[arg.value for arg in args])),
                            Parameter(TraitMatcher(BuiltIns["str"]), quantifier='+'):
                                lambda *args: py_value(max(*[arg.value for arg in args])),
                            IterParam:
                                lambda ls: py_value(max(*[arg.value for arg in ls.value]))
                            })
BuiltIns['min'] = Function({Parameter(BuiltIns["num"], quantifier='+'):
                                lambda *args: py_value(min(*[arg.value for arg in args])),
                            Parameter(TraitMatcher(BuiltIns["str"]), quantifier='+'):
                                lambda *args: py_value(min(arg.value for arg in args)),
                            IterParam: lambda ls: py_value(min(arg.value for arg in ls.value))
                            })
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
BuiltIns['map'] = Function({Pattern(SeqParam, FunctionParam): lambda ls, fn: piliize([fn.call(val) for val in ls.value]),
                            Pattern(FunctionParam, SeqParam): lambda fn, ls: piliize([fn.call(val) for val in ls.value])})
BuiltIns['filter'] = Function({Pattern(ListParam, FunctionParam):
                                   lambda ls, fn: piliize([v for v in ls.value
                                                           if BuiltIns['bool'].call(fn.call(v)).value]),
                               Pattern(FunctionParam, ListParam):
                                   lambda fn, ls: piliize([v for v in ls.value
                                                           if BuiltIns['bool'].call(fn.call(v)).value])})
BuiltIns['sum'] = Function({SeqParam: lambda ls: BuiltIns['+'].call(*ls.value)})
BuiltIns['trim'] = Function({StringParam: lambda text: py_value(text.value.strip()),
                            Pattern(StringParam, StringParam): lambda t, c: py_value(t.value.strip(c.value))})
BuiltIns['upper'] = Function({StringParam: lambda text: py_value(text.value.upper())})
BuiltIns['lower'] = Function({StringParam: lambda text: py_value(text.value.lower())})
BuiltIns['match'] = Function({Pattern(StringParam, StringParam):
                                  lambda s, p: py_value(re.match(p.value, s.value)),
                              Pattern(StringParam, StringParam, StringParam):
                                  lambda s, p, f: py_value(re.match(p.value, s.value, f.value))})
# BuiltIns['self'] = lambda: Context.env.caller or Context.env or py_value(None)
# def Args(fn: Function):
#     arg_list = piliize([opt.value for opt in fn.args])
#     arg_list.add_option(FunctionParam, lambda fn: Args(fn))
#     return arg_list
# BuiltIns['args'] = lambda: Args(Context.env)

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

# SeqTrait.add_option(Pattern(IntegralParam, IntegralParam), Native(list_get))
# SeqTrait.add_option(Pattern(IntegralParam, IntegralParam, IntegralParam), Native(list_get))
BuiltIns['slice'] = Function({Pattern(SeqParam, IntegralParam, IntegralParam): list_slice,
                              Pattern(SeqParam, IntegralParam, IntegralParam, IntegralParam): list_slice})
BuiltIns['push'] = Function({Pattern(ListParam, AnyParam):
                             lambda ls, item: ls.value.append(item)})
BuiltIns['join'] = Function({Pattern(SeqParam, StringParam):
                             lambda ls, sep: py_value(sep.value.join(BuiltIns['str'].call(item).value for item in ls.value))})
BuiltIns['split'] = Function({Pattern(StringParam, StringParam): lambda txt, sep: piliize([py_value(s) for s in txt.value.split(sep.value)])})

BuiltIns['new'] = Function({Pattern(TableParam, AnyPattern[0]): lambda t, *args, **kwargs: Record(t, *args, **kwargs)})

SeqTrait.options.append(Option(IntegralParam, Native(list_get)))
SeqTrait.options.append(Option(AnyPlusPattern,
                               Native(lambda args: BuiltIns['slice'](Args(Context.env.caller) + args))))
BuiltIns['List'].integrate_traits()
BuiltIns['Tuple'].integrate_traits()

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