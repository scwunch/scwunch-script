import re
import warnings
from collections import deque
from fractions import Fraction
from typing import TypeVar, Generic
from . import state
from .state import BuiltIns
from .utils import NoMatchingOptionError, RuntimeErr, SlotErr, TypeErr, write_number, MatchErr, frozendict, \
    InitializationErr, MissingNameErr, PatternErr, SyntaxErr, call, KeyErr, limit_str, ZeroIndexErr, IndexErr, \
    DuplicateNameErr

print(f'loading {__name__}.py')

PyFunction = type(lambda: None)

def pili_repr(rec):
    return BuiltIns['str'].call(Args(rec, flags={'info'}))

class OptionCatalog:
    op_list: list
    op_map: dict
    default_option = None
    def __init__(self, options=None, *traits, default=None):
        self.op_list = []
        self.op_map = {}
        if options:
            for patt, res in options.items():
                self.assign_option(Option(patt, res))
        for trait in traits:
            for option in trait.op_list:
                self.assign_option(option)
            self.op_map.update(trait.op_map)
        if default:
            self.assign_default(default)

    def assign_default(self, resolution):
        if self.default_option is None:
            self.default_option = Option(state.DEFAULT_PATTERN)
        self.default_option.resolution = resolution

    def assign_option(self, pattern, resolution=None, *, no_clobber=False, prioritize=False):
        match pattern, resolution:
            case Option(pattern=pattern, resolution=resolution) as option, None:
                key = pattern.to_args()
            case _:
                if resolution is None:
                    raise AssertionError("Why are you trying to add a null option?")
                option = Option(pattern, resolution)
                if isinstance(pattern, Args):
                    key = pattern
                else:
                    key = option.pattern.to_args()
                pattern = option.pattern

        if key is not None:
            if no_clobber and key in self.op_map:
                # don't overwrite existing key
                return self.op_map[key]
            else:
                self.op_map[key] = option
                return option

        if state.settings['sort_options']:
            for i, opt in enumerate(self.op_list):
                if option.pattern == opt.pattern:
                    if not no_clobber:
                        self.op_list[i] = option
                    break
                elif option.pattern <= opt.pattern:
                    self.op_list.insert(i, option)
                    break
            else:
                self.op_list.append(option)
        elif opt := self.select_by_pattern(pattern):
            if not no_clobber:
                opt.resolution = resolution
        elif prioritize:
            self.op_list.insert(0, option)
        else:
            self.op_list.append(option)

        return option

    def remove_option(self, pattern):
        opt = self.select_by_pattern(pattern)
        if opt is None:
            raise NoMatchingOptionError(f'cannot find option "{pattern}" to remove')
        opt.nullify()

    def select_and_bind(self, key):
        try:
            if key in self.op_map:
                return self.op_map[key], {}
        except TypeError as e:
            if not (e.args and e.args[0].startswith('unhashable type')):
                raise e
        if state.debug and (self is BuiltIns['call'] or self is BuiltIns['call?']):
            state.debug = None  # pause debugger
        option = bindings = None
        high_score = 0
        for opt in self.op_list:
            bindings = opt.pattern.match(key)
            score, saves = bindings is not None, bindings
            if score == 1:
                if state.debug is None:
                    state.debug = True  # unpause debug
                return opt, saves
            if score > high_score:
                high_score = score
                option, bindings = opt, saves
        if not state.debug and state.debug is not False:
            state.debug = True  # unpause debug
        if option is None and self.default_option:
            bindings = self.default_option.pattern.match(key)
            if bindings is not None:
                return self.default_option, bindings
        return option, bindings

    def select_by_pattern(self, patt, default=None):
        for opt in self.op_list:
            if opt.pattern == patt:
                return opt
        return default


class Record:
    name = None
    # table: Table
    # data: dict[int, Record]
    # key: Record
    truthy = True

    def __init__(self, table, *data_tuple, **data_dict):
        self.table = table
        i = len(data_tuple)
        if i > len(self.table.defaults):
            raise RuntimeErr(f"Line {state.line}: too many values provided for creating new instance of {self.table};"
                             f" Expected a maximum of {len(self.table.defaults)} values but got {i}: {data_tuple}")
        val: Function | None
        defaults = (val.call(self) if val else BuiltIns['blank'] for val in self.table.defaults[i:])
        self.data = [*data_tuple, *defaults]
        for k, v in data_dict.items():
            self.set(k, v)
        table.add_record(self)

    @property
    def value(self):
        return getattr(self, '_value', getattr(self, 'obj', self))

    @value.setter
    def value(self, value):
        self._value = value

    def get(self, name: str, *default, search_table_frame_too=False):
        if name in self.table.getters:
            match self.table.getters[name]:
                case int() as idx:
                    return self.data[idx]
                case Function() as fn:
                    return fn.call(self)
        if search_table_frame_too and self.table.frame:
            val = self.table.frame[name]
            if val is not None:
                return val
        if not default:
            raise SlotErr(f"Line {state.line}: no field found with name '{name}'.")
        return default[0]

    def set(self, name: str, value):
        match self.table.setters.get(name):
            case int() as idx:
                # TODO: check for type agreement ...
                #  or skip it in this context, rely on the type-checking of foo.bar = "value"
                self.data[idx] = value
                return value
            case Function() as fn:
                return fn.call(self, value)
            case None:
                raise SlotErr(f"Line {state.line}: no field found with name '{name}'.")

    def call(self, *args, safe_call=False):
        """ A Record can be called on multiple values (like calling a regular function),
            in which case it will build an Args object.  Or, it can be called on an already built Args object. """
        match args:
            case [Args() as args]:
                pass
            case _:
                args = Args(*args)

        option, bindings = self.select(args)
        if option:
            return option.resolve(args, bindings, self)
        if safe_call:
            return BuiltIns['blank']
        raise NoMatchingOptionError(f'{limit_str(args)} not found in "{self.name or self}"')

    def select(self, args):
        return self.table.catalog.select_and_bind(args)

    def hashable(self):
        try:
            return isinstance(hash(self), int)
        except TypeError:
            return False

    def to_string(self, as_repr=False):
        if not as_repr:
            if self.name:
                return py_value(self.name)
            return py_value(f'{self.table}{py_value(self.data).to_string().value}')
        elements = [self.table.to_string().value]
        if self.name:
            elements.append(repr(self.name))
        elements.extend(f'{key}:{pili_repr(self.get(key)).value}' for key in self.table.getters)
        return py_value('<' + ' '.join(elements) + '>')

    def __str__(self):
        return self.to_string().value

    def __repr__(self):
        return f"Record<{self.table}>({self.data})"


T = TypeVar('T', None, bool, int, Fraction, float, str, tuple, frozenset, set, list)
A = TypeVar('A')

class PyValue(Record, Generic[T]):
    def __init__(self, table, value: T):
        self.value = value
        super().__init__(table)

    def to_string(self, as_repr=False):
        if self.value is None:
            return py_value('blank')
        if self.table is BuiltIns['Bool']:
            return py_value('true') if self.value else py_value('false')
        if BuiltIns['num'] in self.table.traits:
            return py_value(write_number(self.value, state.settings['base']))
        if self.table is BuiltIns['String']:
            return py_value(repr(self.value)) if as_repr else self
        if BuiltIns['seq'] in self.table.traits or BuiltIns['set'] in self.table.traits:
            items = ', '.join(repr(item.value) if isinstance(item.value, str)
                                               else BuiltIns['str'].call(item).value
                              for item in self)
            match self.value:
                case list():
                    return py_value(f'[{items}]')
                case tuple():
                    return py_value(f'({items}{"," * (len(self.value) == 1)})')
                case set() | frozenset():
                    return py_value('{' + items + '}')
        if self.table == BuiltIns['Match']:
            return py_value(str(self.value))
        raise NotImplementedError(f'str[{self.table}]')
        # return py_value(str(self.value))

    def __index__(self) -> int | None:
        if not isinstance(self.value, int | bool):
            raise TypeErr(f"Value used as seq index must have trait int. "
                          f"{self} is a record of {self.table}")
        if not self.value:
            raise ZeroIndexErr(f"Pili indices start at ±1.  0 is not a valid index.")
        return self.value - (self.value > 0)

    def call(self, *args, safe_call=False):
        if BuiltIns['seq'] not in self.table.traits:
            return super().call(*args, safe_call=safe_call)
        match args:
            case [Args() as args]:
                pass
            case _:
                args = Args(*args)
        seq = self.value
        match args:
            case Args(positional_arguments=(PyValue() as index, )):
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
            if not safe_call:
                raise IndexErr(f"Line {state.line}: {e}")
        except TypeError as e:
            if not safe_call:
                if index.value is None:
                    raise TypeErr(f"Line {state.line}: Pili sequence indices start at 1, not 0.")
                raise TypeErr(f"Line {state.line}: {e}")
        except (IndexErr, TypeErr) as e:
            if not safe_call:
                raise e
        assert safe_call
        return BuiltIns['blank']

    def assign_option(self, key, val: Record, prioritize=None):
        key: Args
        assert self.table == BuiltIns['List']
        index: PyValue[int] = key[0]  # noqa
        if index.value == len(self.value) + 1:
            self.value.append(val)
        else:
            self.value[index] = val
        return val

    @property
    def truthy(self):
        return bool(self.value)

    def __hash__(self):
        try:
            return hash(self.value)
        except TypeError:
            return id(self)

    def __eq__(self, other):
        return isinstance(other, PyValue) and self.value == other.value or self.value == other

    def __iter__(self):
        match self.value:
            case tuple() | list() | set() | frozenset() as iterable:
                return iter(iterable)
            case str() as string:
                return (PyValue(BuiltIns['String'], c) for c in string)
        raise TypeErr(f"Line {state.line}: {self.table} {self} is not iterable.")

    def __repr__(self):
        match self.value:
            case frozenset() as f:
                return repr(set(f))
            case Fraction(numerator=n, denominator=d):
                return f"{n}/{d}"
            case None:
                return 'blank'
            case v:
                return repr(v)


class2table = dict(bool='Bool', int="Integer", Fraction='Fraction', float="Float", str='String', tuple="Tuple",
                   list='List', set='Set', frozenset='FrozenSet', dict='Function')
def py_value(value: T | object):
    match value:
        case None:
            return BuiltIns['blank']
        case True:
            return BuiltIns['true']
        case False:
            return BuiltIns['false']
        case Fraction(denominator=1, numerator=value) | (int() as value):
            return PyValue(BuiltIns['Integer'], value)
        case Fraction() | float() | str():
            table = BuiltIns[class2table[type(value).__name__]]
        # case list():
        #     return List(list(map(py_value, value)))
        case tuple() | list() | set() | frozenset():
            table = BuiltIns[class2table[type(value).__name__]]
            t = type(value)
            value = t(map(py_value, value))
        case dict() as d:
            return Function({py_value(k): py_value(v) for k, v in d.items()})
        case Record():
            return value
        # case Parameter():
        #     return ParamSet(value)
        # case Matcher() as t:
        #     return ParamSet(Parameter(t))
        case _:
            return PyObj(value)
    return PyValue(table, value)

def to_python(rec: Record):
    match rec:
        case PyValue(value=value):
            if isinstance(value, str):
                return value
            try:
                iter(value)
            except TypeError:
                return value
            t = type(value)
            return t(map(to_python, value))
        case PyObj(obj=obj):
            return obj
        case Function(op_map=op_map, op_list=op_list):
            d = {}
            for k, opt in op_map.items():
                key = to_python(k[0]) if len(k) == 1 else to_python(k)
                d[key] = to_python(opt.value)
            for opt in op_list:
                d[opt.pattern] = to_python(opt.resolution)
            return d
        case Args(positional_arguments=args, flags=flags, named_arguments=kwargs) as rec:
            if not flags and not kwargs:
                return tuple(map(to_python, args))
            return {to_python(k): to_python(v) for k, v in rec.dict().items()}
        case _:
            return rec


class PyObj(Record, Generic[A]):
    def __init__(self, obj):
        self.obj = obj
        super().__init__(BuiltIns['PythonObject'])

    def to_string(self, as_repr=False):
        return py_value(repr(self.obj))

    def get(self, name, *default, search_table_frame_too=False):
        return py_value(getattr(self.obj, name, *default))

    def call(self, *args, safe_call=False):
        match args:
            case [Args(positional_arguments=args, flags=flags, named_arguments=kwargs)]:
                pass
            case _:
                flags = ()
                kwargs = {}
        kwargs = {k: v.value for k, v in kwargs.items()}
        kwargs.update(dict(zip(flags, [True] * len(flags))))
        return py_value(self.obj(*(arg.value for arg in args), **kwargs))

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
                    return py_value(
                        obj(*map(extract_pyvalue, args), **{k: extract_pyvalue(v) for k, v in kwargs.items()}))
                case _:
                    raise Exception
            kwargs = {**args.named_arguments, **dict(zip(args.flags, [BuiltIns['true']] * len(args.flags)))}
            return fn(*args.positional_arguments, **kwargs)


class Range(Record):
    data: list[PyValue[int]]
    def __init__(self, *args: PyValue):
        step = py_value(1)
        match args:
            case []:
                start = py_value(1)
                end = py_value(-1)
            case [end]:
                start = step = py_value(1)
            case [start, end]:
                step = py_value(1)
            case [start, end, step]:
                # start, stop, step = (a.value for a in args)
                if step.value == 0:
                    raise RuntimeErr(f"Line {state.line}: Third argument in range (step) cannot be 0.")
            case _:
                raise RuntimeErr(f"Line {state.line}: Too many arguments for range")
        if start == BuiltIns['blank']:
            start = py_value(0)
        if end == BuiltIns['blank']:
            end = py_value(0)
        super().__init__(BuiltIns['Range'], start, end, step)

    def __iter__(self):
        start, end, step = tuple(f.value for f in self.data)
        if start is None:
            return
        i = start
        while i <= end if step > 0 else i >= end:
            yield py_value(i)
            i += step

    @property
    def slice(self) -> slice | None:
        start, end, step = (f.value for f in self.data)
        if start > 0:
            start -= 1
        if step < 0:
            if end < 0:
                end -= 1
            elif end in (0, 1):
                end = 0
            else:
                end -= 2
        else:
            if end < 0:
                end += 1

        """
        1..3      => [0:3]
        1..-2     => [0:-1]
        -1..-3:-1 => [-1:-4:-1]
        4..2:-1   => [3:0:-1]
        """
        return slice(start or None, end or None, step)


class Function(Record, OptionCatalog):
    frame = None

    def __init__(self, options=None, name=None,
                 table_name='Function', traits=(), frame=None, uninitialized=False, default=None):
        if name:
            self.name = name
        if uninitialized:
            self.uninitialized = True
        if frame:
            self.frame = frame        
        OptionCatalog.__init__(self, options or {}, *traits, default=default)
        super().__init__(BuiltIns[table_name])

    def get(self, name: str, *default, search_table_frame_too=False):
        if self.frame:
            val = self.frame[name]
            if val:
                return val
        return super().get(name, *default, search_table_frame_too=search_table_frame_too)

    def set(self, name: str, value: Record):
        if self.frame:
            if name in self.frame.vars:
                self.frame.vars[name] = value
            else:
                self.frame.locals[name] = value
            return value
        return super().set(name, value)

    def select(self, args):
        option, bindings = self.select_and_bind(args)
        if option is not None:
            return option, bindings
        return self.table.catalog.select_and_bind(args)

    # def to_dict(self):
    #     d = {}
    #     for k, v in self.op_map.items():
    #         key = k[0].value if len(k) == 1 else k.positional_arguments
    #         d[key] = v.value.value
    #     return d

    def to_string(self, as_repr=False):
        if self.name and not as_repr:
            return py_value(self.name)
        items = (f'{pili_repr(k[0] if len(k) == 1 else py_value(k.positional_arguments)).value}: '
                 f'{pili_repr(v.value).value if v.value else "..."}'
                 for k, v in self.op_map.items())
        options = (f'{pili_repr(opt.pattern).value} => ...' for opt in self.op_list)
        names = ()
        if self.frame:
            names = (f'{k} = {pili_repr(v).value}' for k, v in self.frame.items())
        segments = []
        items, options, names = tuple(items), tuple(options), tuple(names)
        if names:
            segments.append(names)
        if items:
            segments.append(items)
        if options:
            segments.append(options)
        ret = '{' + '; '.join(', '.join(seg) for seg in segments) + '}'
        if self.name:
            return py_value(f'<{self.__class__.__name__} "{self.name}" {ret}>')
        return py_value(ret)

    def __repr__(self):
        return f"Function({self.name or ''})"


class Trait(Function):
    # noinspection PyDefaultArgument
    def __init__(self, options={}, *fields, name=None, fn_options={}, fn_fields=[], uninitialized=False, default=None):
        self.options = [Option(patt, res) for (patt, res) in options.items()]
        self.fields = list(fields)
        super().__init__(fn_options, *fn_fields,
                         name=name, table_name='Trait', uninitialized=uninitialized, default=default)

    def upsert_field(self, field):
        for i, f in self.fields:
            if f.name == field.name:
                self.fields[i] = field
                return
        self.fields.append(field)

    def __repr__(self):
        return f"Trait({self.name or self.fields})"


class Table(Function):
    name = None
    catalog: OptionCatalog
    records: list[Record] | dict[Record, Record] | set[Record] | None
    # getters = dict[str, tuple[int, Field]]
    # setters = dict[str, tuple[int, Field]]
    # fields = list[PiliField]
    types: dict[str]  # Matcher
    # noinspection PyDefaultArgument
    def __init__(self, *traits: Trait, name: str = None, fn_options: dict = {}, fn_fields: list = [],
                 uninitialized: bool = False, default=None):
        self.traits = (Trait(name=name), *traits)
        self.getters = {}
        self.setters = {}
        self.defaults = ()
        super().__init__(fn_options, *fn_fields,
                         name=name, table_name='Table', traits=traits, uninitialized=uninitialized, default=default)
        if uninitialized:
            self.uninitialized = True
        else:
            self.integrate_traits()
        match self:
            case VirtTable():
                pass
            case ListTable():
                self.records = []
            case DictTable():
                self.records = {}
            case SetTable():
                self.records = set()
            case _:
                raise TypeError("Oops, don't use the Table class — use a derived class instead.")

    @property
    def trait(self):
        return self.traits[0]

    def raw_fields(self):
        for trait in self.traits:
            yield from trait.fields

    def integrate_traits(self):
        defaults: dict[str, Function | None] = {}
        types: dict[str, Pattern] | dict[str] = {}
        fields: dict[str, PiliField] = {}

        for trait in self.traits:
            # if trait.frame:
            #     for name, value in trait.frame.locals.items():
            #         if name not in self.frame:
            #             self.frame[name] = value
            # ^^ I thought about transferring all names from all traits into the current table, but that's a bit heavy
            for trait_field in trait.fields:
                name = trait_field.name
                if name in fields:
                    fields[name] += trait_field
                else:
                    fields[name] = PiliField(trait_field)

                pattern = getattr(trait_field, 'type', None)
                if pattern:
                    # if isinstance(pattern, Parameter):
                    #     assert pattern.binding == name
                    if name in types and not types[name].issubset(pattern):
                        raise SlotErr(f"Line {state.line}: Could not integrate table {self.name}; "
                                      f"type of Field \"{name}\" ({types[name]}) "
                                      f"doesn't match type of {trait_field.__class__.__name__} \"{name}\" "
                                      f"of trait {trait.name}.")
                    elif isinstance(trait_field, Setter):
                        pass  # types[name] = AnyMatcher()
                    else:
                        types[name] = pattern

                match trait_field:
                    case Slot(default=default):
                        # if slot: allow adding of default
                        # if formula: skip (formula should overwrite slot)
                        # if setter: skip (setter overwrites slot, hopefully a formula will also be defined)
                        if name in defaults:
                            # this means a slot was already defined.  Add a default if none exists
                            if defaults[name] is None:
                                defaults[name] = default
                        elif name not in self.getters and name not in self.setters:
                            # this means no slot, formula, or setter was defined.  So add a slot.
                            self.getters[name] = self.setters[name] = len(defaults)
                            defaults[name] = default

                    case Formula(formula=fn):
                        if name not in self.getters:
                            self.getters[name] = fn
                    case Setter(fn=fn):
                        if name not in self.setters:
                            self.setters[name] = fn
                        # types[name] = AnyMatcher()

        self.defaults = tuple(defaults[n] for n in defaults)
        self.types = types
        self.fields = fields
        patt = ParamSet(*(Parameter(types[name],
                                    name,
                                    "?" * (defaults[name] is not None))
                          for name in defaults))

        def make_option(table: Table):
            return Closure(lambda args: BuiltIns['new'].call(Args(table) + args))

        self.assign_option(patt,
                           make_option(self),
                           no_clobber=True)

        self.catalog = OptionCatalog({}, *self.traits)

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        if not isinstance(self.records, set):
            self.records[key] = value  # noqa

    def __contains__(self, item):
        return item.table == self

    def truthy(self):
        return bool(self.records)

    def add_record(self, record: Record):
        match self:
            case VirtTable():
                pass
            case ListTable():
                record.index = len(self.records)
                self.records.append(record)
            case DictTable():
                raise NotImplementedError() # self.records[record.key] = record
            case SetTable():
                self.records.add(record)

    def __repr__(self):
        if self.name:
            return self.name
        return f"Table({self.traits})"


class ListTable(Table):
    records: list[Record]

    # def __init__(self, *fields, name=None):
    #     super().__init__(*fields, name=name)
    #     self.records = []

    def __getitem__(self, key: PyValue[int]):
        try:
            return self.records[key.__index__()]
        except (TypeError, AttributeError):
            raise RuntimeErr(f"Index must be integer in range for ListTable.")
        except IndexError:
            return None


class MetaTable(ListTable):
    def __init__(self):
        self.name = 'Table'
        self.records = [self]
        self.table = self
        self.data = []
        self.index = 0
        self.traits = ()
        self.getters = {}
        self.setters = {}
        self.defaults = ()
        self.slot_dict = {}
        self.formula_dict = {}
        self.setter_dict = {}
        self.op_list = []
        self.op_map = {}
        self.catalog = OptionCatalog()


class BootstrapTable(ListTable):
    def __init__(self, name):
        self.name = name
        self.records = []
        self.traits = ()
        self.getters = {}
        self.setters = {}
        self.defaults = ()
        self.slot_dict = {}
        self.formula_dict = {}
        self.setter_dict = {}
        self.op_list = []
        self.op_map = {}
        self.catalog = OptionCatalog()
        Record.__init__(self, BuiltIns['Table'])


class DictTable(Table):
    records: dict[Record, Record]
    key_field: int

    def __init__(self, key_field: int = 0, *fields, name=None):
        self.key_field = key_field
        super().__init__(*fields, name=name)
        self.records = {}

    def __getitem__(self, key: Record):
        return self.records.get(key)


class SetTable(Table):
    records: set[Record]

    # def __init__(self, *fields, name=None):
    #     super().__init__(*fields, name=name)
    #     self.records = set([])
    def __getitem__(self, key: Record):
        return key


class VirtTable(SetTable):
    records = None

    # def __init__(self, *fields, name=None):
    #     self.records = None
    #     Table.__init__(self, *fields, name=name)

    @property
    def truthy(self):
        return True


class Field:
    type = None

    def __init__(self, name: str, type=None, default=None, formula=None):
        self.name = name
        if type:
            self.type = type
        # if default is None:
        #     default = py_value(None)
        # if formula is None:
        #     formula = py_value(None)
        # super().__init__(BuiltIns['Field'])
        # , name=py_value(name),
        # type=ParamSet(Parameter(type)) if type else BuiltIns['blank'],
        # is_formula=py_value(formula is not None),
        # default=default, formula=formula)


class Slot(Field):
    def __init__(self, name, type, default=None):
        match default:
            case Function(op_list=[Option(pattern=
                                          ParamSet(parameters=[Parameter()])
                                          )]):
                pass  # assert that default is a function whose sole option is [<patt> self]: ...
            case _ if default is not None:
                raise AssertionError('slot default did not match the appropriate function signature')
        self.default = default
        super().__init__(name, type, default)

    def to_string(self, as_repr=False):
        if not as_repr:
            return py_value(self.name)
        ret = ' = ' + pili_repr(self.default).value if self.default else ''
        return py_value(f"<slot {repr(self.name)} {pili_repr(self.type).value}{ret}>")

    def __repr__(self):
        return f"Slot({self.name}: {self.type}{' (' + str(self.default) + ')' if self.default else ''})"


class Formula(Field):
    def __init__(self, name, type, formula):
        if isinstance(formula, Function):
            self.formula = formula
        elif isinstance(formula, PyFunction):
            self.formula = Function({AnyMatcher(): formula})
        else:
            raise ValueError
        super().__init__(name, type, None, formula)

    def to_string(self, as_repr=False):
        if not as_repr:
            return py_value(self.name)
        return py_value(f"<formula {repr(self.name)} {pili_repr(self.formula).value}>")

    def __repr__(self):
        return f"Formula({self.name}: {str(self.formula)})"


class Setter(Field):
    fn: Function

    def __init__(self, name: str, fn: Function):
        if isinstance(fn, Function):
            self.fn = fn
        elif isinstance(fn, PyFunction):
            self.fn = Function({ParamSet(Parameter(AnyMatcher()), Parameter(AnyMatcher())): fn})
        else:
            raise ValueError
        super().__init__(name)

    def to_string(self, as_repr=False):
        if not as_repr:
            return py_value(self.name)
        return py_value(f"<setter {repr(self.name)} {pili_repr(self.fn).value}>")

    def __repr__(self):
        return f"Setter({self.name}: {self.fn})"


class Args(Record):
    # I tried to make Args no longer child of Record, but then the dot-operator fails to pattern match on it
    positional_arguments: list[Record] | tuple[Record, ...]
    named_arguments: dict[str, Record]
    flags: set[str]

    def __init__(self, *args: Record, flags: set[str] = None, named_arguments: dict[str, Record] = None,
                 **kwargs: Record):
        self.positional_arguments = args
        self.flags = flags or set()
        self.named_arguments = named_arguments or kwargs
        super().__init__(BuiltIns['Args'])

    def __len__(self):
        return len(self.positional_arguments) + len(self.named_arguments) + len(self.flags)

    def __getitem__(self, key):
        if key in self.flags:
            return BuiltIns['true']
        return self.named_arguments.get(key, self.positional_arguments[key])

    def try_get(self, key):
        match key:
            case str():
                return BuiltIns['true'] if key in self.flags else self.named_arguments.get(key, None)
            case int():
                try:
                    return self.positional_arguments[key]
                except IndexError:
                    return None
        raise TypeError(key)

    def __iter__(self):
        if self.flags or self.named_arguments:
            raise NotImplementedError
        return iter(self.positional_arguments)

    def keys(self):
        return self.named_arguments.keys()

    def __add__(self, other):
        match other:
            case Args(positional_arguments=pos, flags=flags, named_arguments=kwargs):
                pass
            case tuple() as pos:
                flags = set()
                kwargs = {}
            case set() as flags:
                pos = ()
                kwargs = {}
            case dict() as kwargs:
                pos = ()
                flags = set()
            case _:
                return NotImplemented
        return Args(*self.positional_arguments, *pos,
                    flags=self.flags.union(flags),
                    **self.named_arguments, **kwargs)

    def __radd__(self, other):
        match other:
            case Args(positional_arguments=pos, flags=flags, named_arguments=kwargs):
                pass
            case tuple() as pos:
                flags = set()
                kwargs = {}
            case set() as flags:
                pos = ()
                kwargs = {}
            case dict() as kwargs:
                pos = ()
                flags = set()
            case _:
                return NotImplemented
        return Args(*pos, *self.positional_arguments,
                    flags=self.flags.union(flags),
                    **self.named_arguments, **kwargs)

    def __eq__(self, other):
        return isinstance(other, Args) and self.dict() == other.dict()

    def dict(self):
        d = dict(enumerate(self.positional_arguments))
        d.update(self.named_arguments)
        for s in self.flags:
            d[s] = BuiltIns['true']
        return d

    def __hash__(self):
        d = self.dict()
        return hash((frozenset(d), frozenset(d.values())))

    def to_string(self, as_repr=False):
        pos = ', '.join(pili_repr(arg).value for arg in self.positional_arguments)
        names = ', '.join(f"{name}={pili_repr(value).value}"
                          for name, value in self.named_arguments.items())
        flags = ', '.join('!' + f for f in self.flags)
        string = f'[{"; ".join(item for item in (pos, names, flags) if item)}]'
        if as_repr:
            return f'<Args {string}>'
        return string

    def __str__(self):
        return repr(self)

    def __repr__(self):
        pos = map(str, self.positional_arguments)
        names = (f"{k}={v}" for (k, v) in self.named_arguments.items())
        flags = ('!' + str(f) for f in self.flags)
        return f"Args({', '.join(pos)}; {', '.join(names)}; {' '.join(flags)})"
    

class Pattern(Record):
    def __init__(self):
        super().__init__(BuiltIns['Pattern'])

    def match(self, arg: Record) -> None | dict[str, Record]:
        raise NotImplementedError(self.__class__.__name__)

    def issubset(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def match_and_bind(self, arg: Record):
        match = self.match(arg)
        if match is None:
            raise MatchErr(f"Line {state.line}: "
                           f"pattern '{self}' did not match value {arg}")
        state.env.update(match)
        return arg

    def merge(self, binding: str, default: Record = None):
        return BindingPattern(self, binding, default)

    def bytecode(self):
        raise NotImplementedError(self.__class__.__name__)

    def to_string(self, as_repr=False):
        return py_value(f'<{self.table} '
                        f'{' '.join(f"{k}={v}" for k, v in self.__dict__.items()
                                    if k not in {'hash', 'table', 'data', 'index'})}>')

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"{tuple(f"{k}={v}" for k, v in self.__dict__.items() if k not in {'hash', 'table', 'data', 'index'})}")

    def __str__(self):
        return repr(self)


class Matcher(Pattern):
    hash: int
    def match_score(self, arg: Record) -> None | dict[str, Record]:
        try:
            return self.match(arg)
        except NotImplementedError:
            if self.basic_score(arg):
                return {}

    def match(self, arg: Record) -> None | dict[str, Record]:
        if self.basic_score(arg):
            return {}

    def basic_score(self, arg):
        # implemented by subclasses
        raise NotImplementedError(self.__class__.__name__)

    def issubset(self, other):
        print('WARNING: Matcher.issubset method not implemented properly yet.')
        return self.equivalent(other)

    def equivalent(self, other):
        raise NotImplementedError(self.__class__.__name__)

    def get_rank(self):
        if hasattr(self, 'rank'):
            return self.rank
        raise NotImplementedError(self.__class__.__name__)

    def __lt__(self, other):
        return self.get_rank() < other.get_rank()

    def __le__(self, other):
        match other:
            case IntersectionMatcher(matchers=patterns):
                return all(self < p for p in patterns)
            case UnionMatcher(matchers=patterns):
                return any(self <= p for p in patterns)
            case Matcher():
                return self.get_rank() <= other.get_rank()
            # case Parameter(pattern=pattern):
            #     return self <= pattern

            case _:
                return NotImplemented

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"{tuple(v for k, v in self.__dict__.items() if k not in {'hash', 'table', 'data', 'index'})}")

    def __eq__(self, other):
        return (self.__class__ is other.__class__
                and hash(self) == hash(other)
                and self.__dict__ == other.__dict__)

    def __hash__(self):
        try:
            return self.hash
        except AttributeError:
            props = frozendict(self.__dict__, cls=self.__class__)
            self.hash = hash(props)
            return self.hash

class TableMatcher(Matcher):
    table_: Table
    rank = 5, 0

    def __init__(self, table):
        assert isinstance(table, Table)
        self.table_ = table
        super().__init__()

    def basic_score(self, arg: Record) -> bool:
        return arg.table == self.table_

    def issubset(self, other):
        match other:
            case TableMatcher(table_=table):
                return table == self.table_
            case TraitMatcher(trait=trait):
                return trait in self.table_.traits
        return False

    def equivalent(self, other):
        return isinstance(other, TableMatcher) and self.table_ == other.table

    def __str__(self):
        return str(self.table_)

class TraitMatcher(Matcher):
    trait: Trait
    rank = 6, 0

    def __init__(self, trait):
        self.trait = trait
        super().__init__()

    def basic_score(self, arg: Record) -> bool:
        return self.trait in arg.table.traits

    def issubset(self, other):
        return isinstance(other, TraitMatcher) and other.trait == self.trait

    def equivalent(self, other):
        return isinstance(other, TraitMatcher) and other.trait == self.trait

    def __str__(self):
        return str(self.trait)

class ValueMatcher(Matcher):
    value: Record
    rank = 1, 0

    def __init__(self, value):
        self.value = value
        super().__init__()

    def basic_score(self, arg: Record) -> bool:
        return arg == self.value

    def issubset(self, other):
        match other:
            case ValueMatcher(value=value):
                return value == self.value
            case TableMatcher(table_=table):
                return self.value.table == table
            case TraitMatcher(trait=trait):
                return trait in self.value.table.traits
        return False

    def equivalent(self, other):
        return isinstance(other, ValueMatcher) and other.value == self.value

    def __str__(self):
        return '@' + str(self.value)

class ArgsMatcher(Matcher):
    rank = 5, 0
    params = None  # : ParamSet | None = None

    def __init__(self, *params):
        match params:
            case [ParamSet() as params]:
                self.params = params
            case _ if params:
                self.params = ParamSet(*params)
        super().__init__()

    def match(self, arg: Record) -> None | dict[str, Record]:
        if not isinstance(arg, Args):
            return
        if self.params is None:
            return {}
        return self.params.match(arg)


class FunctionSignatureMatcher(Matcher):
    def __init__(self, pattern, return_type):
        self.pattern = pattern
        self.return_type = return_type
        super().__init__()

    def basic_score(self, arg):
        if not hasattr(arg, 'op_list'):
            return False
        arg: Function

        def options():
            yield from arg.op_list
            yield from arg.op_map.values()

        if all(option.pattern.issubset(self.pattern)
               and option.return_type.issubset(self.return_type)
               for option in options()):
            return True

    def issubset(self, other):
        match other:
            case FunctionSignatureMatcher(pattern=patt, return_type=ret):
                return self.pattern.issubset(patt) and self.return_type.issubset(ret)
            case TraitMatcher(trait=BuiltIns.get('fn')) | TableMatcher(table_=BuiltIns.get('Function')):
                return True
        return False

    def equivalent(self, other):
        return (isinstance(other, FunctionSignatureMatcher)
                and other.pattern == self.pattern
                and other.return_type == self.return_type)

    def __repr__(self):
        return f"FunctionSignatureMatcher({self.pattern} => {self.return_type})"

class KeyValueMatcher(Matcher):
    items: dict  # dict[Record, Parameter]
    def __init__(self, items: dict):
        self.items = frozendict(items)
        super().__init__()

    def match(self, arg):
        if not isinstance(arg, Function):
            return
        d = arg.op_map
        bindings = {}
        for key, param in self.items.items():
            try:
                value = d[Args(key)].value
            except KeyError:
                if param.required:
                    return
                continue
            sub_bindings = param.match(value)
            if sub_bindings is None:
                return None
            else:
                bindings.update(sub_bindings)
        return bindings


class AnyMatcher(Matcher):
    rank = 100, 0
    def __new__(cls):
        global ANYMATCHER
        ANYMATCHER = object.__new__(cls)
        Matcher.__init__(ANYMATCHER)
        cls.__new__ = lambda cls: ANYMATCHER
        return ANYMATCHER

    def basic_score(self, arg: Record) -> True:
        return True

    def issubset(self, other):
        return isinstance(other, AnyMatcher)

    def equivalent(self, other):
        return isinstance(other, AnyMatcher)

    def __str__(self):
        return 'any'

class EmptyMatcher(Matcher):
    rank = 3, 0
    def basic_score(self, arg: Record) -> bool:
        match arg:
            case VirtTable():
                return False
            case PyValue(value=str() | tuple() | frozenset() | list() | set() as v) | Table(records=v):
                return len(v) == 0
            case Function(op_list=options, op_map=hashed_options):
                return bool(len(options) + len(hashed_options))
            case Record(table=Table(traits=traits)) if BuiltIns['seq'] in traits:
                return BuiltIns['len'].call(arg).value == 0
            case _:
                return False

    def issubset(self, other):
        return isinstance(other, EmptyMatcher)

    def equivalent(self, other):
        return isinstance(other, EmptyMatcher)

    def __str__(self):
        return 'empty'

class IterMatcher(Matcher):
    # params: ParamSet
    def __init__(self, *params):
        match params:
            case [ParamSet() as params]:
                self.params = params
            case _ if params:
                self.params = ParamSet(*params)
        super().__init__()

    def match(self, arg: Record) -> None | dict[str, Record]:
        try:
            it = iter(arg)  # noqa
        except TypeError:
            return
        return self.params.match(Args(*it))


def dot_call_fn(rec: Record, *name_and_or_args, safe_get=False, swizzle=False, safe_call=False):
    match name_and_or_args:
        case [PyValue(value=str(name)) | str(name), *args]:
            args, = args or (None,)
        case [Args() as args]:
            return rec.call(args, safe_call=safe_call)
        case _:
            raise ValueError('either name or args should be non-None')

    if swizzle and swizzle.truthy:
        return py_value([dot_call_fn(item, name, args, safe_get=safe_get, safe_call=safe_call) for item in rec])

    if hasattr(rec, "uninitialized"):
        raise InitializationErr(f"Line {state.line}: "
                                f"Cannot call or get property of {limit_str(rec)} before initialization.")
    # 1. Try to resolve slot/formula in left
    prop = rec.get(name, None)  # , search_table_frame_too=True)
    if prop is not None:
        if args is None:
            return prop
        return prop.call(args, safe_call=safe_call)
    if args is None:
        args: Args = Args()
    # 2. Try to find function in table and traits
    for scope in (rec.table, *rec.table.traits):
        method = scope.get(name, None)
        if method is not None:
            return method.call(Args(rec) + args, safe_call=safe_call)
    # 3. Finally, try  to resolve name normally
    fn = state.deref(name, None)
    if fn is None:
        if safe_get:
            return BuiltIns['blank']
        raise KeyErr(f"Line {state.line}: {limit_str(rec)} has no slot '{name}' and no record with that "
                     f"name found in current scope either.")
    return fn.call(Args(rec) + args, safe_call=safe_call)


class FieldMatcher(Matcher):
    ordered_fields: tuple
    fields: dict  # dict[str, Parameter]
    def __init__(self, ordered_fields: tuple[Pattern, ...], fields: dict = None, **kwargs):
        self.ordered_fields = tuple(f if isinstance(f, Parameter) else Parameter(f)
                                    for f in ordered_fields)
        if fields is None:
            fields = kwargs
        else:
            fields.update(kwargs)
        for f, p in fields.items():
            if not isinstance(p, Parameter):
                fields[f] = Parameter(p)
        self.fields = frozendict(fields)
        super().__init__()

    def match(self, arg: Record) -> None | dict[str, Record]:
        bindings = {}
        for key, param in self.items():
            if isinstance(key, int):
                try:
                    prop = arg.data[key]
                except IndexError:
                    if param.required:
                        raise RuntimeErr(f"Line {state.line}: "
                                         f"{arg} does not have enough slots to unpack for Field Matcher.  "
                                         f"Try using fewer fields, or named fields instead.")
                    continue
            else:
                prop = dot_call_fn(arg, key, safe_get=True)
            if prop is BuiltIns['blank']:
                if param.required:
                    return None
                continue
            sub_bindings = param.match(prop)
            if sub_bindings is None:
                return None
            else:
                bindings.update(sub_bindings)
        return bindings

    def items(self):
        yield from self.fields.items()
        yield from enumerate(self.ordered_fields)

    def get_rank(self):
        return 2, -1

class ExprMatcher(Matcher):
    expr: any  # Node
    rank = 2, 0
    def __init__(self, expr):
        self.expr = expr
        super().__init__()

    def basic_score(self, arg: Record) -> bool:
        return self.expr.evaluate().truthy

class LambdaMatcher(Matcher):
    """ this matcher is only used internally, users cannot create LambdaMatchers """
    fn: PyFunction
    rank = 2, 0

    def __init__(self, fn: PyFunction):
        self.fn = fn
        super().__init__()

    def basic_score(self, arg: Record) -> bool:
        return self.fn(arg)


class NotMatcher(Matcher):
    def __init__(self, pattern: Pattern):
        self.pattern = pattern
        super().__init__()

    def match(self, arg: Record) -> None | dict[str, Record]:
        if self.pattern.match(arg) is None:
            return {}

    def get_rank(self):
        return AnyMatcher.rank[0] - self.pattern.get_rank()[0], 0

    def __lt__(self, other):
        return other <= self.pattern

    def __le__(self, other):
        return other < self.pattern

    def __str__(self):
        return "~" + str(self.pattern)

class IntersectionMatcher(Matcher):
    # I'm confused.  I think I made this inherit from "Pattern" rather than "Matcher" so that you could do
    # intersections of multiple parameters in a row
    # eg foo[(num+) & (int*, ratio*)]: ...
    # but somehow it's getting compared with matchers now.
    matchers: tuple[Matcher, ...]
    def __init__(self, *matchers: Matcher):
        # if binding is not None:
        #     raise Exception("This should be a parameter, not an Intersection.")
        match len(matchers):
            case 0:
                raise ValueError(f"Line {state.line}: IntersectionMatcher called with 0 matchers.  "
                                 f"Use AnyMatcher(invert=True) instead.")
            case 1:
                raise ValueError(f"Line {state.line}: IntersectionMatcher called with only 1 matcher."
                                 f"Catch this and return that single matcher na lang.")
        self.matchers = matchers
        super().__init__()

    def match(self, arg: Record) -> None | dict[str, Record]:
        bindings = {}
        for m in self.matchers:
            sub_match = m.match(arg)
            if sub_match is None:
                return
            bindings.update(sub_match)
        return bindings

    def get_rank(self):
        ranks = [m.get_rank()[0] for m in self.matchers]
        return tuple(sorted(ranks))

    def issubset(self, other):
        match other:
            case Matcher() as other_matcher:
                return any(m.issubset(other_matcher) for m in self.matchers)
            case IntersectionMatcher() as patt:
                return any(matcher.issubset(patt) for matcher in self.matchers)
            case UnionMatcher(matchers=patterns):
                return any(self.issubset(patt) for patt in patterns)
            case Parameter(pattern=pattern):
                return self.issubset(pattern)
        return False

    def __lt__(self, other):
        match other:
            case IntersectionMatcher(matchers=other_matchers):
                return any(self <= p for p in other_matchers)
            case UnionMatcher(matchers=patterns):
                return any(self <= p for p in patterns)
            case _:
                return any(p <= other for p in self.matchers)

    def __le__(self, other):
        match other:
            case IntersectionMatcher(matchers=other_matchers):
                return any(self <= p for p in other_matchers)
            case UnionMatcher(matchers=patterns):
                return any(self <= p for p in patterns)
            case _:
                return any(p <= other for p in self.matchers)

    def __str__(self):
        return '&'.join(map(str, self.matchers))

class UnionMatcher(Matcher):
    rank = 7, 0
    matchers: tuple[Matcher, ...]
    def __init__(self, *matchers):
        match len(matchers):
            case 0:
                raise ValueError(f"Line {state.line}: UnionMatcher called with 0 matchers.  "
                                 f"Use AnyMatcher() instead.")
            case 1:
                raise ValueError(f"Line {state.line}: UnionMatcher called with only 1 matcher."
                                 f"Catch this and return that single matcher na lang.")
        self.matchers = matchers
        super().__init__()

    def match(self, arg: Record) -> None | dict[str, Record]:
        for m in self.matchers:
            sub_match = m.match(arg)
            if sub_match is None:
                continue
            return sub_match

    def issubset(self, other):
        return all(p.issubset(other) for p in self.matchers)

    def __lt__(self, other):
        match other:
            case IntersectionMatcher():
                return all(p < other for p in self.matchers)
            case UnionMatcher(matchers=patterns):
                return self.matchers < patterns
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case UnionMatcher(matchers=patterns):
                return self.matchers <= patterns
            case Matcher() | IntersectionMatcher():
                return all(p < other for p in self.matchers)
            case _:
                raise NotImplementedError

    def __str__(self):
        return '|'.join(map(str, self.matchers))


class BindingPattern(Pattern):
    def __init__(self, pattern: Pattern, binding: str, default: Record = None):
        self.pattern = pattern
        self.binding = binding
        self.default = default
        super().__init__()

    def match(self, arg):
        bindings = self.pattern.match(arg)
        if bindings is None:
            if self.default:
                return {self.binding: self.default}
            return
        bindings[self.binding] = arg
        return bindings

    def merge(self, binding: str = None, default: Record = None):
        return BindingPattern(self.pattern, binding or self.binding, default or self.default)

    def __repr__(self):
        return (f"BindingPattern({self.pattern} {self.binding or '<no binding>'}"
                f"{' = ' + str(self.default) if self.default else ''})")

class Parameter(BindingPattern):
    pattern: Pattern
    binding: str = None
    quantifier: str  # "+" | "*" | "?" | "!" | ""
    optional: bool
    required: bool
    multi: bool
    default: Record = None
    # def __new__(cls, pattern, binding: str = None, quantifier="", default=None):
    #     #########################################################
    #     # remove this code once it is safe to do so
    #     if not isinstance(pattern, Pattern):
    #         if isinstance(pattern, Parameter):
    #             print(f'WARNING Line {state.line}: PARAMETER {repr(pattern)} passed pattern of another param!!!!')
    #             if pattern.quantifier:
    #                 raise ValueError("Double quantifier")
    #             pattern = pattern.pattern
    #         else:
    #             print(
    #                 f'WARNING Line {state.line}: Non-pattern {repr(pattern)} passed as pattern parameter to Parameter.')
    #             pattern = patternize(pattern)
    #     #########################################################
    #     if quantifier[:1] in ('+', '*'):
    #         if default is not None:
    #             raise ValueError(
    #                 f'Cannot instantiate multi Parameter with default:'
    #                 f' Parameter({pattern}, {binding}, {quantifier}, {default})')
    #         return object.__new__(MultiParam)
    #     if default and binding is None is getattr(pattern, 'binding', None):
    #         raise ValueError(f'Cannot instantiate non-binding parameter with a default: '
    #                          f' Parameter({pattern}, {binding}, {quantifier}, {default})')
    #     return object.__new__(SingleParam)

    def __init__(self, pattern, binding: str = None, quantifier="", default=None):
        while isinstance(pattern, Parameter):
            binding = binding or pattern.binding
            default = default or pattern.default
            pattern = pattern.pattern
        #########################################################
        # remove this code once it is safe to do so
        # if not isinstance(pattern, Pattern):
        #     if isinstance(pattern, Parameter):
        #         print(f'WARNING Line {state.line}: PARAMETER {repr(pattern)} passed pattern of another param!!!!')
        #         if pattern.quantifier:
        #             raise ValueError("Double quantifier")
        #         pattern = pattern.pattern
        #     else:
        #         print(f'WARNING Line {state.line}: Non-pattern {repr(pattern)} passed as pattern parameter to Parameter.')
        #         pattern = patternize(pattern)
        #########################################################
        super().__init__(pattern, binding)
        if default is not None:
            self.quantifier = quantifier or '?'
            if binding is None or self.multi:
                raise ValueError(f'Cannot instantiate {"multi" if self.multi else "non-binding"} Parameter with default:'
                                 f' Parameter({pattern}, {binding}, {quantifier}, {default})')
            self.default = default
        else:
            self.quantifier = quantifier
            if quantifier.startswith('?'):
                self.default = BuiltIns['blank']

        # self.quantifier = quantifier
        # if default is None and quantifier.startswith('?'):
        #     default = BuiltIns['blank']
        # elif default:
        #     if binding is None or self.multi:
        #         raise ValueError(f'Cannot instantiate {"multi" if self.multi else "non-binding"} Parameter with default:'
        #                          f' Parameter({pattern}, {binding}, {quantifier}, {default})')
        #     if quantifier == '':
        #         self.quantifier = '?'
        # if self.multi:
        #     self.binding = binding
        #     self.pattern = pattern
        # else:
        #     pattern = pattern.bind(binding, default)
        #     self.pattern, self.binding, self.default = pattern.pattern, pattern.binding, pattern.default
        # elif binding or default:
        #     self.pattern = pattern.bind(binding, default)
        #     self.binding = self.pattern.binding  # I don't think this should be necessary
        # else:
        #     self.pattern = pattern
        #     self.binding = getattr(pattern, 'binding', None)

        # if binding is None or self.multi:
        #     self.pattern = pattern
        #     if default:
        #         raise ValueError(f'Cannot instantiate {"multi" if self.multi else "non-binding"} Parameter with default:'
        #                          f' Parameter({pattern}, {binding}, {quantifier}, {default})')
        # else:
        #     if default is None and quantifier.startswith('?'):
        #         default = BuiltIns['blank']
        #     self.pattern = BindingPattern(pattern, binding, default)

    # @property
    # def binding(self):
    #     patt = self
    #     while patt := getattr(patt, 'pattern', None):
    #         if hasattr(patt, 'binding'):
    #             return patt.binding

    # @property
    # def default(self):
    #     patt = self
    #     while patt := getattr(patt, 'pattern', None):
    #         if hasattr(patt, 'default'):
    #             return patt.default

    optional = property(lambda self: self.default or self.quantifier[:1] in ('?', '*'))
    required = property(lambda self: self.default is None and self.quantifier[:1] in ('', '+'))
    multi = property(lambda self: self.quantifier[:1] in ('+', '*'))

    def merge(self, binding=None, default=None):
        return Parameter(self.pattern, binding or self.binding, self.quantifier, default or self.default)

    def match(self, arg: Record) -> None | dict[str, Record]:
        if self.multi:
            return ParamSet(self).match(arg)
        return super().match(arg)
        return self.pattern.match(arg)
        bindings = self.pattern.match(arg)
        if bindings is None:
            return
        if self.binding:
            bindings.update({self.binding: arg})
        return bindings

    def match_and_bind(self, arg):
        return super().match_and_bind(arg)

    def issubset(self, other):
        if not isinstance(other, Parameter):
            raise NotImplementedError(f"Not yet implemented Parameter.issubset({other.__class__})")
        if self.multi and not other.multi or self.optional and other.required:
            return False
        return self.pattern.issubset(other.pattern)

    def compare_quantifier(self, other):
        return "_?+*".find(self.quantifier) - "_?+*".find(other.quantifier)

    def bytecode(self):
        vm: list
        vm = [Inst().match(self.pattern, self.binding if not self.multi else None)]  # , self.default)]

        match self.quantifier:
            case '':
                pass
            case '?':
                prepend = [Inst().bind(self.binding, self.default)] * bool(self.binding) \
                          + [Inst().split(1, len(vm)+1)]
                vm[:0] = prepend
            case '+':
                vm.append(Inst().split(-len(vm), 1))
            case '*':
                vm = [Inst().jump(len(vm)+1), *vm, Inst().split(-len(vm), 1)]
            case '??':
                # prioritize the non-matching (default) branch
                prepend = [Inst().bind(self.binding, self.default)] * bool(self.binding) \
                          + [Inst().split(len(vm) + 1, 1)]
                vm[:0] = prepend
            case '+?':
                # prioritize the shortest branch
                vm.append(Inst().split(1, -len(vm)))
            case '*?':
                # prioritize the shortest branch
                vm = [Inst().jump(len(vm)+1), *vm, Inst().split(1, -len(vm))]
            case _:
                assert False

        if self.multi and self.binding:
            vm.insert(0, Inst().save(self.binding))
            vm.append(Inst().save(self.binding))

        return vm

    def __lt__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern < other.pattern
            case Matcher():
                return self < Parameter(other)
        return NotImplemented

    def __le__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern <= other.pattern
            case Matcher():
                return self <= Parameter(other)
        return NotImplemented

    def __eq__(self, other):
        match other:
            case Parameter() as param:
                pass
            case Matcher() | Pattern():
                param = Parameter(other)
            case ParamSet(parameters=(param, ), named_params={}):
                pass
            case _:
                return False
        return self.quantifier == param.quantifier and self.pattern == param.pattern and self.default == param.default

    def __hash__(self):
        return hash((self.pattern, self.quantifier, self.default))

    def __gt__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern > other.pattern
            case Matcher():
                return self > Parameter(other)
        return NotImplemented

    def __ge__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern >= other.pattern
            case Matcher():
                return self >= Parameter(other)
        return NotImplemented

    def __str__(self):
        patt = str(self.pattern) + ' ' if self.pattern is not AnyMatcher() else ''
        q = self.quantifier * (not (self.quantifier == '?' and self.default))
        default = ' = ' + str(self.default) if self.default else ''
        return patt + (self.binding or '') + q + default

    def __repr__(self):
        return (f"Parameter({self.pattern}{f' {self.binding}' if self.binding else ''}{self.quantifier}"
                f"{'='+str(self.default) if self.default else ''})")


class SingleParam(Parameter, BindingPattern):
    pass

class MultiParam(Parameter, BindingPattern):
    pass

class BindPropertyPattern(BindingPattern):
    def __init__(self, rec: Record, field_name: str, pattern: Pattern = None, default=None):
        super().__init__(pattern or AnyMatcher(), '', default)
        self.rec = rec
        self.field_name = field_name

    def match_and_bind(self, arg: Record):
        match = self.pattern.match(arg)
        if match is None:
            if self.default is None:
                raise MatchErr(f"Line {state.line}: "
                               f"pattern '{self}' did not match value {arg}")
            else:
                arg = self.default
        else:
            state.env.update(match)
        self.rec.set(self.field_name, arg)
        return arg

    def __eq__(self, other):
        return (isinstance(other, BindPropertyPattern)
                and self.rec == other.rec and self.field_name == other.field_name and self.default == other.default)
    def __hash__(self):
        return hash((super(), self.field_name, self.rec))

    def __repr__(self):
        return f"BindProperty({self.pattern} {self.rec}.{self.field_name}{' = ' + str(self.default) if self.default else ''})"


class BindKeyPattern(Parameter):
    def __init__(self, rec: Function, key, pattern: Pattern = None, default=None):
        super().__init__(pattern or AnyMatcher(), default=default)
        self.rec: Function = rec
        self.key = key

    def match_and_bind(self, arg: Record):  # Record | Closure
        match = self.pattern.match(arg)
        if match is None:
            if self.default is None:
                raise MatchErr(f"Line {state.line}: "
                               f"pattern '{self}' did not match value {arg}")
            else:
                arg = self.default
        else:
            state.env.update(match)
        self.rec.assign_option(self.key, arg)
        return arg

    def __eq__(self, other):
        return (isinstance(other, BindKeyPattern)
                and self.rec == other.rec and self.key == other.field_name and self.default == other.default)

    def __hash__(self):
        return hash((super(), self.key, self.rec))

    def __repr__(self):
        return f"BindKey({self.pattern} {self.rec}.{self.key}{' = ' +str(self.default) if self.default else ''})"

class VarPatt(Parameter):
    def __init__(self, name: str):
        super().__init__(AnyMatcher())
        self.var_name = name

    def match_and_bind(self, arg: Record) -> Record:
        if self.var_name in state.env.locals:
            raise DuplicateNameErr(f'Line {state.line}: Cannot declare var {self.var_name} because it is already a local.')
        state.env.vars[self.var_name] = arg
        return arg

    def __repr__(self):
        return f'VarPatt({self.var_name})'


class LocalPatt(VarPatt):
    def match_and_bind(self, arg: Record) -> Record:
        if self.var_name in state.env.vars:
            raise DuplicateNameErr(f'Line {state.line}: Cannot declare local {self.var_name} because it is already a var.')
        state.env.locals[self.var_name] = arg
        return arg

class ParamSet(Pattern):
    parameters: tuple
    named_params: frozendict
    names_of_ordered_params: frozenset
    allow_arbitrary_kwargs: str | bool | None
    vm: list
    def __init__(self, *parameters, named_params: dict = None, kwargs: str = None):
        self.parameters = parameters
        self.named_params = frozendict(named_params or {})
        self.names_of_ordered_params = frozenset(param.binding for param in parameters
                                                 if param.binding and not param.multi)
        self.allow_arbitrary_kwargs = kwargs
        # self.vm = VM(parameters)
        super().__init__()
        self.vm = []
        for param in self.parameters:
            self.vm.extend(param.bytecode())
        # if kwargs:
        #     self.vm.append(Inst().bind_remaining(kwargs))

    def prepend(self, param: Parameter):
        self.parameters = (param, *self.parameters)
        self.vm[:0] = param.bytecode()

    def issubset(self, other):
        return (isinstance(other, ParamSet)
                and all(p1.issubset(p2) for (p1, p2) in zip(self.parameters, other.parameters))
                and all(self.named_params[k].issubset(other.named_params[k])
                        for k in set(self.named_params).union(other.named_params)))

    def __len__(self):
        return len(self.parameters) + len(self.named_params)

    def __getitem__(self, item):
        return self.named_params.get(item, self.parameters[item])

    def to_tuple(self):
        if self.named_params:
            return None
        key: list[Record] = []
        for parameter in self.parameters:
            match parameter.pattern.matchers:
                case (ValueMatcher(value=value), ) if value.hashable():
                    key.append(value)
                case _:
                    return None
        return tuple(key)

    def to_args(self):
        pos_args = []
        names = {}
        for id, param in self:
            match param:
                case Parameter(quantifier='', pattern=ValueMatcher(value=val)):
                    pass
                case _:
                    return None
            if isinstance(id, int):
                pos_args.append(val)
            else:
                names[id] = val
        return Args(*pos_args, **names)

    def __iter__(self):
        yield from enumerate(self.parameters)
        yield from self.named_params.items()

    def match(self, args: Args | Record) -> None | dict[str, Record]:
        if not isinstance(args, Args):
            if BuiltIns['iter'] not in args.table.traits:
                return
            args = Args(*args)  # noqa
        kwargs: dict[str, Record] = dict(args.named_arguments)
        for f in args.flags:
            kwargs[f] = BuiltIns['true']
        bindings: dict[str, Record] = {}

        # check for agreement of named parameters
        for name, param in self.named_params.items():
            param: Parameter
            if name in kwargs:
                match = param.match(kwargs[name])
                if match is None:
                    return
                bindings.update(match)
                bindings[name] = kwargs[name]
                del kwargs[name]
            elif param.default:
                bindings[name] = param.default
            elif param.required:
                return

        if not self.allow_arbitrary_kwargs:
            # check for illegal kwargs
            for name in kwargs:
                if name not in self.names_of_ordered_params and name not in bindings:
                    return

        return virtual_machine(self.vm, args.positional_arguments, kwargs, bindings, self.allow_arbitrary_kwargs)

    def __lt__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters < other.parameters

    def __le__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters <= other.parameters

    def __eq__(self, other):
        return (isinstance(other, ParamSet)
                and self.parameters == other.parameters
                and self.named_params == other.named_params)

    def __hash__(self):
        return hash((self.parameters, self.named_params))

    def __gt__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters > other.parameters

    def __ge__(self, other):
        if not isinstance(other, ParamSet):
            return NotImplemented
        return self.parameters >= other.parameters

    def __str__(self):
        def comma_sep(params):
            return ', '.join(map(str, params))
        return f"[{comma_sep(self.parameters)}{'; ' + comma_sep(self.named_params) if self.named_params else ''}]"

    def __repr__(self):
        return f"ParamSet({', '.join(map(repr, self.parameters))}{'; ' + str(self.named_params) if self.named_params else ''})"

class Inst:
    opcode: str = 'tail'
    next = None  # Inst | None
    matcher: Matcher = None
    matchers: tuple[Matcher, ...] = ()
    i: int = None
    name: str = None
    binding: str = None
    default: Record = None
    branches = None
    Match = 'Match'
    MatchName = 'MatchName'
    MatchAll = 'MatchAll'
    Success = 'Success'
    Jump = 'Jump'
    Split = 'Split'
    Save = 'Save'
    Bind = 'Bind'
    BindRemaining = 'BindRemaining'
    BackRef = 'BackRef'
    Merge = 'Merge'
    # def __init__(self, opcode: str = 'tail', *, ch=None, i=None, next=None, branches=None, complements=None):
    #     self.opcode = opcode
    #     if ch is not None:
    #         self.ch = ch
    #     if i is not None:
    #         self.i = i
    #     if next is not None:
    #         self.next = next
    #     if branches is not None:
    #         self.branches = branches
    #     if complements is not None:
    #         self.complements = complements

    def match(self, matcher: Matcher, binding: str = None, default: Record = None):
        self.opcode = Inst.Match
        self.matcher = matcher
        self.binding = binding
        self.default = default
        return self

    def match_name(self, name: str, matcher: Matcher, next=None):
        self.opcode = Inst.MatchName
        self.name = name
        self.matcher = matcher
        self.next = next
        return self

    def match_all(self, *matchers: Matcher, binding: str = None, default: Record = None):
        self.opcode = Inst.MatchAll
        self.matchers = matchers
        self.binding = binding
        self.default = default
        return self

    def success(self):
        self.opcode = Inst.Success
        return self

    def jump(self, next=None):
        self.opcode = Inst.Jump
        self.next = next
        return self

    def split(self, next, *branches):
        self.opcode = Inst.Split
        self.next = next
        self.branches = branches
        return self

    def save(self, name: str, next=None):
        self.opcode = Inst.Save
        self.name = name
        self.next = next
        return self

    def bind(self, name: str, default):
        self.opcode = Inst.Bind
        self.name = name
        self.default = default
        return self

    def bind_remaining(self, name: str):
        self.opcode = Inst.BindRemaining
        self.name = name
        return self

    def back_ref(self, i: int, next=None):
        self.opcode = Inst.BackRef
        self.i = i
        self.next = next
        return self

    def merge(self, step, count=2, next=None):
        self.step = step
        self.count = count
        self.next = next
        return self

    def __iter__(self):
        node = self
        seen = {self}

        while True:
            yield node
            node = self.next
            if node in seen:
                raise ValueError("Cycle found.")
            seen.add(node)

    def __str__(self):
        if self.opcode == 'tail':
            return f"tail ({hex(hash(self) % 65536)[-4:]})"
        res = self.str_node().strip()
        return res + self.str_branches()

    def __repr__(self):
        match self.opcode:
            case Inst.Match:
                props = self.matcher, self.binding, self.default
            case Inst.Jump:
                props = self.next,
            case Inst.Split:
                props = self.next, *self.branches
            case Inst.Save:
                props = self.name, self.i
            case Inst.Bind:
                props = self.name, self.default
            case _:
                props = ()
        props = (str(el) for el in props if el is not None)
        return f"{self.opcode} {' '.join(props)}"

    def tree_repr(self):
        return '\n'.join(self.treer(1, 0, {}, []))

    def str_node(self):
        args = (f" {el}" for el in (self.matcher, self.name) if el is not None)
        return f"{self.opcode}{''.join(args)}"

    def str_branches(self, seen=None, max_depth=10):
        if max_depth == 0:
            return ' > ...'
        if seen is None:
            seen = set()
        if self in seen:
            return f' > (cycle {10-max_depth})'
        seen.add(self)
        try:
            if self.next:
                # if self.next in seen:
                #     return f' > (cycle {10 - max_depth})'
                # seen.add(self.next)
                return ' > ' + self.next.str_node() + ' > ' + self.next.str_branches(seen, max_depth-1)
            elif self.branches:
                return ' >> ' + ' / '.join(branch.str_node() + ' > ' + branch.str_branches(seen, max_depth-1)
                                           for branch in self.branches)
            else:
                return ''
        except:
            return ' > ???'

    def treer(self, idx: int, depth: int, seen: dict, acc: list, branch_sym=''):
        tab = ' ' * (depth-len(branch_sym)) + branch_sym + f"{idx}. "
        if self in seen:
            acc.append(f'{tab}(Cycle: {seen[self]}. {self.opcode})')
        else:
            seen[self] = idx
            acc.append(f"{tab}{self.str_node()}")
            if self.next:
                self.next.treer(idx + 1, depth + 4*(self.opcode == Inst.Match), seen, acc)
            elif self.branches:
                for node in self.branches:
                    node.treer(idx + 1, depth+2, seen, acc, '> ')
        return acc


class ThreadList(deque):
    pass

def virtual_machine(prog: list[Inst], args: tuple, kwargs: dict, initial_bindings: dict, allow_arbitrary_kwargs=False):
    arg_idx: int
    arg: Record
    # kwargs = dict(args.named_arguments)
    # args = args.positional_arguments
    initial_bindings = frozendict(initial_bindings)

    def outer_loop():
        yield from enumerate(args)
        yield len(args), None

    def make_thread(step_idx: int, bindings=initial_bindings, saved=frozendict(), kwargs=frozendict(kwargs)):  # noqa
        # arg: Record = args[arg_idx]
        while step_idx < len(prog):
            step = prog[step_idx]
            match step.opcode:
                case Inst.Match:
                    if step.binding in kwargs:
                        # if step.binding in bindings:
                        #     print('DEBUG NOTE: what if positional param already matched by named arg?')
                        #     pass
                        if (match := step.matcher.match(kwargs[step.binding])) is not None:
                            bindings += match
                            bindings += {step.binding: kwargs[step.binding]}
                            kwargs -= step.binding
                        else:
                            # name found in kwargs, but did not match.  No possibility of any threads succeeding.
                            yield 'WHOLE PATTERN FAILURE'
                    elif arg is not None and (match := step.matcher.match(arg)) is not None:
                        bindings += match
                        if step.binding:
                            bindings += {step.binding: arg}
                        yield 'NEXT'
                    # elif step.default:
                    #     bindings += {step.binding: step.default}
                    else:
                        yield 'DEAD'
                case Inst.Jump:
                    step_idx += step.next
                    continue
                case Inst.Split:
                    for branch in reversed(step.branches):
                        current.append(make_thread(step_idx + branch, bindings, saved, kwargs))
                    step_idx += step.next
                    continue
                case Inst.Save:
                    item = saved.get(step.name, None)
                    if item is None:
                        saved += {step.name: arg_idx}
                    else:
                        saved += {step.name: slice(item, arg_idx)}
                case Inst.Bind:
                    bindings += {step.name: step.default}
                # case Inst.BindRemaining:
                #     remaining = {k: v for k, v in kwargs.items() if k not in bindings}
                #     bindings += {step.name: py_value(remaining)}
                # case Inst.Merge:
                #     for slc in reversed(step.slices):
                #         matched, bindings = virtual_machine(prog[slc], args, kwargs)
                case _:
                    raise AssertionError("unrecognized opcode: ", step.opcode)
            step_idx += 1
        # prog reached the end
        if arg_idx == len(args) and (not kwargs or allow_arbitrary_kwargs):
            # and all(name in bindings or name in saved for name in kwargs):
            # SUCCESS
            bindings = dict(bindings)
            if allow_arbitrary_kwargs:
                bindings.update(kwargs)
            for name, part in saved.items():
                bindings[name] = py_value(tuple(args[part]))
            yield bindings
        yield 'DEAD'

    current = ThreadList([make_thread(0)])
    pending = ThreadList()

    for arg_idx, arg in outer_loop():
        while current:
            thread = current.pop()
            match next(thread):
                case dict() as bindings:
                    return bindings
                case 'WHOLE PATTERN FAILURE':
                    return
                case 'NEXT':
                    pending.appendleft(thread)
                case 'DEAD':
                    pass
                case other:
                    raise AssertionError(f"thread yielded unexpected {other}")
        current, pending = pending, ThreadList()
    # return 0, {}


class RegEx(Pattern, PyValue[str]):
    value: str
    flags: str

    def __init__(self, value: str, flags: str = ''):
        PyValue.__init__(self, BuiltIns['RegEx'], value)
        self.data = [py_value(flags)]

    @property
    def flags(self):
        return self.data[0].value

    def match(self, arg: PyValue[str]) -> None | dict[str, Record]:
        if re.match(self.value, arg.value) is not None:
            return {}


def patternize(val) -> Pattern:
    match val:
        case Pattern():
            return val
        case Table():
            return TableMatcher(val)
        case Trait():
            return TraitMatcher(val)
        case Record():
            return ValueMatcher(val)
        case _:
            raise TypeErr(f"Line {state.line}: Could not patternize {val}")


class Closure:
    """ essentially just a block of code together with the context (Frame) in which it was bound to a function-option """
    # block: Block  # Block class not yet defined
    scope = None

    def __init__(self, block):
        if isinstance(block, PyFunction):
            self.fn = block
        else:
            self.block = block
        self.scope = state.env

    def execute(self, args=None, bindings=None, *, fn=None, option=None, link_frame=None):
        fn = fn or link_frame
        path = state.env.file_path
        if hasattr(self, 'block') and self.block.pos:
            path = self.block.pos.file.path
        env = Frame(self.scope, args, bindings, fn, option, path)
        if link_frame:
            link_frame.frame = env
        state.push(env, fn, option)
        val = BuiltIns['blank']
        if hasattr(self, 'block'):
            val = self.block.execute()
        else:
            env.return_value = self.fn(args)
        state.pop()
        return env.return_value or val

    def __repr__(self):
        return f"Closure({len(self.block.statements)})"


# class Native(Closure):
#     # I don't think the motivation for this subclass is sound.
#     # I think I've already eliminated the need for it in the two list_get options,
#     # now I need to eliminate it from the Table.integrate_traits method
#     def __init__(self, fn: PyFunction):
#         print('DEPRECATION WARNING: Native(Closure).__init__', fn)
#         self.fn = fn
#         self.scope = state.env
#
#     def execute(self, args=None, caller=None, bindings=None, *, fn=None):
#         print('DEPRECATION WARNING: Native(Closure).execute: ', self.fn)
#         assert args is not None or fn is not None
#         env = Frame(self.scope, args, caller, bindings, fn)
#         state.push(state.line, env)
#         line = state.line
#         if isinstance(args, tuple):
#             env.return_value = self.fn(*args)
#         else:
#             env.return_value = self.fn(args)
#         state.line = line
#         state.pop()
#         return env.return_value or caller or fn
#
#     def __repr__(self):
#         return f"Native({self.fn})"

class Frame:
    return_value = None

    def __init__(self, scope, args=None, bindings=None, fn=None, option=None, file_path=None):
        # self.names = bindings or {}
        self.vars = {}
        self.locals = bindings or {}
        # self.block = block
        # self.scope = code_block.scope
        self.scope = scope
        self.args = args
        self.fn = fn
        self.option = option
        self.file_path = file_path or state.source_path

    def assign(self, name: str, value: Record):
        scope = self
        while scope:
            if name in scope.vars:
                scope.vars[name] = value
                return value
            scope = scope.scope
        self.locals[name] = value
        # if isinstance(self.fn, Function):
        #     self.fn.slot_dict[name] = value
        return value

    def __getitem__(self, key: str):
        return self.vars.get(key, self.locals.get(key, None))

    def update(self, bindings: dict):
        for name, rec in bindings.items():
            self.assign(name, rec)

    def items(self):
        yield from self.locals.items()
        yield from self.vars.items()

    def __repr__(self):
        return (f"Frame({len(self.vars) + len(self.locals)} names; "
                f"{'running' if self.return_value is None else 'finished: ' + str(self.return_value)})")


class GlobalFrame(Frame):
    file_path = None
    block = None
    scope = None
    args = None
    caller = None
    fn = None

    def __init__(self, bindings: dict[str, Record]):
        self.vars = {}
        self.locals = bindings


class Option(Record):
    value = None
    block = None
    fn = None
    alias = None
    dot_option = False
    return_type = None

    def __init__(self, pattern, resolution=None):
        match pattern:
            case ParamSet():
                self.pattern = pattern
            case Parameter() as param:
                self.pattern = ParamSet(param)
            case _:
                self.pattern = ParamSet(Parameter(patternize(pattern)))
        if resolution is not None:
            self.resolution = resolution
        super().__init__(BuiltIns['Option'])  # , signature=self.pattern, code_block=self.resolution)

    # def is_null(self):
    #     return (self.value and self.block and self.fn and self.alias) is None
    # def not_null(self):
    #     return (self.value or self.block or self.fn or self.alias) is not None
    def nullify(self):
        if self.value is not None:
            del self.value
        if self.block is not None:
            del self.block
        if self.fn is not None:
            del self.fn
        if self.alias is not None:
            del self.alias

    def set_resolution(self, resolution):
        if self.alias:
            self.alias.set_resolution(resolution)
            return
        self.nullify()
        match resolution:
            case Closure():
                self.block = resolution
            case PyFunction():
                self.fn = resolution
            case Option():
                self.alias = resolution
            case Record():
                self.value = resolution
                self.return_type = ValueMatcher(resolution)
            # case _:  # assume type Block (not defined yet)
            #     self.block = resolution
            case _:
                raise ValueError(f"Line {state.line}: Could not assign resolution {resolution} to option {self}")

    def get_resolution(self):
        if self.value is not None:
            return self.value
        return self.block or self.fn or self.alias

    resolution = property(get_resolution, set_resolution, nullify)

    def resolve(self, args, bindings=None, caller=None):
        if self.alias:
            return self.alias.resolve(args, bindings, caller)
        if self.value is not None:
            return self.value
        if self.fn:
            return call(self.fn, args)
        if self.block:
            return self.block.execute(args, bindings, fn=caller, option=self)
        raise NoMatchingOptionError(f"Line {state.line}: Could not resolve null option")

    def __eq__(self, other):
        return isinstance(other, Option) and (self.pattern, self.resolution) == (other.pattern, other.resolution)

    def __repr__(self):
        if self.value:
            return f"Opt({self.pattern}={self.value})"
        if self.block or self.fn:
            return f"Opt({self.pattern}: {self.block or self.fn})"
        if self.alias:
            return f"Opt({self.pattern} -> {self.alias})"
        return f"Opt({self.pattern} -> null)"

class PiliField(Record):
    name: str
    type: Pattern = None
    getter: Function | bool = None
    setter: Function | bool = None
    default: Record = None
    def __init__(self, field: Field):
        self.name = field.name
        match field:
            case Slot(type=type, default=default):
                self.type = type
                self.default = default
                self.getter = self.setter = True
            case Formula(type=type, formula=getter):
                self.type = type
                self.getter = getter
            case Setter(fn=setter):
                self.setter = setter
        super().__init__(BuiltIns["Field"])

    def wrap_default(self):
        if self.default is None:
            return BuiltIns['blank']
        frame = Frame(state.root, bindings={'value': self.default})
        return Function({}, frame=frame)

    def __iadd__(self, other: Field):
        if self.name != other.name:
            raise ValueError("Incompatible fields: different name")
        if self.type is not None and other.type is not None and not self.type.issubset(other.type):
            raise ValueError('Incompatible field types')
        if self.type is None and (type := getattr(other, 'type', None)) is not None:
            self.type = type
        match other:
            case Slot(default=default):
                if self.default is None:
                    self.default = default
                if not self.getter:
                    self.getter = True
                if not self.setter:
                    self.setter = True
            case Formula(formula=getter):
                if not self.getter:
                    self.getter = getter
            case Setter(fn=setter):
                if not self.setter:
                    self.setter = setter

