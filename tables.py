import math
from fractions import Fraction
from Env import *
from typing import TypeVar, Generic

PyFunction = type(lambda: None)

def memoize(fn):
    memory = {}
    if type(fn) == PyFunction:
        def inner(*args):
            try:
                return memory[args]
            except KeyError:
                result = fn(*args)
                memory[args] = result
                return result
    elif type(fn) == type(list.append):
        def inner(self, *args):
            try:
                return memory[self, args]
            except KeyError:
                result = fn(self, *args)
                memory[self, args] = result
                return result
    else:
        raise TypeError
    return inner

def memoize_property(fn):
    memory = {}

    def inner(self):
        try:
            return memory[self]
        except KeyError:
            result = fn(self)
            memory[self] = result
            return result
    return property(inner)

def dependencies(*slots: str):
    def memoize(fn):
        memory = {}

        def wrapper(self: Record):
            state = self, tuple(self.get(slot) for slot in slots)
            if state in memory:
                return memory[state]
            result = fn(self)
            memory[state] = result
            return result
        return wrapper
    return memoize

class frozendict(dict):
    def __hash__(self):
        try:
            return hash((frozenset(self), frozenset(self.values())))
        except TypeError:
            return hash(frozenset(self))

class OptionCatalog:
    op_list: list
    op_map: dict
    # noinspection PyDefaultArgument
    def __init__(self, options={}, *traits):
        self.op_list = []
        self.op_map = {}
        if options:
            for patt, res in options.items():
                self.assign_option(Option(patt, res))
        for trait in traits:
            for option in trait.options:
                self.assign_option(option)

    def assign_option(self, pattern, resolution=None, *, no_clobber=False):
        match pattern, resolution:
            case Option(pattern=pattern, resolution=resolution) as option, None:
                pass
            case _:
                if resolution is None:
                    raise AssertionError("Why are you trying to add a null option?")
                option = Option(pattern, resolution)
                pattern = option.pattern

        key = pattern.to_args()
        if key is not None:
            if not no_clobber or key not in self.op_map:
                self.op_map[key] = option
                return option
            else:
                return self.op_map[key]

        if Context.settings['sort_options']:
            for i, opt in enumerate(self.op_list):
                if option.pattern <= opt.pattern:
                    self.op_list.insert(i, option)
                    break
                elif option.pattern == opt.pattern and not no_clobber:
                    self.op_list[i] = option
                    break
            else:
                self.op_list.append(option)
        elif opt := self.select_by_pattern(pattern):
            opt.resolution = resolution
        else:
            self.op_list.append(option)

        return option

    def remove_option(self, pattern):
        opt = self.select_by_pattern(pattern)
        if opt is None:
            raise NoMatchingOptionError(f'cannot find option "{pattern}" to remove')
        opt.nullify()

    def select_and_bind(self, key):
        if isinstance(key, tuple):
            key = Args(*key)
        if key in self.op_map:
            return self.op_map[key], {}
        # match key:
        #     case tuple() if key in self.op_map:
        #         return self.op_map[key], {}
        #     case Args():
        #         if key in self.op_map:
        #             return self.op_map[key], {}
        #     case Args(positional_arguments=pos) if not (key.named_arguments or key.flags):
        #         if pos in self.op_map:
        #             return self.op_map[pos], {}

        option = bindings = None
        high_score = 0
        for opt in self.op_list:
            score, saves = opt.pattern.match_zip(key)
            if score == 1:
                return opt, saves
            if score > high_score:
                high_score = score
                option, bindings = opt, saves
        return option, bindings

    def select_by_pattern(self, patt, default=None):
        # return [*[opt for opt in self.op_list if opt.pattern == patt], None][0]
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
            raise RuntimeErr(f"Line {Context.line}: too many values provided for creating new instance of {self.table};"
                             f" Expected a maximum of {len(self.table.defaults)} values but got {i}: {data_tuple}")
        defaults = (val.call(self) if val else BuiltIns['blank'] for val in self.table.defaults[i:])
        self.data = [*data_tuple, *defaults]
        for k, v in data_dict.items():
            self.set(k, v)
            
        table.add_record(self)

    # @property
    # def mro(self):
    #     yield from self.table.traits

    # key property
    # def get_key(self):
    #     match self.table:
    #         case ListTable():
    #             return py_value(self.index)
    #         case DictTable(key_field=fid):
    #             return self.get_by_index(fid)
    #         case SetTable():
    #             return self
    #     raise RuntimeErr
    # def set_key(self, new_key):
    #     match self.table:
    #         case ListTable():
    #             raise RuntimeErr(f"Cannot set automatically assigned key (index)")
    #         case DictTable(key_field=fid):
    #             self.set_by_index(fid, new_key)
    #         case SetTable():
    #             raise RuntimeErr(f"Cannot set key of SetTable.")
    #         case _:
    #             raise RuntimeErr
    # key = property(get_key, set_key)

    def get(self, name: str, *default):
        if name in self.table.getters:
            match self.table.getters[name]:
                case int() as idx:
                    return self.data[idx]
                case Function() as fn:
                    return fn.call(self)

        if not default:
            raise SlotErr(f"Line {Context.line}: no field found with name '{name}'.")
        return default[0]

    def set(self, name: str, value):
        match self.table.setters.get(name):
            case int() as idx:
                # TODO: check for type agreement
                self.data[idx] = value
                return BuiltIns['blank']
            case Function() as fn:
                return fn.call(self, value)
            case None:
                raise SlotErr(f"Line {Context.line}: no field found with name '{name}'.")

    def call(self, *args, flags=None, **kwargs):
        return self(Args(*args, flags=flags, **kwargs))
        # if not isinstance(args, Args):
        #     args = Args(args)
        # option, bindings = self.select_by_args(args)
        # if option:
        #     return option.resolve(args, self, bindings)
        # raise NoMatchingOptionError(f"Line {Context.line}: key {args} not found in {self.name or self}")

        # option, bindings = self.select(*key)
        # if option:
        #     return option.resolve(key, self, bindings)
        # raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in {self.name or self}")

    def __call__(self, args):
        option, bindings = self.select(args)
        if option:
            return option.resolve(args, self, bindings)
        raise NoMatchingOptionError(f"Line {Context.line}: {args} not found in {self.name or self}")

    def select(self, args):
        return self.table.catalog.select_and_bind(args)
    #     for t in self.mro:
    #         option, bindings = t.select_and_bind(args)
    #         if option:
    #             return option, bindings
    #     return None, None

    def hashable(self):
        try:
            return isinstance(hash(self), int)
        except TypeError:
            return False

    # @property
    # def truthy(self):
    #     return True

    def to_string(self):
        if self.name:
            return py_value(self.name)
        # if self.instanceof(BuiltIns['BasicType']):
        #     return Value(self.name)
        return py_value(str(self))

    # def __eq__(self, other):
    #     if not isinstance(other, Record):
    #         return False
    #     if isinstance(self.table, VirtTable) or isinstance(other.table, VirtTable):
    #         return id(self) == id(other)
    #     return (self.table, self.key) == (other.table, other.key)
    #
    # def __hash__(self):
    #     return id(self)

    def __repr__(self):
        return f"Record<{self.table}>({self.data})"

    # def __eq__(self, other):
    #     if getattr(self, "value", object()) == getattr(other, "value", object()) and self.value is not NotImplemented:
    #         return True
    #     if self is other:
    #         return True
    #     if not isinstance(other, Function):
    #         return False
    #     if getattr(self, "value", object()) == getattr(other, "value", object()) and self.value is not NotImplemented:
    #         return True
    #     if self.type is not other.type:
    #         return False
    #     if self.env != other.env or self.name != other.name:
    #         return False
    #     # for opt in self.options:
    #     #     if opt.resolution != getattr(other.select_by_pattern(opt.pattern), 'resolution', None):
    #     #         return False
    #     return True


T = TypeVar('T', None, bool, int, Fraction, float, str, tuple, frozenset, set, list)
A = TypeVar('A')
class PyValue(Record, Generic[T]):
    def __init__(self, table, value: T):
        self.value = value
        # self.slices = list(slices)
        super().__init__(table)

    def to_string(self):
        for singleton in ('true', 'false', 'blank'):
            if self is BuiltIns[singleton]:
                return py_value(singleton)
        if BuiltIns['num'] in self.table.traits:
            return py_value(write_number(self.value, Context.settings['base']))
        # if self.instanceof(BuiltIns['list']):
        #     return py_value(f"[{', '.join(v.to_string().value for v in self.value)}]")
        if isinstance(self.value, frozenset):
            return py_value(str(set(self.value)))
        return py_value(str(self.value))

    def __index__(self) -> int | None:
        if not isinstance(self.value, int | bool):
            raise TypeErr(f"Line {Context.line}: Index must be integer, not {self.table}")
        if not self.value:
            # return None  -->  TypeError: __index__ returned non-int (type NoneType)
            raise ValueError(f"Line {Context.line}: Pili indices start at 1, so 0 is not a valid index.")
        return self.value - (self.value > 0)

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
        raise TypeErr(f"Line {Context.line}: {self.table} {self} is not iterable.")

    def __repr__(self):
        match self.value:
            case frozenset() as f:
                return repr(set(f))
            case Fraction(numerator=n, denominator=d):
                return f"{n}/{d}"
            case v:
                return repr(v)
        # return f"Record({self.value})"


class2table = dict(bool='Bool', int="Integer", Fraction='Fraction', float="Float", str='String', tuple="Tuple",
                   list='List', set='Set', frozenset='FrozenSet', dict='Dictionary')

def py_value(value: T):
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
        case dict():
            raise NotImplementedError
        case Record():
            return value
        case Parameter():
            return ArgsMatcher(value)
        case Matcher() as t:
            return ArgsMatcher(Parameter(t))
        case _:
            return PyObj(value)
    return PyValue(table, value)

piliize = py_value

class PyObj(Record, Generic[A]):
    def __init__(self, obj):
        self.obj = obj
        super().__init__(BuiltIns['PythonObject'])

    def to_string(self):
        return py_value(repr(self.obj))

List = py_value

class Function(Record, OptionCatalog):
    def __init__(self, options=None, *fields, name=None, table_name='Function', traits=()):
        if name:
            self.name = name
        self.slot_dict = {}
        self.formula_dict = {}
        self.setter_dict = {}
        for f in fields:
            self.update_field(f)
        # self.trait = Trait(options, *fields)
        OptionCatalog.__init__(self, options or {}, *traits)
        super().__init__(BuiltIns[table_name])

    # @property
    # def mro(self):
    #     if self.trait:
    #         yield self.trait
    #     yield from super().mro
    #     # return self.trait, *super().mro

    def update_field(self, field):
        name = field.name
        match field:
            case Slot(default=default):
                self.slot_dict[name] = default.call(self)
            case Formula(formula=fn):
                self.formula_dict[name] = fn
            case Setter(fn=fn):
                self.setter_dict[name] = fn

    def get(self, name: str, *default):
        if name in self.slot_dict:
            return self.slot_dict[name]
        if name in self.formula_dict:
            return self.formula_dict[name].call(self)
        return super().get(name, *default)

    def set(self, name: str, value: Record):
        if name in self.slot_dict:
            self.slot_dict[name] = value
            return value
        if name in self.setter_dict:
            return self.setter_dict[name].call(self, value)
        return super().set(name, value)

    def select(self, args):
        option, bindings = self.select_and_bind(args)
        if option is not None:
            return option, bindings
        return self.table.catalog.select_and_bind(args)

    def __repr__(self):
        if self is Context.root:
            return 'root'
        return f"Function({self.name or ''})"


class Trait(Function):
    # trait = None
    # noinspection PyDefaultArgument
    def __init__(self, options={}, *fields, name=None, fn_options={}, fn_fields=[]):
        # if name:
        #     self.name = name
        # if own_trait:
        #     self.trait = own_trait
        self.options = [Option(patt, res) for (patt, res) in options.items()]
        # self.hashed_options = {}
        # self.field_ids = {}
        self.fields = list(fields)
        # if options:
        #     for patt, val in options.items():
        #         self.add_option(patt, val)
        # for field in fields:
        #     for field in fields:
        #         self.fields.append(field)
        super().__init__(fn_options, *fn_fields, name=name, table_name='Trait')

    # def add_option(self, pattern, resolution=None):
    #     option = Option(pattern, resolution)
    #
    #     # try to hash option
    #     key: list[Record] = []
    #     for parameter in option.pattern.parameters:
    #         t = parameter.matcher
    #         if isinstance(t, ValueMatcher) and t.guard is None and not t.invert and t.value.hashable():
    #             key.append(t.value)
    #         else:
    #             if Context.settings['sort_options']:
    #                 for i, opt in enumerate(self.options):
    #                     if option.pattern <= opt.pattern:
    #                         self.options.insert(i, option)
    #                         break
    #                     elif option.pattern == opt.pattern:
    #                         self.options[i] = option
    #                         break
    #                 else:
    #                     self.options.append(option)
    #             else:
    #                 self.options.append(option)
    #             break
    #     else:
    #         self.hashed_options[tuple(key)] = option
    #
    #     return option
    #
    # def remove_option(self, pattern):
    #     opt = self.select_by_pattern(pattern)
    #     if opt is None:
    #         raise NoMatchingOptionError(f'cannot find option "{pattern}" to remove')
    #     opt.nullify()
    #
    # def assign_option(self, pattern, resolution=None):
    #     opt = self.select_by_pattern(pattern)
    #     if opt is None:
    #         return self.add_option(pattern, resolution)
    #     else:
    #         opt.resolution = resolution
    #     return opt
    #
    # def select_and_bind(self, key):
    #     match key:
    #         case tuple() if key in self.hashed_options:
    #             return self.hashed_options[key], {}
    #         case Args(positional_arguments=pos) if not (key.named_arguments or key.flags):
    #             if pos in self.hashed_options:
    #                 return self.hashed_options[pos], {}
    #
    #     option = bindings = None
    #     high_score = 0
    #     for opt in self.options:
    #         score, saves = opt.pattern.match_zip(key)
    #         if score == 1:
    #             return opt, saves
    #         if score > high_score:
    #             high_score = score
    #             option, bindings = opt, saves
    #     return option, bindings
    #
    # def select_by_pattern(self, patt, default=None):
    #     # return [*[opt for opt in self.options if opt.pattern == patt], None][0]
    #     for opt in self.options:
    #         if opt.pattern == patt:
    #             return opt
    #     return default

    def upsert_field(self, field):
        for i, f in self.fields:
            if f.name == field.name:
                self.fields[i] = field
                return
        self.fields.append(field)
        # if field.name in self.field_ids:
        #     fid = self.field_ids[field.name]
        #     self.fields[fid] = field
        # else:
        #     self.field_ids[field.name] = len(self.fields)
        #     self.fields.append(field)

    # def add_own_option(self, pattern, resolution=None):
    #     if self.trait is None:
    #         self.trait = Trait()
    #     return self.add_option(pattern, resolution)

    def __repr__(self):
        return f"Trait({self.name or self.fields})"
        # match self.options, self.hashed_options:
        #     case _ if self.name:
        #         return f"Trait({self.name})"
        #     case [], {}:
        #         return "Trait()"
        #     case [opt], {}:
        #         return f"Trait({opt})"
        #     case [], dict() as d if len(d) == 1:
        #         return f"Trait({tuple(d.values())[0]})"
        #     case list() as opts, dict() as hopts:
        #         return f"Trait({len(opts) + len(hopts)})"
        # return f"OptionMap({self.table})"

def flat_gen(*args):
    for arg in args:
        if isinstance(arg, str):
            yield arg
        else:
            try:
                for item in flat_gen(*arg):
                    yield item
            except TypeError:
                yield arg

class Table(Function):
    name = None
    # records: list[Record] | dict[Record, Record] | set[Record] | None
    # getters = dict[str, tuple[int, Field]]
    # setters = dict[str, tuple[int, Field]]
    # fields = list[Field]
    # noinspection PyDefaultArgument
    def __init__(self, *traits: Trait, name=None, fn_options={}, fn_fields=[]):
        self.traits = (Trait(), *traits)
        self.getters = {}
        self.setters = {}
        self.defaults = ()
        # for field in fields:
        #     self.fields.append(field)
        super().__init__(fn_options, *fn_fields, name=name, table_name='Table', traits=traits)
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


        # match fields:
        #     case list() | tuple():
        #         self.fields += list(field_tuple)
        #         for i, field in enumerate(fields):
        #             self.field_ids[field.name] = i + 1
        #     case dict():
        #         for name, monad in fields.items():
        #             self.field_ids[name] = len(self.fields)
        #             self.fields.append(Slot(name, monad))
        #     case None:
        #         pass
        #     case _:
        #         raise TypeError(f"Invalid argument type for fields: {type(fields)} {fields}")

    @property
    def trait(self):
        return self.traits[0]

    def integrate_traits(self):
        defaults: dict[str, Function | None] = {}
        types: dict[str, Pattern] = {}

        for trait in self.traits:
            for trait_field in trait.fields:
                name = trait_field.name
                pattern: Pattern = getattr(trait_field, 'type', None)
                if pattern:
                    assert pattern.binding == name
                    if name in types and not types[name].issubset(pattern):
                        raise SlotErr(f"Line {Context.line}: Could not integrate table {self.name}; "
                                      f"Field {name}'s type ({types[name]}) "
                                      f"doesn't match {trait_field.__class__.__name__} {name} of {trait.name}.")
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

        self.defaults = tuple(defaults[n] for n in defaults)
        patt = ArgsMatcher(*(Parameter(types[name],
                                       name,
                                       "?" * (defaults[name] is not None))
                                for name in defaults))

        self.assign_option(patt, Native(lambda args: BuiltIns['new'](Args(Context.env.caller) + args)), no_clobber=True)

        self.catalog = OptionCatalog({}, *self.traits)

    # def get_field(self, name: str):
    #     _, field = self.getters.get(name,
    #                                 self.setters.get(name, (0, None)))
    #     return field

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        self.records[key] = value

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
                self.records[record.key] = record
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
        self.name = 'MetaTable'
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


class Field(Record):
    type = None
    def __init__(self, name: str, type=None, default=None, formula=None):
        self.name = name
        if type:
            self.type = type
        if default is None:
            default = py_value(None)
        if formula is None:
            formula = py_value(None)
        super().__init__(BuiltIns['Field'])
        # , name=py_value(name),
        # type=ArgsMatcher(Parameter(type)) if type else BuiltIns['blank'],
        # is_formula=py_value(formula is not None),
        # default=default, formula=formula)


class Slot(Field):
    def __init__(self, name, type, default=None):
        match default:
            case Function(op_list=[Option(pattern=
                                          ArgsMatcher(parameters=(Parameter(pattern=Pattern(binding='self')), ))
                                          )]):
                pass  # assert that default is a function whose sole option is [<patt> self]: ...
            case _:
                assert default is None
        self.default = default
        super().__init__(name, type, default)

    # def get_data(self, rec, idx):
    #     return rec.data[idx]
    #
    # def set_data(self, rec, idx, value):
    #     rec.data[idx] = value
    #     return BuiltIns['blank']

    def __repr__(self):
        return f"Slot({self.name}: {self.type}{' ('+str(self.default)+')' if self.default else ''})"


class Formula(Field):
    def __init__(self, name, type, formula):
        self.formula = formula
        super().__init__(name, type, None, formula)

    # def get_data(self, rec, idx):
    #     return self.formula.call(rec)

    def __repr__(self):
        return f"Formula({self.name}: {str(self.formula)})"

class Setter(Field):
    fn: Function
    def __init__(self, name: str, fn: Function):
        self.fn = fn
        super().__init__(name)

    # def set_data(self, rec, idx, value):
    #     return self.fn.call(rec, value)

    def __repr__(self):
        return f"Setter({self.name}: {self.fn})"

class Matcher:
    guard = None
    invert = False
    def match_score(self, arg: Record) -> bool | float:
        return self.basic_score(arg)
        # score = self.basic_score(arg)
        # if self.invert:
        #     score = not score
        # if score and self.guard:
        #     result = self.guard.call(arg)
        #     return score * BuiltIns['bool'].call(result).value
        # return score

    def basic_score(self, arg):
        # implemented by subclasses
        raise NotImplementedError

    def issubset(self, other):
        print('WARNING: Matcher.issubset method not implemented properly yet.')
        return self.equivalent(other)

    def equivalent(self, other):
        return True
        # return (other.guard is None or self.guard == other.guard) and self.invert == other.invert

    # def call_guard(self, arg: Record) -> bool:
    #     if self.guard:
    #         result = self.guard.call(arg)
    #         return BuiltIns['bool'].call(result).value
    #     return True

    def get_rank(self):
        return self.rank
        # rank = self.rank
        # if self.invert:
        #     rank = tuple(100 - n for n in rank)
        # if self.guard:
        #     rank = (rank[0], rank[1] - 1, *rank[1:])
        # return rank

    def __lt__(self, other):
        return self.get_rank() < other.get_rank()

    def __le__(self, other):
        return self.get_rank() <= other.get_rank()

    # def __eq__(self, other):
    #     return self.get_rank() == other.get_rank()

class TableMatcher(Matcher):
    table: Table
    rank = 5, 0

    def __init__(self, table, name=None, guard=None, inverse=False):
        assert isinstance(table, Table)
        self.table = table
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool:
        return arg.table == self.table or self.table in arg.table.traits

    def issubset(self, other):
        match other:
            case TableMatcher(table=table):
                return table == self.table
            case TraitMatcher(trait=trait):
                return trait in self.table.traits
        return False

    def equivalent(self, other):
        return isinstance(other, TableMatcher) and self.table == other.table

    def __repr__(self):
        return f"TableMatcher({self.table})"

class TraitMatcher(Matcher):
    trait: Trait
    rank = 6, 0

    def __init__(self, trait):
        self.trait = trait

    def basic_score(self, arg: Record) -> bool:
        return self.trait in arg.table.traits

    def issubset(self, other):
        return isinstance(other, TraitMatcher) and other.trait == self.trait

    def equivalent(self, other):
        return isinstance(other, TraitMatcher) and other.trait == self.trait

    def __repr__(self):
        return f"TraitMatcher({self.trait})"


class ValueMatcher(Matcher):
    value: Record
    rank = 1, 0

    def __init__(self, value):
        self.value = value

    def basic_score(self, arg: Record) -> bool:
        return arg == self.value

    def issubset(self, other):
        match other:
            case ValueMatcher(value=value):
                return value == self.value
            case TableMatcher(table=table):
                return self.value.table == table
            case TraitMatcher(trait=trait):
                return trait in self.value.table.traits
        return False

    def equivalent(self, other):
        return isinstance(other, ValueMatcher) and other.value == self.value

    def __repr__(self):
        return f"ValueMatcher({self.value})"


class FunctionMatcher(Matcher):
    # pattern: ArgsMatcher
    # return_type: Matcher
    def __init__(self, pattern, return_type, name=None, guard=None, inverse=False):
        self.pattern = pattern
        self.return_type = return_type
        super().__init__(name, guard, inverse)

    def basic_score(self, arg):
        if not hasattr(arg, 'op_list'):
            return False
        arg: Function

        def options():
            yield from arg.op_list
            yield from arg.op_map.values()

        if all((option.pattern.issubset(self.pattern) and option.return_type.issubset(self.return_type)
                for option in options())):
            return True

    def issubset(self, other):
        match other:
            case FunctionMatcher(pattern=patt, return_type=ret):
                return self.pattern.issubset(patt) and self.return_type.issubset(ret)
            case TraitMatcher(trait=BuiltIns.get('fn')) | TableMatcher(table=BuiltIns.get('Function')):
                return True
        return False

    def equivalent(self, other):
        return (isinstance(other, FunctionMatcher)
                and other.pattern == self.pattern
                and other.return_type == self.return_type)


class AnyMatcher(Matcher):
    rank = 100, 0
    def basic_score(self, arg: Record) -> True:
        return True

    def issubset(self, other):
        return isinstance(other, AnyMatcher)

    def equivalent(self, other):
        return isinstance(other, AnyMatcher)

    def __repr__(self):
        return f"AnyMatcher()"

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
            case _:
                return False

    def issubset(self, other):
        return isinstance(other, EmptyMatcher)

    def equivalent(self, other):
        return isinstance(other, EmptyMatcher)

    def __repr__(self):
        return f"EmptyMatcher()"


class ExprMatcher(Matcher):
    def __init__(self, expr):
        self.expression = expr
    def basic_score(self, arg):
        print(f"Line {Context.line}: WARNING: expr pattern not fully implemented yet.")
        return self.expression.evaluate().truthy


class IterMatcher(Matcher):
    parameters: tuple
    def __init__(self, *params):
        self.parameters = params

    def basic_score(self, arg: Record):
        return self.match_zip(arg)[0]

    def match_zip(self, arg: Record):
        try:
            it = iter(arg)  # noqa
        except TypeError:
            return 0, {}
        state = MatchState(self.parameters, list(it))
        return state.match_zip()


class FieldMatcher(Matcher):
    fields: dict

    def __init__(self, **fields):
        self.fields = fields

    def basic_score(self, arg: Record):
        for name, param in self.fields.items():
            prop = arg.get(name, None)
            if prop is None:
                if not param.optional:
                    return False
                continue
            if not param.pattern.match_score(prop):
                return False
        return True

    def match_zip(self, arg: Record):
        raise NotImplementedError
        # state = MatchState((), Args(**dict(((name, arg.get(name)) for name in self.fields))))
        # return state.match_zip()


class ArgsMatcher(Matcher):
    parameters: tuple
    named_params: frozendict
    def __init__(self, *parameters, **named_params):
        self.parameters = parameters
        self.named_params = frozendict(named_params)
        super().__init__(BuiltIns['ArgsMatcher'])  # , parameters=py_value(parameters))

    def match_score(self, *values: Record) -> int | float:
        return self.match_zip(values)[0]

    def issubset(self, other):
        return (isinstance(other, ArgsMatcher)
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
            match param.pattern.matchers:
                case (ValueMatcher(value=value), ) if value.hashable():
                    if isinstance(id, int):
                        pos_args.append(value)
                    else:
                        names[id] = value
                case _:
                    return None
        # for param in self.parameters:
        #     match param.pattern.matchers:
        #         case (ValueMatcher(value=value),) if value.hashable():
        #             pos_args.append(value)
        #         case _:
        #             return None
        # for name, param in self.named_params.items():
        #     match param.pattern.matchers:
        #         case (ValueMatcher(value=value),) if value.hashable():
        #             names[name] = value
        #         case _:
        #             return None
        return Args(*pos_args, **names)

    def __iter__(self):
        yield from enumerate(self.parameters)
        yield from self.named_params.items()

    @memoize
    def min_len(self) -> int:
        count = 0
        for param in self.parameters:
            count += not param.optional
        return count

    @memoize
    def max_len(self) -> int | float:
        for param in self.parameters:
            if param.quantifier in ("+", "*"):
                return math.inf
        return len(self.parameters)

    def match_zip(self, args=None) -> tuple[float | int, dict[str, Record]]:
        if args is None:
            return 1, {}
        if len(args) == 0 == self.min_len():
            return 1, {}
        if not self.min_len() <= len(args) <= self.max_len():
            return 0, {}
        if isinstance(args, tuple):
            args = Args(*args)
        return MatchState(self.parameters, args).match_zip()
        # state = MatchState(self.parameters, args)
        # return self.match_zip_recursive(state)

    @memoize
    def min_len(self) -> int:
        count = 0
        for param in self.parameters:
            count += not param.optional
        return count

    @memoize
    def max_len(self) -> int | float:
        for param in self.parameters:
            if param.quantifier in ("+", "*"):
                return math.inf
        return len(self.parameters)

    def __lt__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters < other.parameters

    def __le__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters <= other.parameters

    def __eq__(self, other):
        return (isinstance(other, ArgsMatcher)
                and self.parameters == other.parameters
                and self.named_params == other.named_params)

    def __hash__(self):
        return hash((self.parameters, self.named_params))

    def __gt__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters > other.parameters

    def __ge__(self, other):
        if not isinstance(other, ArgsMatcher):
            return NotImplemented
        return self.parameters >= other.parameters

    def __repr__(self):
        return f"ArgsMatcher({', '.join(self.parameters)}{'; ' + str(self.named_params) if self.named_params else ''})"

class Pattern(Record):
    binding: str | None = None
    def __init__(self, binding=None):
        if binding is not None:
            self.binding = binding
        super().__init__(BuiltIns['Pattern'])

class Intersection(Pattern):
    matchers: tuple[Matcher, ...]  # acts as an intersection of matchers
    binding: str | None = None

    def __init__(self, *matchers: Matcher, binding=None):
        self.matchers = matchers
        super().__init__(binding)

    def match_score(self, arg: Record):
        return all(m.match_score(arg) for m in self.matchers)

    def issubset(self, other):
        match other:
            case Matcher() as other_matcher:
                return any(m.issubset(other_matcher) for m in self.matchers)
            case Intersection() as patt:
                return any(matcher.issubset(patt) for matcher in self.matchers)
            case Union(patterns=patterns):
                return any(self.issubset(patt) for patt in patterns)
            case Parameter(pattern=pattern):
                return self.issubset(pattern)
        return False

    def __lt__(self, other):
        match other:
            case Intersection(matchers=other_matchers):
                return (len(self.matchers) > len(other_matchers)
                        or len(self.matchers) == len(other_matchers) and self.matchers < other_matchers)
            case Union(patterns=patterns):
                return any(self <= p for p in patterns)
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case Intersection(matchers=other_matchers):
                return (len(self.matchers) > len(other_matchers)
                        or len(self.matchers) == len(other_matchers) and self.matchers <= other_matchers)
            case Union(patterns=patterns):
                return any(self <= p for p in patterns)
            case _:
                raise NotImplementedError

    def __eq__(self, other):
        match other:
            case Intersection(matchers=matchers):
                return matchers == self.matchers
            case Union(patterns=(Pattern() as patt, )):
                return self == patt
            case Matcher() as m:
                return len(self.matchers) == 1 and self.matchers[0] == m
        return False


class Union(Pattern):
    patterns: tuple[Pattern, ...]
    def __init__(self, *patterns, binding=None):
        self.patterns = patterns
        super().__init__(binding)

    def match_score(self, arg: Record):
        return any(p.match_score(arg) for p in self.patterns)

    def issubset(self, other):
        return all(p.issubset(other) for p in self.patterns)

    def __lt__(self, other):
        match other:
            case Intersection():
                return all(p < other for p in self.patterns)
            case Union(patterns=patterns):
                return self.patterns < patterns
            case _:
                raise NotImplementedError

    def __le__(self, other):
        match other:
            case Intersection():
                return all(p <= other for p in self.patterns)
            case Union(patterns=patterns):
                return self.patterns <= patterns
            case _:
                raise NotImplementedError

    def __eq__(self, other):
        match self.patterns:
            case ():
                return isinstance(other, Union) and other.patterns == ()
            case (Pattern() as patt, ):
                return patt == other
        return isinstance(other, Union) and self.patterns == other.patterns


class Parameter:
    pattern: Pattern | None = None
    quantifier: str  # "+" | "*" | "?" | "!" | ""
    count: tuple[int, int | float]
    optional: bool
    required: bool
    multi: bool
    default = None

    @property
    def binding(self): return self.pattern.binding

    def __init__(self, pattern, quantifier="", default=None):
        self.pattern = patternize(pattern)
        # self.name = self.pattern.binding
        if default:
            if isinstance(default, Option):
                self.default = default
            else:
                self.default = Option(ArgsMatcher(), default)
            match quantifier:
                case "":
                    quantifier = '?'
                case "+":
                    quantifier = "*"
        self.quantifier = quantifier

    def issubset(self, other):
        if not isinstance(other, Parameter):
            raise NotImplementedError(f"Not yet implemented Parameter.issubset({other.__class__})")
        if self.count[1] > other.count[1] or self.count[0] < other.count[0]:
            return False
        return self.pattern.issubset(other.pattern)

    def _get_quantifier(self) -> str:
        return self._quantifier
    def _set_quantifier(self, quantifier: str):
        self._quantifier = quantifier
        match quantifier:
            case "":
                self.count = (1, 1)
            case "?":
                self.count = (0, 1)
            case "+":
                self.count = (1, math.inf)
            case "*":
                self.count = (0, math.inf)
            case "!":
                self.count = (1, 1)
                # union matcher with `nonempty` pattern
        self.optional = quantifier in ("?", "*")
        self.required = quantifier in ("", "+")
        self.multi = quantifier in ("+", "*")
    quantifier = property(_get_quantifier, _set_quantifier)

    def match_score(self, value) -> int | float: ...

    def compare_quantifier(self, other):
        return "_?+*".find(self.quantifier) - "_?+*".find(other.quantifier)

    def __lt__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern < other.pattern
        return NotImplemented

    def __le__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern <= other.pattern
        return NotImplemented

    def __eq__(self, other):
        match other:
            case Parameter() as param:
                pass
            case Matcher() | Pattern():
                param = Parameter(other)
            case ArgsMatcher(parameters=(param, ), named_params={}):
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
        return NotImplemented

    def __ge__(self, other):
        match other:
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.pattern >= other.pattern
        return NotImplemented

    def __repr__(self):
        return f"Parameter({self.pattern} {self.binding or ''}{self.quantifier})"


# class OLDPattern(Record):
#     """
#     a sequence of zero or more parameters, together with their quantifiers
#     """
#     def __init__(self, *parameters, **named_params):
#         self.parameters = parameters
#         self.named_params = named_params
#         super().__init__(BuiltIns['ArgsMatcher'])  # , parameters=py_value(parameters))
#
#     def truthy(self):
#         return bool(self.parameters)
#
#     def match_score(self, *values: Record) -> int | float:
#         return self.match_zip(values)[0]
#
#     def issubset(self, other):
#         return all(p1.issubset(p2) for (p1, p2) in zip(self.parameters, other.parameters))
#
#     def __len__(self):
#         return len(self.parameters)
#
#     def __getitem__(self, item):
#         return self.parameters[item]
#
#     def to_tuple(self):
#         key: list[Record] = []
#         for parameter in self.parameters:
#             t = parameter.matcher
#             if isinstance(t, ValueMatcher) and t.guard is None and not t.invert and t.value.hashable():
#                 key.append(t.value)
#             else:
#                 return None
#         return tuple(key)
#
#     def to_args(self):
#         pos_args = []
#         names = {}
#         for param in self.parameters:
#             t = param.matcher
#             if isinstance(t, ValueMatcher) and t.guard is None and not t.invert and t.value.hashable():
#                 pos_args.append(t.value)
#             else:
#                 return None
#         for name, param in self.named_params.items():
#             t = param.matcher
#             if isinstance(t, ValueMatcher) and t.guard is None and not t.invert and t.value.hashable():
#                 names[name] = t.value
#             else:
#                 return None
#         return Args(*pos_args, **names)
#
#     @memoize
#     def min_len(self) -> int:
#         count = 0
#         for param in self.parameters:
#             count += not param.optional
#         return count
#
#     @memoize
#     def max_len(self) -> int | float:
#         for param in self.parameters:
#             if param.quantifier in ("+", "*"):
#                 return math.inf
#         return len(self.parameters)
#
#     def match_zip(self, args=None) -> tuple[float|int, dict[str, Record]]:
#         if args is None:
#             return 1, {}
#         if len(args) == 0 == self.min_len():
#             return 1, {}
#         if not self.min_len() <= len(args) <= self.max_len():
#             return 0, {}
#         if isinstance(args, tuple):
#             args = Args(*args)
#         return MatchState(self.parameters, args).match_zip()
#         # state = MatchState(self.parameters, args)
#         # return self.match_zip_recursive(state)
#
#     def __lt__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters < other.parameters
#
#     def __le__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters <= other.parameters
#
#     def __eq__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return False
#         return self.parameters == other.parameters
#         # if isinstance(other, ArgsMatcher):
#         #     for p1, p2 in zip(self.parameters, other.parameters):
#         #         if p1 != p2:
#         #             return False
#         #     return True
#         # if len(self.parameters) != 1:
#         #     return False
#         # param1 = self.parameters[0]
#         # if isinstance(other, Matcher):
#         #     param2 = Parameter(other)
#         # else:
#         #     param2 = other
#         # return param1 == param2
#
#     def __hash__(self):
#         return hash(self.parameters)
#
#     def __gt__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters > other.parameters
#
#     def __ge__(self, other):
#         if not isinstance(other, ArgsMatcher):
#             return NotImplemented
#         return self.parameters >= other.parameters
#
#     def __repr__(self):
#         return f"ArgsMatcher{self.parameters}"
#     # def __len__(self): ...
#     # def __getitem__(self, item): ...
#     # # def __eq__(self, other): ...
#     # def __hash__(self): ...

class MatchState:
    def __init__(self, parameters, args, i_param=0, i_arg=0, score=0, param_score=0, bindings=None):
        if isinstance(parameters, ArgsMatcher):
            self.parameters = parameters
        else:
            self.parameters = ArgsMatcher(*parameters)
        self.args = args
        self.i_param = i_param
        self.i_arg = i_arg
        self.score = score
        self.param_score = param_score
        self.bindings = bindings or {}
        # self.done = i_param == len(parameters) and i_arg == len(args)
        if i_param > len(parameters) or i_arg > len(args):
            raise RuntimeErr(f"Line {Context.line}: ArgsMatcher error: i_param or i_arg went out of bounds.")

    # @property
    # def done(self):
    #     if self.i_param > len(self.parameters) or self.i_arg > len(self.args):
    #         raise RuntimeErr(f"Line {Context.line}: ArgsMatcher error: i_param or i_arg went out of bounds.")
    #     return self.i_param == len(self.parameters) and self.i_arg == len(self.args)

    @property
    def success(self):
        if self.i_param > len(self.parameters) or self.i_arg > len(self.args):
            raise RuntimeErr(f"Line {Context.line}: ArgsMatcher error: i_param or i_arg went out of bounds.")
        if isinstance(self.args, Args):
            if self.i_arg < len(self.args.positional_arguments):
                return False
            else:
                return self.i_param == len(self.parameters) and self.score
        return self.i_param == len(self.parameters) and self.i_arg == len(self.args) and self.score
        # return self.done and self.score

    @property
    def param(self):
        if self.i_param < len(self.parameters):
            return self.parameters[self.i_param]

    @property
    def arg(self):
        if self.i_arg < len(self.args.positional_arguments):
            return self.args[self.i_arg]

    def branch(self, **kwargs):
        bindings = self.bindings.copy().update(kwargs.get('binding', {}))
        for key in self.__dict__:
            if key not in kwargs:
                kwargs[key] = self.__dict__[key]
        return MatchState(**kwargs, bindings=bindings)

    def match_zip_OLD(self):
        while param := self.param:
            name = param.pattern.binding
            if name in self.args.named_arguments and name not in self.bindings:
                branch = self.branch()
                branch.bindings[name] = self.args.named_arguments[name]
                branch.i_param += 1
                score, bindings = branch.match_zip()
                if score:
                    return score, bindings
            if name in self.args.flags and name not in self.bindings:
                self.bindings[name] = BuiltIns['true']
                self.i_param += 1
                continue
            if isinstance(param, UnionParam):
                for param in param.parameters:
                    new_params = [*self.parameters[:self.i_param], param, *self.parameters[self.i_param + 1:]]
                    score, bindings = self.branch(parameters=new_params).match_zip()
                    if score:
                        return score, bindings
                return 0, {}

            key: str | int = param.name or self.i_param
            self.param_score *= param.multi
            if self.arg is not None:
                match_value = param.matcher.match_score(self.arg)
            else:
                match_value = 0  # no arguments left to process match
            if not match_value and not param.optional and not self.bindings.get(key):
                return 0, {}
            match param.quantifier:
                case "":
                    # match patt, save, and move on
                    if not match_value:
                        return 0, {}
                    self.bindings[key] = self.arg
                    self.score += match_value
                    self.i_arg += 1
                    self.i_param += 1
                case "?":
                    # try match patt and save... move on either way
                    self.i_param += 1
                    if match_value:
                        branch = self.branch()
                        branch.i_arg += 1
                        branch.score += match_value
                        branch.bindings[key] = self.arg
                        score, bindings = branch.match_zip()
                        if score:
                            return score, bindings
                    elif param.default:
                        self.bindings[key] = param.default.resolve()
                case "+" | "*":
                    if key not in self.bindings:
                        self.bindings[key] = []
                        if not match_value and param.default:
                            self.bindings[key] = param.default.resolve()
                    if match_value:
                        branch = self.branch()
                        branch.i_arg += 1
                        branch.param_score += match_value
                        branch.bindings[key].append(self.arg)
                        score, bindings = branch.match_zip()
                        if score:
                            return score, bindings
                    if self.param_score:  #  if len(saves[key].value):
                        self.score += self.param_score / len(self.bindings[key])
                    self.i_param += 1
        if self.success:
            return self.score_and_bindings()
        return 0, {}

    def match_zip(self):
        while param := self.param:
            name = param.pattern.binding
            if name in self.args.named_arguments and name not in self.bindings:
                branch = self.branch()
                branch.bindings[name] = self.args.named_arguments[name]
                branch.i_param += 1
                score, bindings = branch.match_zip()
                if score:
                    return score, bindings
            if name in self.args.flags and name not in self.bindings:
                self.bindings[name] = BuiltIns['true']
                self.i_param += 1
                continue
            if isinstance(param.pattern, Union):
                # TODO: I don't think this is logically sound.  I think I need to find a new approach.  I need to abstract more.
                param_tuple = self.parameters.parameters
                param_list = list(param_tuple)
                for patt in param.pattern.patterns:
                    param_list[self.i_param] = Parameter(patt)
                    self.parameters.parameters = tuple(param_list)
                    new_params = [*self.parameters[:self.i_param], param, *self.parameters[self.i_param + 1:]]
                    score, bindings = self.branch(parameters=new_params).match_zip()
                    if score:
                        self.parameters.parameters = param_tuple
                        return score, bindings
                self.parameters.parameters = param_tuple
                return 0, {}

            key: str | int = name or self.i_param
            if self.arg is not None:
                match_value = param.pattern.match_score(self.arg)
            else:
                match_value = 0  # no arguments left to process match

            # clear param score if you have moved on...
            # Wait, this is wrong.  I need to clear the param score only when I
            assert not (self.param_score and not param.multi)
            # self.param_score *= param.multi

            if param.required and match_value == 0 == self.param_score:
                return 0, self.bindings
            match param.quantifier:
                case "":
                    # match patt, save, and move on
                    if not match_value:
                        return 0, {}
                    self.bindings[key] = self.arg
                    self.score += match_value
                    self.i_arg += 1
                    self.i_param += 1
                case "?":
                    # try match patt and save... move on either way
                    self.i_param += 1
                    if match_value:
                        branch = self.branch()
                        branch.bindings[key] = self.arg
                        branch.score += match_value
                        branch.i_arg += 1
                        score, bindings = branch.match_zip()
                        if score:
                            return score, bindings
                    elif param.default:
                        self.bindings[key] = param.default.resolve()
                case "+" | "*":
                    if key not in self.bindings:
                        self.bindings[key] = []
                        if not match_value and param.default:
                            self.bindings[key] = param.default.resolve()
                    if match_value:
                        branch = self.branch()
                        branch.i_arg += 1
                        branch.param_score += match_value
                        branch.bindings[key].append(self.arg)
                        score, bindings = branch.match_zip()
                        if score:
                            return score, bindings
                    if self.param_score:  #  if len(saves[key].value):
                        self.score += self.param_score / len(self.bindings[key])
                    self.i_param += 1
                    self.param_score = 0
        if self.success:
            return self.score_and_bindings()
        return 0, {}

    def score_and_bindings(self):
        for key, value in self.bindings.items():
            if isinstance(value, list):
                self.bindings[key] = py_value(tuple(value))
        self.score /= len(self.parameters)
        return self.score, self.bindings





def patternize(val):
    match val:
        case Pattern():
            return val
        case Table():
            return Pattern(TableMatcher(val))
        case Trait():
            return Pattern(TraitMatcher(val))
        case Record():
            return Pattern(ValueMatcher(val))
        case Matcher() as matcher:
            return Pattern(matcher)
        case _:
            raise TypeErr(f"Line {Context.line}: Could not patternize {val}")


class Args(Record):
    positional_arguments: list[Record] | tuple[Record, ...]
    named_arguments: dict[str, Record]
    flags: set[str]
    def __init__(self, *args: Record, flags: set[str] = None, **kwargs: Record):
        self.positional_arguments = args
        self.flags = flags or set()
        self.named_arguments = kwargs
        super().__init__(BuiltIns['Args'])

    def __len__(self):
        return len(self.positional_arguments) + len(self.named_arguments) + len(self.flags)

    def __getitem__(self, key):
        if key in self.flags:
            return BuiltIns['true']
        return self.named_arguments.get(key, self.positional_arguments[key])

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

    def __repr__(self):
        pos = map(str, self.positional_arguments)
        names = (f"{k}={v}" for (k, v) in self.named_arguments.items())
        flags = ('!'+str(f) for f in self.flags)
        return f"Args({', '.join(pos)}; {', '.join(names)}; {' '.join(flags)})"


# class Alias:
#     function
#     closure
#     name

class CodeBlock:
    exprs: list
    scope = None
    def __init__(self, block):
        self.exprs = list(map(Context.make_expr, block.nodes))
        self.scope = Context.env
    def execute(self, args=None, caller=None, bindings=None, *, fn=None):
        if args is not None or fn is not None:
            closure = Closure(self, args, caller, bindings, fn)
            Context.push(Context.line, closure)

            def finish():
                Context.pop()
                return closure.return_value or caller or fn
        else:
            def finish():  # noqa
                return BuiltIns['blank']
        line = Context.line
        for expr in self.exprs:
            Context.line = expr.line
            expr.evaluate()
            if Context.env.return_value:
                break
            if Context.break_loop or Context.continue_:
                break
        Context.line = line
        return finish()

    def __repr__(self):
        return f"CodeBlock({len(self.exprs)})"

class Native(CodeBlock):
    def __init__(self, fn: PyFunction):
        self.fn = fn
        self.scope = Context.env

    def execute(self, args=None, caller=None, bindings=None, *, fn=None):
        closure = Closure(self, args, caller, bindings, fn)
        Context.push(Context.line, closure)
        line = Context.line
        if isinstance(args, tuple):
            closure.return_value = self.fn(*args)
        else:
            closure.return_value = self.fn(args)
        Context.line = line
        Context.pop()
        return closure.return_value

    def __repr__(self):
        return f"Native({self.fn})"

class Closure:
    return_value = None
    def __init__(self, code_block, args=None, caller=None, bindings=None, fn=None):
        # self.names = bindings or {}
        self.vars = bindings or {}
        self.locals = {}
        self.code_block = code_block
        self.scope = code_block.scope
        self.args = args
        self.caller = caller
        self.fn = fn

    def assign(self, name: str, value: Record):
        scope = self
        while scope:
            if name in scope.vars:
                scope.vars[name] = value
                return value
            scope = scope.scope
        self.locals[name] = value
        return value

    def __repr__(self):
        return (f"Closure({len(self.vars) + len(self.locals)} names; " 
                f"{'running' if self.return_value is None else 'finished: ' + str(self.return_value)})")

class TopNamespace(Closure):
    code_block = None
    scope = None
    args = None
    caller = None
    def __init__(self, bindings: dict[str, Record]):
        self.vars = {}
        self.locals = bindings

class Option(Record):
    value = None
    block = None
    fn = None
    alias = None
    dot_option = False
    return_type = AnyMatcher()
    def __init__(self, pattern, resolution=None):
        match pattern:
            case ArgsMatcher():
                self.pattern = pattern
            case Parameter() as param:
                self.pattern = ArgsMatcher(param)
            case _:
                self.pattern = ArgsMatcher(Parameter(patternize(pattern)))
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
            case CodeBlock(): self.block = resolution
            # case PyFunction(): self.block = Native(resolution)
            case PyFunction(): self.fn = resolution
            case Option(): self.alias = resolution
            case Record():
                self.value = resolution
                self.return_type = ValueMatcher(resolution)
            case _:
                raise ValueError(f"Line {Context.line}: Could not assign resolution {resolution} to option {self}")
    def get_resolution(self):
        if self.value is not None:
            return self.value
        return self.block or self.fn or self.alias

    resolution = property(get_resolution, set_resolution, nullify)

    def resolve(self, args=None, caller=None, bindings=None):
        if self.alias:
            return self.alias.resolve(args, caller, bindings)
        if self.value is not None:
            return self.value
        if self.fn:
            if isinstance(args, Args):
                return call(self.fn, args)
            return self.fn(*args)
        if self.dot_option:
            caller = args[0]
        if self.block is None:
            raise NoMatchingOptionError(f"Line {Context.line}: Could not resolve null option")
        return self.block.execute(args, caller, bindings)

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


class Operator:
    def __init__(self, text, fn=None,
                 prefix=None, postfix=None, binop=None, ternary=None,
                 associativity='left',
                 chainable=False,
                 static=False):
        Op[text] = self
        self.text = text
        # self.precedence = precedence
        if fn:
            if not fn.name:
                fn.name = text
            BuiltIns[text] = fn
        self.fn = fn
        self.associativity = associativity  # 'right' if 'right' in flags else 'left'
        self.prefix = prefix  # 'prefix' in flags
        self.postfix = postfix  # 'postfix' in flags
        self.binop = binop  # 'binop' in flags
        self.ternary = ternary
        self.static = static  # 'static' in flags
        self.chainable = chainable

        assert self.binop or self.prefix or self.postfix or self.ternary

    def eval_args(self, lhs, rhs) -> list[Record]:
        raise NotImplementedError('Operator.prepare_args not implemented')

    def __repr__(self):
        return self.text
