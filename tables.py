import math
from fractions import Fraction
from Env import *
from typing import TypeVar, Generic

PyFunction = type(lambda: None)


# class FunctionRecordPrototype:
#     return_value = None
#     name: str = None
#     # mro = ()
#     env = None
#     caller = None  # self
#     block = None
#     return_type = None
#     is_null = True
#     options = None
#     hashed_options = None
#     def __init__(self):
#         # self.mro = (self, )
#         self.env = Context.env
#         if self.options is None:
#             self.options = []
#         if self.hashed_options is None:
#             self.hashed_options = {}
#
#     def add_option(self, pattern, resolution=None):
#         option = Option(pattern, resolution)
#
#         # try to hash option
#         key: list[Record] = []
#         for parameter in option.pattern.parameters:
#             t = parameter.matcher
#             if isinstance(t, ValueMatcher) and t.guard is None and not t.invert and t.value.hashable():
#                 key.append(t.value)
#             else:
#                 self.options.insert(0, option)
#                 break
#         else:
#             self.hashed_options[tuple(key)] = option
#
#         return option
#
#     def remove_option(self, pattern):
#         opt = self.select_by_pattern(pattern)
#         if opt is None:
#             raise NoMatchingOptionError(f'cannot find option "{pattern}" to remove')
#         opt.nullify()
#
#     def assign_option(self, pattern, resolution=None):
#         opt = self.select_by_pattern(pattern)
#         if opt is None:
#             return self.add_option(pattern, resolution)
#         opt.nullify()
#         opt.assign(resolution)
#         return opt
#
#     @property
#     def mro(self):
#         self: Record
#         return self, *(s.prototype for s in self.slices), self.table.prototype
#
#     def select_and_bind(self, key: list, walk_prototype_chain=True, ascend_env=False):
#         try:
#             return self.hashed_options[tuple(key)], {}
#         except (TypeError, KeyError):
#             pass
#         option = bindings = high_score = 0
#         for opt in self.options:
#             score, saves = opt.pattern.match_zip(key)
#             if score == 1:
#                 return opt, saves
#             if score > high_score:
#                 high_score = score
#                 option, bindings = opt, saves
#         if option:
#             return option, bindings
#         if walk_prototype_chain and self.mro:
#             for t in self.mro[1:]:
#                 try:
#                     return t.select_and_bind(key, False, ascend_env)
#                 except NoMatchingOptionError:
#                     pass
#         if ascend_env and self.env:
#             try:
#                 return self.env.select_and_bind(key, walk_prototype_chain, True)
#             except NoMatchingOptionError:
#                 pass
#         raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in {self.name or self}")
#         # -> tuple[Option, dict[str, Function]]:
#
#     def select_by_pattern(self, patt=None, default=None, ascend_env=False):
#         # return [*[opt for opt in self.options if opt.pattern == patt], None][0]
#         for opt in self.options:
#             if opt.pattern == patt:
#                 return opt
#         if ascend_env and self.env:
#             return self.env.select_by_pattern(patt, default)
#         return default
#
#     def select_by_name(self, name: str, ascend_env=True):
#         return self.select_by_value((py_value(name),), ascend_env)
#
#     def select_by_value(self, values, ascend_env=True):
#         # try self first
#         for fn in self.mro:
#             try:
#                 return fn.hashed_options[values]
#             except KeyError:
#                 continue
#
#         if ascend_env and self.env:
#             try:
#                 return self.env.select_by_value(values, True)
#             except NoMatchingOptionError:
#                 pass
#         raise NoMatchingOptionError(f"Line {Context.line}: {values} not found in {self}")
#
#     def call(self, *key, ascend=False):
#         if key and isinstance(key[0], list | tuple):
#             pass
#         try:
#             option, bindings = self.select_and_bind(key, ascend_env=ascend)
#         except NoMatchingOptionError as e:
#             # if ascend and self.env:
#             #     try:
#             #         return self.env.call(*key, ascend=True)
#             #     except NoMatchingOptionError:
#             #         pass
#             raise e
#         return option.resolve(key, self, bindings)
#
#     def deref(self, name: str, ascend_env=True):
#         option = self.select_by_value((py_value(name),), ascend_env)
#         return option.resolve((), self)
#
#     def instanceof(self, type):
#         self: Record
#         return type == self.table or type in self.slices
#         # types = self.mro[1:]
#         # if prototype in types:
#         #     return 1
#         # for t in types:
#         #     k = t.instanceof(prototype)
#         #     if k:
#         #         return k / 2
#         # return 0
#         # # return len(self.type) and int(prototype in self.type) or self.type.instanceof(prototype)/2

class OptionMap:
    """ essentially just container to hold the options of tables, a proxy for methods """
    def __init__(self, options=None):
        self.options = []
        self.hashed_options = {}
        if options:
            for patt, val in options.items():
                self.add_option(patt, val)

    def add_option(self, pattern, resolution=None):
        option = Option(pattern, resolution)

        # try to hash option
        key: list[Record] = []
        for parameter in option.pattern.parameters:
            t = parameter.matcher
            if isinstance(t, ValueMatcher) and t.guard is None and not t.invert and t.value.hashable():
                key.append(t.value)
            else:
                if Context.settings['sort_options']:
                    for i, opt in enumerate(self.options):
                        if option.pattern <= opt.pattern:
                            self.options.insert(i, option)
                            break
                        elif option.pattern == opt.pattern:
                            self.options[i] = option
                            break
                    else:
                        self.options.append(option)
                else:
                    self.options.append(option)
                break
        else:
            self.hashed_options[tuple(key)] = option

        return option

    def remove_option(self, pattern):
        opt = self.select_by_pattern(pattern)
        if opt is None:
            raise NoMatchingOptionError(f'cannot find option "{pattern}" to remove')
        opt.nullify()

    def assign_option(self, pattern, resolution=None):
        opt = self.select_by_pattern(pattern)
        if opt is None:
            return self.add_option(pattern, resolution)
        opt.nullify()
        opt.assign(resolution)
        return opt

    def select_and_bind(self, key: tuple):
        if key in self.hashed_options:
            return self.hashed_options[key], {}

        option = bindings = None
        high_score = 0
        for opt in self.options:
            score, saves = opt.pattern.match_zip(key)
            if score == 1:
                return opt, saves
            if score > high_score:
                high_score = score
                option, bindings = opt, saves
        return option, bindings

    def select_by_pattern(self, patt, default=None):
        # return [*[opt for opt in self.options if opt.pattern == patt], None][0]
        for opt in self.options:
            if opt.pattern == patt:
                return opt
        return default

    def __repr__(self):
        match self.options, self.hashed_options:
            case [], {}:
                return "OptionMap()"
            case [opt], {}:
                return f"OptionMap({opt})"
            case [], dict() as d if len(d) == 1:
                return f"OptionMap({tuple(d.values())[0]})"
            case list() as opts, dict() as hopts:
                return f"OptionMap({len(opts) + len(hopts)})"
        # return f"OptionMap({self.table})"


class Record:
    # table: Table
    _filters: set | None = None
    # filters: set[FilterSlice]
    # slices: list[Slice]
    # data: dict[int, Record]
    # key: Record
    truthy = True
    def __init__(self, table, *slices, **data):
        super().__init__()
        self.table = table
        self.slices = list(slices)
        self.data = [py_value(None)] * len(self.table.fields)  # if len(self.table.fields) else []
        for i, field in enumerate(self.table.fields):
            match field:
                case Slot():
                    if field.name in data:
                        self.data[i] = data[field.name]
                    elif field.default is not None:
                        self.data[i] = field.default.call(self)
                    else:
                        self.data[i] = py_value(None)
                case Formula():
                    pass  # self.data[i] = py_value(None)
        table.add_record(self)

    @property
    def mro(self):
        return (self.table.prototype, )
    # key property
    def get_key(self):
        match self.table:
            case ListTable():
                return py_value(self.index)
            case DictTable(key_field=fid):
                return self.get_by_index(fid)
            case SetTable():
                return self
        raise RuntimeErr
    def set_key(self, new_key):
        match self.table:
            case ListTable():
                raise RuntimeErr(f"Cannot set automatically assigned key (index)")
            case DictTable(key_field=fid):
                self.set_by_index(fid, new_key)
            case SetTable():
                raise RuntimeErr(f"Cannot set key of SetTable.")
            case _:
                raise RuntimeErr
    key = property(get_key, set_key)

    @property
    def filters(self):
        if self._filters is None:
            self.update_slices()
        return self._filters

    def update_slices(self, *slices):
        if self._filters is None or not slices:
            self._filters = set(filter(lambda s: s.filter.call(self).truthy,
                                             self.table.slices))
        else:
            for s in slices:
                if s.filter.call(self).truthy:
                    self._filters.add(s)

    def get(self, name: str, *default):
        try:
            index = self.table.field_ids[name]
        except KeyError:
            if not default:
                raise SlotErr(f"Line {Context.line}: no field found with name '{name}'.")
            return default[0]
        return self.get_by_index(index)

    def get_by_index(self, index: int):
        field = self.table.fields[index]
        match field:
            case Slot():
                return self.data[index]
            case Formula():
                raise NotImplementedError
            case _:
                raise TypeError(f"Invalid Field subtype: {type(field)}")

    def set(self, name: str, value):
        index = self.table.field_ids[name]
        return self.set_by_index(index, value)

    def set_by_index(self, index: int, value):
        field = self.table.fields[index]
        match field:
            case Slot():
                # first validate the type
                self.data[index] = value
            case Formula():
                raise RuntimeErr("Cannot set formula field.")
            case _:
                raise TypeError(f"Invalid Field subtype: {type(field)}")
        self._filter_slices = None

    def call(self, *key):
        for t in self.mro:
            option, bindings = t.select_and_bind(key)
            if option:
                return option.resolve(key, self, bindings)
        raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in {self.name or self}")

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

    def __eq__(self, other):
        if not isinstance(other, Record):
            return False
        if isinstance(self.table, VirtTable) or isinstance(other.table, VirtTable):
            return id(self) == id(other)
        return (self.table, self.key) == (other.table, other.key)

    def __hash__(self):
        return id(self)

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


T = TypeVar('T', None, bool, int, Fraction, float, str, tuple, frozenset)
A = TypeVar('A')
class PyValue(Record, Generic[T]):
    def __init__(self, table, value: T, *slices):
        self.value = value
        super().__init__(table, *slices, key=self)

    def to_string(self):
        if self.table in (BuiltIns['ratio'], BuiltIns['float']):
            return py_value(write_number(self.value, Context.settings['base']))
        # if self.instanceof(BuiltIns['list']):
        #     return py_value(f"[{', '.join(v.to_string().value for v in self.value)}]")
        if isinstance(self.value, frozenset):
            return py_value(str(set(self.value)))
        return py_value(str(self.value))

    def __index__(self) -> int:
        if not isinstance(self.value, int | bool):
            raise TypeErr(f"Line {Context.line}: Index must be integer, not {self.table}")
        if not self.value:
            # return None
            raise ValueError("Pili lists are one-based, not zero-based.")
        return self.value - (self.value > 0)

    @property
    def truthy(self):
        return bool(self.value)

    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        return isinstance(other, PyValue) and self.value == other.value or self.value == other
    def __repr__(self):
        match self.value:
            case frozenset() as f:
                return repr(set(f))
            case Fraction(numerator=n, denominator=d):
                return f"{n}/{d}"
            case v:
                return repr(v)
        # return f"Record({self.value})"

def py_value(value: T):
    slices = []
    if value is None:
        return BuiltIns['blank']
    t = type(value)
    match t.__name__:
        case 'bool':
            return BuiltIns[str(value).lower()]
        case 'Fraction' | 'int':
            table = BuiltIns['ratio']
            if value.denominator == 1:
                value = int(value)
                slices.append(BuiltIns['int'])
        case 'float' | 'str' as tn:
            table = BuiltIns[tn]
            return PyValue[t](table, value)
        case 'tuple' | 'frozenset' as tn:
            table = BuiltIns[tn.title()]
            value = t(piliize(v) for v in value)
        case unknown_type:
            raise TypeErr(f"Unhandled python type for PyValue: {unknown_type} {value}")
    if isinstance(table, SetTable):
        return PyValue[t](table, value, *slices)
    return table[value] or PyValue[type(value)](table, value, *slices)

class PyObj(Record, Generic[A]):
    def __init__(self, obj):
        self.obj = obj
        super().__init__(BuiltIns['PyObj'])


class List(Record):
    records: list
    _type = None
    def __init__(self, initial_list: list = None, type=None):
        if initial_list is not None and not isinstance(initial_list, list):
            pass
        self.records = initial_list or []
        if type is not None:
            self._type = type
        super().__init__(BuiltIns['List'])

    def __getitem__(self, index: PyValue[int]) -> Record:
        return self.records[index]

    def __setitem__(self, index: PyValue[int], value: Record):
        if not self._type.match_score(value):
            raise TypeErr(f"Line {Context.line}: invalid type for assigning to this list.")
        self.records[index] = value

    def insert(self, index: PyValue[int], value: Record):
        if not self._type.match_score(value):
            raise TypeErr(f"Line {Context.line}: invalid type for assigning to this list.")
        self.records.insert(index, value)

    def slice(self, i: PyValue[int], j: PyValue[int]):
        if i.value in (None, 0, False):
            i = None
        else:
            i = i.__index__()
        if j.value in (None, 0, False):
            j = None
        else:
            j = i.__index__()
        return self.records[i:j + 1]


def piliize(value: any):
    match value:
        case None | bool() | int() | Fraction() | float() | str() | tuple() | frozenset():
            return py_value(value)
        case list():
            return List([piliize(v) for v in value])
        case set():
            s = SetTable()
            s.records = {piliize(v) for v in value}
            return s
        case dict():
            d = DictTable()
            for k, v in value.items():
                d.records[piliize(k)] = piliize(v)  # this is not the proper full implementation
            return d
        case Record():
            return value
        case Parameter():
            return Pattern(value)
        case Matcher() as t:
            return Pattern(Parameter(t))
        case _:
            return PyObj(value)

class Function(Record):
    option_map: OptionMap
    def __init__(self, options=None, *, name=None):
        self.name = name
        self.option_map = OptionMap(options)
        super().__init__(BuiltIns[__class__.__name__])

    @property
    def mro(self):
        return self.option_map, *super().mro

    def add_option(self, *args):
        return self.option_map.add_option(*args)

    def assign_option(self, *args):
        return self.option_map.assign_option(*args)

    def __repr__(self):
        if self is Context.root:
            return 'root'
        return f"Function({self.name or ''})"


class Table(Function):
    name = None
    fields: list
    field_ids: dict[str, int]
    # records: list[Record] | dict[Record, Record] | set[Record] | None
    def __init__(self, *fields, name=None):
        self.name = name
        self.fields = []
        self.field_ids = {}
        self.option_field = None
        self.sub_slices = set([])
        for field in fields:
            self.upsert_field(field)
        self.prototype = OptionMap()
        self.option_map = OptionMap()
        Record.__init__(self, BuiltIns[__class__.__name__])

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

    def __getitem__(self, item):
        raise NotImplementedError

    def __setitem__(self, key, value):
        self.records[key] = value

    def __contains__(self, item):
        return item.table == self

    def truthy(self):
        return bool(self.records)

    def upsert_field(self, field):
        if field.name in self.field_ids:
            fid = self.field_ids[field.name]
            self.fields[fid] = field
        else:
            self.field_ids[field.name] = len(self.fields)
            self.fields.append(field)

    def add_record(self, record: Record = None, **data: Record):
        if record is None:
            record = Record(self, **data)
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
        return f"Table({self.fields})"


class ListTable(Table):
    records: list[Record]
    def __init__(self, *fields, name=None):
        super().__init__(*fields, name=name)
        self.records = []

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
        self.fields = []
        self.field_ids = {}
        self.records = [self]
        self.table = self
        self.data = []
        self.index = 0
        self.prototype = OptionMap()
        self.option_map = OptionMap()


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
    def __init__(self, *fields, name=None):
        super().__init__(*fields, name=name)
        self.records = set([])
    def __getitem__(self, key: Record):
        return key

class VirtTable(SetTable):
    records = None
    def __init__(self, *fields, name=None):
        self.records = None
        Table.__init__(self, *fields, name=name)

    @property
    def truthy(self):
        return True


class Slice(Table):
    """
    when a filter slice is created or the formula updated, it should add itself to the .slices property of each record that passes the test
    likewise, when a new record is created, it should check if it belongs to each slice of that table.
    and if a record is updated?  also update the slice just in case?
    - optimization opportunity: determine the dependencies of the formula, and then only watch for changes in those fields
    Simpler strategy: make the slices pointer like this:
    - Record._filter_slices: set[Slice] | None
    - Record.slices[]:
        if self._filter_slices is not None:
            return self._filter_slices
        slice.update(self) for slice in self.table.slices
        return self._filter_slices
    - then reset Record._filter_slices to None every time the record is modified
    Or maybe I could do something tricky with hashes?  Like hash all the relevant slots/formulas together...
        but then I still need to redo the slice every time the record is modified
    """
    def __init__(self, parent):
        self.parent = parent
        parent.sub_slices.add(self)
        super().__init__()

    def __contains__(self, item: Record):
        return self in item.slices

    def __repr__(self):
        if self.name is not None:
            return self.name
        return f"{self.__class__.__name__}<{self.parent}>"


class FilterSlice(Slice, VirtTable):
    def __init__(self, parent, filter):
        self.filter = filter
        VirtTable.__init__(self)
        super().__init__(parent)
        for record in parent.records or []:
            record.update_slices(self)

    def __contains__(self, item: Record):
        item.update_slices(self)
        return self in item.slices

class VirtSlice(Slice, VirtTable):
    def __init__(self, parent):
        VirtTable.__init__(self)
        super().__init__(parent)


class ListSlice(Slice, ListTable):
    def __init__(self, parent):
        ListTable.__init__(self)
        super().__init__(parent)


class DictSlice(Slice, DictTable):
    def __init__(self, parent):
        DictTable.__init__(self)
        super().__init__(parent)


class SetSlice(Slice, SetTable):
    def __init__(self, parent):
        SetTable.__init__(self)
        super().__init__(parent)


class Field(Record):
    def __init__(self, name: str, type, default=None, formula=None, setter=None):
        self.name = name
        self.type = type
        if default is None:
            default = py_value(None)
        if formula is None:
            formula = py_value(None)
        super().__init__(BuiltIns['Field'], name=py_value(name), type=Pattern(Parameter(type)),
                         is_formula=py_value(formula is not None),
                         default=default, formula=formula)


class Slot(Field):
    # default: Function

    def __init__(self, name, type, default=None, setter=None):
        self.default = default
        super().__init__(name, type, default, setter)

    def __repr__(self):
        return f"Slot({self.name}: {self.type}{' ('+str(self.default)+')' if self.default else ''})"


class Formula(Field):
    # formula: Function
    setter = None
    def __init__(self, name, type, formula, setter=None):
        self.formula = formula
        self.setter = setter
        super().__init__(name, type, None, formula, setter)

    def __repr__(self):
        return f"Formula({self.name}: {str(self.formula)})"

class Matcher:
    name: str | None
    invert: int = 0
    guard = None

    def __init__(self, name=None, guard=None, invert=False):
        self.name = name
        self.guard = guard
        if isinstance(guard, PyFunction):
            guard.call = guard
        self.invert = int(invert)

    def match_score(self, arg: Record) -> bool | float:
        score = self.basic_score(arg)
        if self.invert:
            score = not score
        if score and self.guard:
            result = self.guard.call(arg)
            return score * BuiltIns['bool'].call(result).value
        return score

    def basic_score(self, arg):
        # implemented by subclasses
        raise NotImplementedError

    def call_guard(self, arg: Record) -> bool:
        if self.guard:
            result = self.guard.call(arg)
            return BuiltIns['bool'].call(result).value
        return True

    def get_matchers(self):
        return frozenset([self])

    def get_rank(self):
        rank = self.rank
        if self.invert:
            rank = tuple(100 - n for n in rank)
        if self.guard:
            rank = (rank[0], rank[1] - 1, *rank[1:])
        return rank

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
        assert not isinstance(table, Slice)
        self.table = table
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool:
        return arg.table == self.table

    def __repr__(self):
        return f"TableMatcher({'!'*self.invert}{self.table}{' as '+self.name if self.name else ''})"

class SliceMatcher(Matcher):
    slices: tuple[Slice, ...]
    def __init__(self, *slices: Slice, name=None, guard=None, inverse=False):
        self.slices = slices
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool:
        for s in self.slices:
            if arg not in s:
                return False
        return True

    @property
    def rank(self):
        return 4, 0, len(self.slices)

    # def __lt__(self, other):
    #     match other:
    #         case Intersection():
    #             return self < min(other.matchers)
    #         case Union():
    #             return self < max(other.matchers)
    #         case SliceMatcher():
    #             return len(self.slices) < len(other.slices)
    #         case TableMatcher() | AnyMatcher():
    #             return True
    #         case Matcher():
    #             return False
    #         case _:
    #             return NotImplemented

    def __repr__(self):
        return f"SliceMatcher({'!'*self.invert}{self.slices[0] if len(self.slices) == 1 else self.slices}{' as '+self.name if self.name else ''})"

class ValueMatcher(Matcher):
    value: Record
    rank = 1, 0

    def __init__(self, value, name=None, guard=None, inverse=False):
        self.value = value
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool:
        return arg == self.value

    # def __lt__(self, other):
    #     match other:
    #         case Intersection():
    #             return self < min(other.matchers)
    #         case Union():
    #             return self < max(other.matchers)
    #         case ValueMatcher():
    #             return False
    #         case Matcher():
    #             return True
    #         case _:
    #             return NotImplemented

    def __repr__(self):
        return f"ValueMatcher({'!'*self.invert}{self.value}{' as '+self.name if self.name else ''})"

class FieldMatcher(Matcher):
    fields: dict[str, Matcher]

    def __init__(self, fields: dict[str, Matcher], name=None, guard=None, inverse=False):
        self.fields = fields
        super().__init__(name, guard, inverse)

    def basic_score(self, arg):
        for prop_name, matcher in self.fields:
            val = arg.get(prop_name, None)
            if val is None or not matcher.match_score(val):
                return False
        return True

    @property
    def rank(self):
        return 2, 0, len(self.fields)

    # def __lt__(self, other):
    #     match other:
    #         case Intersection():
    #             return self < min(other.matchers)
    #         case Union():
    #             return self < max(other.matchers)
    #         case FieldMatcher(fields=fields):
    #             return len(self.fields) < len(fields)
    #         case ValueMatcher():
    #             return False
    #         case Matcher():
    #             return True
    #         case _:
    #             return NotImplemented

class UnionMatcher(Matcher):
    matchers: frozenset[Matcher]
    # params: set[Parameter]  # this would make it more powerful, but not worth it for the added complexity
    # examples;
    #     int+ | str
    #     list[int] | int+

    def __init__(self, *matchers, name=None, guard=None, inverse=False):
        for type in matchers:
            if isinstance(type, Pattern):
                pass
        self.matchers = frozenset(matchers)
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool | float:
        for type in self.matchers:
            m_score = type.match_score(arg)
            if m_score:
                score = m_score / len(self.matchers)
                return score
        return 0

    @property
    def rank(self):
        m = max(self.matchers).rank
        return m[0], m[1]+1, *m[2:]

    # def __lt__(self, other):
    #     match other:
    #         case Intersection():
    #             return max(self.matchers) < min(other.matchers)
    #         case Union():
    #             return max(self.matchers) < max(other.matchers)
    #         case Matcher():
    #             return max(self.matchers) < other
    #         case _:
    #             return NotImplemented

    def __repr__(self):
        return f"UnionMatcher({'!'*self.invert}{set(self.matchers)}{' as '+self.name if self.name else ''})"

class Intersection(Matcher):
    matchers: frozenset[Matcher]

    def __init__(self, *matchers, name=None, guard=None, inverse=False):
        self.matchers = frozenset(matchers)
        super().__init__(name, guard, inverse)

    def basic_score(self, arg: Record) -> bool:
        for type in self.matchers:
            if type.match_score(arg) == 0:
                return False
        return True

    @property
    def rank(self):
        m = min(self.matchers).rank
        return m[0], m[1]-1, *m[2:]

    # def __lt__(self, other):
    #     match other:
    #         case Intersection():
    #             return min(self.matchers) < min(other.matchers)
    #         case Union():
    #             return min(self.matchers) <= max(other.matchers)
    #         case Matcher():
    #             return min(self.matchers) <= other
    #         case _:
    #             return NotImplemented

    def __repr__(self):
        return f"IntersectionMatcher({'!'*self.invert}{self.matchers}{' as ' + self.name if self.name else ''})"

class AnyMatcher(Matcher):
    rank = 100, 0
    def basic_score(self, arg: Record) -> True:
        return True

    # def __lt__(self, other):
    #     if not isinstance(other, Matcher):
    #         return NotImplemented
    #     return False

    def __repr__(self):
        return f"AnyMatcher({'!'*self.invert}{' as '+self.name if self.name else ''})"

class EmptyMatcher(Matcher):
    rank = 3, 0
    def basic_score(self, arg: Record) -> bool:
        match arg:
            case VirtTable():
                return False
            case PyValue(value=str() | tuple() | frozenset() as v) | Table(records=v):
                return len(v) == 0
            case Function(option_map=OptionMap(options=options, hashed_options=hashed_options)):
                return bool(len(options) + len(hashed_options))
            case _:
                return False

    # def __lt__(self, other):
    #     match other:
    #         case ValueMatcher() | EmptyMatcher():
    #             return False
    #         case Intersection(matchers=matchers):
    #             return all(m > self for m in matchers)
    #         case Union(matchers=matchers):
    #             return any(m >= self for m in matchers)
    #         case Matcher():
    #             return True
    #         case _:
    #             return NotImplemented

    # def __eq__(self, other):
    #     return isinstance(other, EmptyMatcher) and \
    #             (self.name, self.invert, self.guard) == (other.name, other.invert, other.guard)

    def __repr__(self):
        return f"EmptyMatcher({'!'*self.invert}{' as '+self.name if self.name else ''})"

class Parameter:
    name: str | None
    matcher: Matcher | None
    quantifier: str  # "+" | "*" | "?" | "!" | ""
    count: tuple[int, int | float]
    optional: bool
    multi: bool

    def __init__(self, matcher, name=None, quantifier=""):
        self.name = name
        match matcher:
            case Pattern(parameters=(Parameter(matcher=m, name=None, quantifier=""),)):
                self.matcher = m
            case Parameter(matcher=m, name=None, quantifier=""):
                self.matcher = m
            case Matcher():
                self.matcher = matcher
            case _:
                raise TypeError(f"Failed to create Parameter of matcher: {matcher}")
        self.quantifier = quantifier

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
        self.multi = quantifier in ("+", "*")
    quantifier = property(_get_quantifier, _set_quantifier)

    def match_score(self, value) -> int | float: ...

    def compare_quantifier(self, other):
        return "_?+*".find(self.quantifier) - "_?+*".find(other.quantifier)

    def __lt__(self, other):
        match other:
            case UnionParam():
                return self <= max(other.parameters)
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.matcher < other.matcher
        return NotImplemented

    def __le__(self, other):
        match other:
            case UnionParam():
                return self <= max(other.parameters)
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.matcher <= other.matcher
        return NotImplemented

    def __eq__(self, other):
        match other:
            case UnionParam(parameters=(param, )):
                pass
            case Parameter() as param:
                pass
            case Matcher():
                param = Parameter(other)
            case Pattern(parameters=(param, )):
                pass
            case _:
                return False
        return (self.name, self.matcher, self.quantifier) == (param.name, param.matcher, param.quantifier)

    def __hash__(self):
        return hash((self.name, self.matcher, self.quantifier))

    def __gt__(self, other):
        match other:
            case UnionParam():
                return self > max(other.parameters)
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.matcher > other.matcher
        return NotImplemented

    def __ge__(self, other):
        match other:
            case UnionParam():
                return self > max(other.parameters)
            case Parameter():
                q = self.compare_quantifier(other)
                return q < 0 or q == 0 and self.matcher >= other.matcher
        return NotImplemented

    def __repr__(self):
        return f"Parameter({self.matcher} {self.name if self.name else ''}{self.quantifier})"

class UnionParam(Parameter):
    parameters: tuple[Parameter, ...]
    def __init__(self, *parameters, name=None, quantifier=""):
        self.parameters = parameters
        super().__init__(None, name, quantifier)

    def __lt__(self, other):
        return max(self.parameters) < other

    def __le__(self, other):
        if isinstance(other, UnionParam):
            return self.parameters <= other.parameters
        return max(self.parameters) < other

    def __eq__(self, other):
        match other:
            case UnionParam(parameters=params):
                return self.parameters == params
            case Parameter():
                return len(self.parameters) == 1 and self.parameters[0] == other
            case _:
                return False

    def __gt__(self, other):
        return max(self.parameters) >= other

    def __ge__(self, other):
        if isinstance(other, UnionParam):
            return self.parameters >= other.parameters
        return max(self.parameters) >= other

    def __repr__(self):
        return f"Union({self.parameters} {self.name}{self.quantifier})"


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


class Pattern(Record):
    """
    a sequence of zero or more parameters, together with their quantifiers
    """
    def __init__(self, *parameters):
        self.parameters = parameters
        super().__init__(BuiltIns['Pattern'])  # , parameters=py_value(parameters))

    def truthy(self):
        return bool(self.parameters)

    def match_score(self, list_value: List) -> int | float:
        return self.match_zip(tuple(list_value.records))[0]

    def __len__(self):
        return len(self.parameters)

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

    def match_zip(self, args: tuple[Record, ...] = None) -> tuple[float|int, dict[str, Record]]:
        if args is None:
            return 1, {}
        if len(args) == 0 == self.min_len():
            return 1, {}
        if not self.min_len() <= len(args) <= self.max_len():
            return 0, {}
        state = MatchState(self.parameters, args)
        return self.match_zip_recursive(state)

    # def match_zip_recursive(self, args: list, i_inst=0, i_arg=0, score=0, sub_score=0, saves=None):
    #     if saves is None:
    #         saves = {}
    #     while True:
    #         if not (i_inst < len(self.parameters) and i_arg < len(args)):
    #             if i_inst == len(self.parameters) and i_arg == len(args):
    #                 break
    #             elif i_inst >= len(self.parameters):
    #                 pass
    #         alts: UnionParam
    #         match self.parameters[i_inst]:
    #             case Parameter() as param:
    #                 pass
    #             case UnionParam() as alts:
    #                 for param in alts:
    #                     sc, sa = patt.match_zip_recursive(args, i_inst, i_arg, score, sub_score, saves)
    #
    #         key: str|int = param.name or i_inst
    #         sub_score *= param.multi
    #         match_value = param.matcher.match_score(args[i_arg]) if i_arg < len(args) else 0
    #         if not match_value and not param.optional:
    #             return 0, saves
    #         match param.quantifier:
    #             case "":
    #                 # match patt, save, and move on
    #                 if not match_value:
    #                     return 0, {}
    #                 saves[key] = args[i_arg]
    #                 score += match_value
    #                 i_arg += 1
    #                 i_inst += 1
    #             case "?":
    #                 # try match patt and save... move on either way
    #                 if match_value:
    #                     branch_saves = copy_bindings(saves)
    #                     branch_saves[key] = args[i_arg]
    #                     branch = self.match_zip_recursive(
    #                         args, i_inst + 1, i_arg + 1, score + match_value, 0, branch_saves)
    #                     if branch[0]:
    #                         return branch
    #                 # saves[key] = Value(None)  # for some reason this line causes error later on in the execution!  I have no idea why, but I'll have to debug later
    #                 i_inst += 1
    #             case "+":
    #                 if key not in saves:
    #                     if not match_value:
    #                         return 0, {}
    #                     saves[key] = piliize([])
    #                 if match_value:
    #                     branch_saves = copy_bindings(saves)
    #                     branch_saves[key].value.append(args[i_arg])
    #                     sub_score += match_value
    #                     i_arg += 1
    #                     branch = self.match_zip_recursive(args, i_inst, i_arg, score, sub_score, branch_saves)
    #                     if branch[0]:
    #                         return branch
    #                 if sub_score:  #  if len(saves[key].value):
    #                     score += sub_score / len(saves[key].value)
    #                 i_inst += 1
    #             case "*":
    #                 if key not in saves:
    #                     saves[key] = py_value([])
    #                 if match_value:
    #                     branch_saves = copy_bindings(saves)
    #                     branch_saves[key].value.append(args[i_arg])
    #                     branch = self.match_zip_recursive(args, i_inst, i_arg + 1, score, sub_score + match_value, branch_saves)
    #                     if branch[0]:
    #                         return branch
    #                 if sub_score:  # if len(saves[key].value):
    #                     score += sub_score / len(saves[key].value)
    #                 else:
    #                     score += 1/36
    #                 i_inst += 1
    #     return score/len(self.parameters), saves

    def match_zip_recursive(self, state):
        state: MatchState
        while not state.done:
            param = state.param
            if isinstance(param, UnionParam):
                for param in param.parameters:
                    new_params = [*state.parameters[:state.i_param], param, *state.parameters[state.i_param + 1:]]
                    score, bindings = self.match_zip_recursive(state.branch(parameters=new_params))
                    if score:
                        return score, bindings
                return 0, {}

            key: str | int = param.name or state.i_param
            state.param_score *= param.multi
            if state.i_arg < len(state.args):
                match_value = param.matcher.match_score(state.arg)
            else:
                match_value = 0  # no arguments left to process match
            if not match_value and not param.optional and not state.bindings.get(key):
                return 0, {}
            match param.quantifier:
                case "":
                    # match patt, save, and move on
                    if not match_value:
                        return 0, {}
                    state.bindings[key] = state.arg
                    state.score += match_value
                    state.i_arg += 1
                    state.i_param += 1
                case "?":
                    # try match patt and save... move on either way
                    state.i_param += 1
                    if match_value:
                        branch = state.branch()
                        branch.i_arg += 1
                        branch.score += match_value
                        branch.bindings[key] = state.arg
                        score, bindings = self.match_zip_recursive(branch)
                        if score:
                            return score, bindings
                case "+" | "*":
                    if key not in state.bindings:
                        state.bindings[key] = []
                    if match_value:
                        branch = state.branch()
                        branch.i_arg += 1
                        branch.param_score += match_value
                        branch.bindings[key].append(state.arg)
                        score, bindings = self.match_zip_recursive(branch)
                        if score:
                            return score, bindings
                    if state.param_score:  #  if len(saves[key].value):
                        state.score += state.param_score / len(state.bindings[key])
                    state.i_param += 1
        if state.success:
            return state.score_and_bindings()
        return 0, {}

    def __lt__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.parameters < other.parameters

    def __le__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.parameters <= other.parameters

    def __eq__(self, other):
        if not isinstance(other, Pattern):
            return False
        return self.parameters == other.parameters
        # if isinstance(other, Pattern):
        #     for p1, p2 in zip(self.parameters, other.parameters):
        #         if p1 != p2:
        #             return False
        #     return True
        # if len(self.parameters) != 1:
        #     return False
        # param1 = self.parameters[0]
        # if isinstance(other, Matcher):
        #     param2 = Parameter(other)
        # else:
        #     param2 = other
        # return param1 == param2

    def __hash__(self):
        return hash(self.parameters)

    def __gt__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.parameters > other.parameters

    def __ge__(self, other):
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.parameters >= other.parameters

    def __repr__(self):
        return f"Pattern{self.parameters}"
    # def __len__(self): ...
    # def __getitem__(self, item): ...
    # # def __eq__(self, other): ...
    # def __hash__(self): ...

class MatchState:
    def __init__(self, parameters, args, i_param=0, i_arg=0, score=0, param_score=0, bindings=None):
        self.parameters = parameters
        self.args = args
        self.i_param = i_param
        self.i_arg = i_arg
        self.score = score
        self.param_score = param_score
        self.bindings = bindings or {}
        # self.done = i_param == len(parameters) and i_arg == len(args)
        if i_param > len(parameters) or i_arg > len(args):
            raise RuntimeErr(f"Line {Context.line}: Pattern error: i_param or i_arg went out of bounds.")

    @property
    def done(self):
        if self.i_param > len(self.parameters) or self.i_arg > len(self.args):
            raise RuntimeErr(f"Line {Context.line}: Pattern error: i_param or i_arg went out of bounds.")
        return self.i_param == len(self.parameters) and self.i_arg == len(self.args)

    @property
    def success(self):
        return self.done and self.score

    @property
    def param(self):
        return self.parameters[self.i_param]

    @property
    def arg(self):
        return self.args[self.i_arg]

    def branch(self, **kwargs):
        if 'bindings' not in kwargs:
            kwargs['bindings'] = self.copy_bindings()
        for key in self.__dict__:
            if key not in kwargs:
                kwargs[key] = self.__dict__[key]
        return MatchState(**kwargs)

    def copy_bindings(self):
        return self.bindings.copy()

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
        case PyValue():
            return Pattern(Parameter(ValueMatcher(val)))
        case Slice():
            return Pattern(Parameter(SliceMatcher(val)))
        case Table():
            return Pattern(Parameter(TableMatcher(val)))
        case _:
            return Pattern(Parameter(ValueMatcher(val)))


# class FuncBlock:
#     native = None
#     def __init__(self, block):
#         if hasattr(block, 'statements'):
#             self.exprs = list(map(Context.make_expr, block.statements))
#         else:
#             self.native = block
#         self.env = Context.env
#
#     def make_function(self, options, prototype, caller=None):
#         return Function(args=options, type=prototype, env=self.env, caller=caller)
#
#     def execute(self, args=None, scope=None):
#         if scope:
#             def break_():
#                 Context.pop()
#                 return scope.return_value or scope
#         else:
#             scope = Context.env
#
#             def break_():
#                 return py_value(None)
#
#         if self.native:
#             result = self.native(scope, *(args or []))
#             return break_() and result
#         for expr in self.exprs:
#             Context.line = expr.line
#             expr.evaluate()
#             if scope.return_value:
#                 break
#             if Context.break_loop or Context.continue_:
#                 break
#         return break_()
#
#     def __repr__(self):
#         if self.native:
#             return 'FuncBlock(native)'
#         if len(self.exprs) == 1:
#             return f"FuncBlock({self.exprs[0]})"
#         return f"FuncBlock({len(self.exprs)} exprs)"

class CodeBlock:
    exprs: list
    scope = None
    native: PyFunction = None
    def __init__(self, block):
        self.exprs = list(map(Context.make_expr, block.nodes))
        # if hasattr(block, 'statements'):
        #     self.exprs = list(map(Context.make_expr, block.statements))
        # else:
        #     self.native = block
        self.scope = Context.env
    def execute(self, args=None, caller=None, bindings=None):
        if self.native:
            if caller:
                return self.native(caller, *args)
            return self.native(*args)
        if args is not None:
            closure = Closure(self, args, caller, bindings)
            Context.push(Context.line, closure)

            def finish():
                Context.pop()
                return closure.return_value or caller
        else:
            def finish():  # noqa
                return py_value(None)

        for expr in self.exprs:
            Context.line = expr.line
            expr.evaluate()
            if Context.env.return_value:
                break
            if Context.break_loop or Context.continue_:
                break
        return finish()
    def __repr__(self):
        return f"CodeBlock({len(self.exprs)})"

class Closure:
    return_value = None
    def __init__(self, code_block, args=None, caller=None, bindings=None):
        self.names = bindings or {}
        self.code_block = code_block
        self.scope = code_block.scope
        self.args = args
        self.caller = caller

    def execute_moved_to_CodeBlock(self):
        fn = self.caller
        if fn:
            def finish():
                Context.pop()
                return fn.return_value or fn
        else:
            fn = Context.env
            def finish():  # noqa
                return py_value(None)

        for expr in self.exprs:
            Context.line = expr.line
            expr.evaluate()
            if fn.return_value:
                break
            if Context.break_loop or Context.continue_:
                break
        return finish()
    def __repr__(self):
        return f"Closure({len(self.names)} names; {'running' if self.return_value is None else 'finished: '+str(self.return_value)})"

class TopNamespace(Closure):
    code_block = None
    scope = None
    args = None
    caller = None
    def __init__(self, bindings: dict[str, Record]):
        self.names = bindings

class Option(Record):
    resolution = None
    value = None
    block = None
    fn = None
    alias = None
    dot_option = False
    def __init__(self, pattern, resolution=None):
        match pattern:
            case Pattern():
                self.pattern = pattern
            case Parameter() as param:
                self.pattern = Pattern(param)
            case Matcher() as t:  # str() | int() | Function() | Pattern():
                self.pattern = Pattern(Parameter(t))
            case str() as name:
                self.pattern = Pattern(Parameter(ValueMatcher(py_value(name))))
            case _:
                raise TypeErr(f"Line {Context.line}: Invalid option pattern: {pattern}")
        if resolution is not None:
            self.assign(resolution)
        super().__init__(BuiltIns['Option'], signature=self.pattern, code_block=self.resolution)

    def is_null(self):
        return (self.value and self.block and self.fn and self.alias) is None
    def not_null(self):
        return (self.value or self.block or self.fn or self.alias) is not None
    def nullify(self):
        self.resolution = None
        if self.value:
            del self.value
        if self.block:
            del self.block
        if self.fn:
            del self.fn
        if self.alias:
            del self.alias
    def assign(self, resolution):
        if self.alias:
            return self.alias.assign(resolution)
        self.nullify()
        self.resolution = resolution
        match resolution:
            case Record(): self.value = resolution
            case CodeBlock(): self.block = resolution
            case PyFunction(): self.fn = resolution
            case Option(): self.alias = resolution
            case _:
                raise ValueError(f"Line {Context.line}: Could not assign resolution {resolution} to option {self}")
    def resolve(self, args, caller, bindings=None):
        if self.alias:
            return self.alias.resolve(args, caller, bindings)
        if self.value:
            return self.value
        if self.fn:
            return self.fn(*args)
        if self.block is None:
            raise NoMatchingOptionError("Could not resolve null option")
        if self.dot_option:
            caller = args[0]
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
