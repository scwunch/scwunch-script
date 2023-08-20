import math
from fractions import Fraction
from Env import *
from typing import TypeVar, Generic

PyFunction = type(lambda : None)


class FunctionRecordPrototype:
    return_value = None
    name: str = None
    # mro = ()
    env = None
    caller = None  # self
    block = None
    return_type = None
    is_null = True
    options = None
    hashed_options = None
    def __init__(self):
        # self.mro = (self, )
        self.env = Context.env
        if self.options is None:
            self.options = []
        if self.hashed_options is None:
            self.hashed_options = {}

    def add_option(self, pattern, resolution=None):
        option = Option(pattern, resolution)

        # try to hash option
        key: list[Record] = []
        for parameter in option.pattern.parameters:
            t = parameter.matcher
            if isinstance(t, ValueMatcher) and t.guard is None and t.value.hashable():
                key.append(t.value)
            else:
                self.options.insert(0, option)
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

    @property
    def mro(self):
        self: Record
        return self, *(s.prototype for s in self.slices), self.table.prototype

    def select_and_bind(self, key: list, walk_prototype_chain=True, ascend_env=False):
        try:
            return self.hashed_options[tuple(key)], {}
        except (TypeError, KeyError):
            pass
        option = bindings = high_score = 0
        for opt in self.options:
            score, saves = opt.pattern.match_zip(key)
            if score == 1:
                return opt, saves
            if score > high_score:
                high_score = score
                option, bindings = opt, saves
        if option:
            return option, bindings
        if walk_prototype_chain and self.mro:
            for t in self.mro[1:]:
                try:
                    return t.select_and_bind(key, False, ascend_env)
                except NoMatchingOptionError:
                    pass
        if ascend_env and self.env:
            try:
                return self.env.select_and_bind(key, walk_prototype_chain, True)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: key {key} not found in {self.name or self}")
        # -> tuple[Option, dict[str, Function]]:

    def select_by_pattern(self, patt=None, default=None, ascend_env=False):
        # return [*[opt for opt in self.options if opt.pattern == patt], None][0]
        for opt in self.options:
            if opt.pattern == patt:
                return opt
        if ascend_env and self.env:
            return self.env.select_by_pattern(patt, default)
        return default

    def select_by_name(self, name: str, ascend_env=True):
        return self.select_by_value((py_value(name),), ascend_env)

    def select_by_value(self, values, ascend_env=True):
        # try self first
        for fn in self.mro:
            try:
                return fn.hashed_options[values]
            except KeyError:
                continue

        if ascend_env and self.env:
            try:
                return self.env.select_by_value(values, True)
            except NoMatchingOptionError:
                pass
        raise NoMatchingOptionError(f"Line {Context.line}: {values} not found in {self}")

    def call(self, key, ascend=False):
        try:
            option, bindings = self.select_and_bind(key, ascend_env=ascend)
        except NoMatchingOptionError as e:
            # if ascend and self.env:
            #     try:
            #         return self.env.call(*key, ascend=True)
            #     except NoMatchingOptionError:
            #         pass
            raise e
        return option.resolve(key, self, bindings)

    def deref(self, name: str, ascend_env=True):
        option = self.select_by_value((py_value(name),), ascend_env)
        return option.resolve((), self)

    def instanceof(self, type):
        self: Record
        return type == self.table or type in self.slices
        # types = self.mro[1:]
        # if prototype in types:
        #     return 1
        # for t in types:
        #     k = t.instanceof(prototype)
        #     if k:
        #         return k / 2
        # return 0
        # # return len(self.type) and int(prototype in self.type) or self.type.instanceof(prototype)/2

class Prototype(FunctionRecordPrototype):
    """ essentially just container to hold the options of tables, a proxy for methods """
    def __init__(self, table):
        self.table = table
        super().__init__()

    @property
    def mro(self):
        return (self,)

    def __repr__(self):
        return f"Prototype({self.table})"

class Record(FunctionRecordPrototype):
    # table: Table
    _filter_slices: set | None = None
    # slices: set[Slice]
    # data: dict[int, Record]
    # key: Record
    truthy = True
    def __init__(self, table, **data):
        super().__init__()
        self.table = table
        self._non_filter_slices = set(())
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
    def slices(self):
        if isinstance(self, Table):
            pass  # this shouldn't happen... Table class has its own slices property
        if self._filter_slices is None:
            self.update_slices()
        return self._filter_slices.union(self._non_filter_slices)

    def update_slices(self, *slices):
        if self._filter_slices is None or not slices:
            self._filter_slices = set(filter(lambda s: s.filter.call(self).truthy,
                                             self.table.slices))
        else:
            for s in slices:
                if s.filter.call(self).truthy:
                    self._filter_slices.add(s)

    NO_DEFAULT = object()
    def get(self, name: str, default=NO_DEFAULT):
        try:
            index = self.table.field_ids[name]
        except KeyError:
            raise SlotErr(f"Line {Context.line}: no field found with name '{name}'.")
        return self.get_by_index(index)

    def get_by_index(self, index: int, default=NO_DEFAULT):
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
    def __init__(self, table, value: T):
        self.value = value
        super().__init__(table, key=self)

    def to_string(self):
        if self.instanceof(BuiltIns['num']) and self.table is not BuiltIns['bool']:
            return py_value(write_number(self.value, Context.settings['base']))
        # if self.instanceof(BuiltIns['list']):
        #     return py_value(f"[{', '.join(v.to_string().value for v in self.value)}]")
        return py_value(str(self.value))

    @property
    def truthy(self):
        return bool(self.value)

    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        return isinstance(other, PyValue) and self.value == other.value or self.value == other
    def __repr__(self):
        return repr(self.value)  # f"Record({self.value})"

def py_value(value: None | bool | int | Fraction | float | str | tuple | frozenset):
    match type(value).__name__:
        case 'NoneType':
            return BuiltIns['blank']
        case 'bool':
            return BuiltIns[str(value).lower()]
        case 'Fraction' | 'int':
            table = BuiltIns['ratio']
            if value.denominator == 1:
                value = int(value)
        case 'float' | 'str' as t:
            table = BuiltIns[t]
            return PyValue[type(value)](table, value)
        case 'tuple' | 'frozenset' as t:
            table = BuiltIns[t.title()]
            value = type(value)(piliize(v) for v in value)
        case unknown_type:
            raise TypeErr(f"Unhandled python type for PyValue: {unknown_type} {value}")
    if isinstance(table, SetTable):
        return PyValue[type(value)](table, value)
    return table[value] or PyValue[type(value)](table, value)


class PyObj(Record, Generic[A]):
    def __init__(self, obj):
        self.obj = obj
        super().__init__(BuiltIns['PyObj'])


def piliize(value: any):
    match value:
        case None | bool() | int() | Fraction() | float() | str() | tuple() | frozenset():
            return py_value(value)
        case list():
            ls = ListTable()
            ls.records = [piliize(v) for v in value]
            return ls
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


class Table(Record):
    name = None
    fields: list
    field_ids: dict[str, int]
    # records: list[Record] | dict[Record, Record] | set[Record] | None
    def __init__(self, *field_tuple, **fields):
        self.fields = []
        self.field_ids = {}
        self.option_field = None
        self._slices = set([])
        for field in field_tuple:
            self.upsert_field(field)
        for name, monad in fields.items():
            self.upsert_field(Slot(name, monad))
        self.prototype = Prototype(self)
        super().__init__(BuiltIns['Table'])
        #
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

    @property
    def slices(self):
        return self._slices

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
    def __init__(self, *field_tuple, **fields):
        super().__init__(*field_tuple, **fields)
        self.records = []

    def __getitem__(self, key: PyValue[int]):
        try:
            return self.records[key.value]
        except (TypeError, AttributeError):
            raise RuntimeErr(f"Index must be integer in range for ListTable.")
        except IndexError:
            return None


class MetaTable(ListTable):
    def __init__(self):
        self.fields = []
        self.field_ids = {}
        self.records = [self]
        self.table = self
        self.data = []
        self.index = 0


class DictTable(Table):
    records: dict[Record, Record]
    key_field: int
    def __init__(self, key_field: int = 0, *field_tuple, **fields):
        self.key_field = key_field
        super().__init__(*field_tuple, **fields)
        self.records = {}
    def __getitem__(self, key: Record):
        return self.records.get(key)

class SetTable(Table):
    records: set[Record]
    def __init__(self, *field_tuple, **fields):
        super().__init__(*field_tuple, **fields)
        self.records = set([])
    def __getitem__(self, key: Record):
        return key

class VirtTable(SetTable):
    records = None
    def __init__(self, *field_tuple, **fields):
        self.records = None
        Table.__init__(self, *field_tuple, **fields)

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
        parent.slices.add(self)
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
    inverse: bool = False
    guard = None

    def __init__(self, name=None, guard=None, inverse=False):
        self.name = name
        self.guard = guard
        self.inverse = inverse

    def call_guard(self, arg: Record) -> bool:
        if self.guard:
            result = self.guard.call(arg)
            return BuiltIns['bool'].call(result).value
        return True

    def get_matchers(self):
        return frozenset([self])

class TableMatcher(Matcher):
    table: Table

    def __init__(self, table, name=None, guard=None, inverse=False):
        self.table = table
        super().__init__(name, guard, inverse)

    def match_score(self, arg: Record) -> int | float:
        if arg.table == self.table:
            return int(self.call_guard(arg))
        return 0

    def __repr__(self):
        return f"TableMatcher({'!'*self.inverse}{self.table}{' as '+self.name if self.name else ''})"

class SliceMatcher(Matcher):
    slices: tuple[Slice, ...]
    def __init__(self, *slices: Slice, name=None, guard=None, inverse=False):
        self.slices = slices
        super().__init__(name, guard, inverse)

    def match_score(self, arg: Record) -> int | float:
        for s in self.slices:
            if arg not in s:
                return 0
        return self.call_guard(arg)

    def __repr__(self):
        return f"SliceMatcher({'!'*self.inverse}{self.slices}{' as '+self.name if self.name else ''})"

class ValueMatcher(Matcher):
    value: Record

    def __init__(self, value, name=None, guard=None, inverse=False):
        self.value = value
        super().__init__(name, guard, inverse)

    def match_score(self, arg: Record) -> int | float:
        if arg == self.value:
            return int(self.call_guard(arg))
        return 0

    def __repr__(self):
        return f"ValueMatcher({'!'*self.inverse}{self.value}{' as '+self.name if self.name else ''})"

class Union(Matcher):
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

    def match_score(self, arg: Record) -> int | float:
        for type in self.matchers:
            m_score = type.match_score(arg)
            if m_score:
                score = m_score / len(self.matchers)
                return score * self.call_guard(arg)
        return 0

    def get_matchers(self):
        return self.matchers

    def __repr__(self):
        return f"UnionMatcher({'!'*self.inverse}{self.matchers}{' as '+self.name if self.name else ''})"

class Intersection(Matcher):
    matchers: frozenset[Matcher]

    def __init__(self, *matchers, name=None, guard=None, inverse=False):
        self.matchers = frozenset(matchers)
        super().__init__(name, guard, inverse)

    def match_score(self, arg: Record) -> int | float:
        for type in self.matchers:
            if type.match_score(arg) == 0:
                return 0
        return int(self.call_guard(arg))

    def __repr__(self):
        return f"IntersectionMatcher({'!'*self.inverse}{self.matchers}{' as ' + self.name if self.name else ''})"

class AnyMatcher(Matcher):
    def match_score(self, arg: Record) -> int | float:
        return int(self.call_guard(arg))

    def __repr__(self):
        return f"AnyMatcher({'!'*self.inverse}{' as '+self.name if self.name else ''})"

class Parameter:
    name: str | None
    matcher: Matcher | None
    quantifier: str  # "+" | "*" | "?" | ""
    count: tuple[int, int | float]
    optional: bool
    multi: bool

    def __init__(self, matcher, name=None, quantifier=""):
        self.name = name
        self.matcher = matcher
        self.quantifier = quantifier
        match quantifier:
            case "":
                self.count = (1, 1)
            case "?":
                self.count = (0, 1)
            case "+":
                self.count = (1, math.inf)
            case "*":
                self.count = (0, math.inf)
        self.optional = quantifier in ("?", "*")
        self.multi = quantifier in ("+", "*")

    def match_score(self, value) -> int | float: ...

    def try_get_matchers(self):
        if self.quantifier or self.name is not None:
            raise TypeError
        if isinstance(self.matcher, Union):
            return self.matcher.matchers
        return (self.matcher,)

    def __repr__(self):
        return f"Parameter({self.matcher} {self.name}{self.quantifier})"

class UnionParam(Parameter):
    parameters: tuple[Parameter, ...]
    def __init__(self, *parameters, name=None, quantifier=""):
        self.parameters = parameters
        super().__init__(None, name, quantifier)

    def try_get_matchers(self):
        if self.quantifier or self.name is not None:
            raise TypeError
        matchers = []
        for param in self.parameters:
            matchers.extend(param.try_get_matchers())
        return matchers

    def __repr__(self):
        return f"Union({self.parameters} {self.name}{self.quantifier})"


class AnyParam(Parameter):
    def __repr__(self):
        return f"AnyParam({self.name}{self.quantifier})"


def copy_bindings(saves):
    new_saves = {}
    for key, value in saves.items():
        if hasattr(value, "value") and isinstance(value.value, list):
            raise NotImplementedError
            # new_saves[key] = py_value(value.value.copy())
        else:
            new_saves[key] = value
    return new_saves

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

    def match_score(self, list_value: ListTable) -> int | float:
        return self.match_zip(list_value.records)[0]

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

    def match_zip(self, args: list[Record] = None) -> tuple[float|int, dict[str, Record]]:
        if args is None:
            return 1, {}
        if len(args) == 0 == self.min_len():
            return 1, {}
        if not self.min_len() <= len(args) <= self.max_len():
            return 0, {}
        return self.match_zip_recursive(MatchState(self.parameters, args))

    def try_get_params(self):
        if len(self.parameters) != 1:
            raise TypeErr(f"Line {Context.line}: Cannot get union of patterns with multiple parameters.")
        param = self.parameters[0]
        if isinstance(param, UnionParam):
            return param.parameters
        return param,

    def try_get_matchers(self):
        params = self.try_get_params()
        matchers = []
        for p in params:
            matchers.extend(p.try_get_matchers())
        return matchers

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
            match_value = param.matcher.match_score(state.arg) if state.i_arg < len(state.args) else 0
            if not match_value and not param.optional:
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
        if hasattr(block, 'statements'):
            self.exprs = list(map(Context.make_expr, block.statements))
        else:
            self.native = block
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
    def resolve(self, args, env, bindings=None):
        if self.alias:
            return self.alias.resolve(args, env, bindings)
        if self.value:
            return self.value
        if self.fn:
            return self.fn(*args)
        if self.block is None:
            raise NoMatchingOptionError("Could not resolve null option")
        if self.dot_option:
            caller = args[0]
        else:
            caller = env
            # btw: this is possibly the third time the env is getting overriden: it's first set when the FuncBlock is
            # defined, then overriden by the env argument passed to self.resolve, and finally here if it is a dot-option
            # ... I should consider making a multi-layer env, or using the prototype, or multiple-inheritance type-thing
        return self.block.execute(args, caller, bindings)
        # fn = self.block.make_function(bindings or {}, env, caller)
        # Context.push(Context.line, fn, self)
        # return self.block.execute(args, fn)

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


class Function(Record):
    def __init__(self, options=None, args=None, type=None, env=None, caller=None, name=None):
        self.name = name
        # self.type = type or BuiltIns['fn']  # if isinstance(type, tuple) else (type or BuiltIns['fn'],)
        # self.mro = (self, *self.type.mro)
        self.env = env or Context.env
        self.caller = caller
        self.options = []
        self.args = []
        self.hashed_options = {}
        if args:
            for patt, val in args.items():
                self.args.append(self.add_option(patt, val))
        if options:
            for patt, val in options.items():
                self.add_option(patt, val)
        super().__init__(BuiltIns['Function'])

    def __repr__(self):
        if self is Context.root:
            return 'root'
        return f"Function({self.name or ''})"
        # if self.table == Context.root:
        #     return 'root.main'
        # prefix = self.name or ""
        # return prefix + "{}"

# class ListFunc(Function):
#     def __init__(self, *values: Value):
#         super().__init__(type=BuiltIns['List'])
#         for i, val in enumerate(values):
#             self.add_option(ListPatt(Parameter(ValuePattern(Value(i)))), val)
#     def push(self, val):
#         self.add_option(numbered_patt(self.len()+1), val)
#         return self
#     def pop(self, index: int = -1) -> Value:
#         return self.remove_option(Value(index))
#     def len(self) -> int:
#         return len(self.options)


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
