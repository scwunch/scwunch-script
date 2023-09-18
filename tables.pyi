from fractions import Fraction
from Env import *
from Syntax import Block
from Expressions import Expression
from typing import TypeVar, Generic

FlexiPatt = Pattern | Parameter | Matcher | str
PyFunction = type(lambda : None)

# class FunctionRecordPrototype:
#     name: str
#     # mro: tuple[Function, ...]
#     env: Record
#     # caller: Function | None  # self
#     options: list[Option]
#     # named_options: dict[str, Option]
#     hashed_options: dict[tuple[Record, ...], Option]
#     # args: list[Option]
#     block: Block
#     # exec: any
#     # return_type: Pattern
#     # return_value: Record | None
#     # value: val_types
#     is_null: bool
#     def add_option(self, pattern: FlexiPatt, resolution: opt_resolution = None) -> Option: ...
#     def remove_option(self, pattern: FlexiPatt): ...
#     def assign_option(self, pattern: Pattern, resolution: opt_resolution = None) -> Option: ...
#     def index_of(self, key: list[Function]) -> int | None: ...
#     def select_and_bind(self, key: list[Record] | tuple[Record], walk_prototype_chain=True, ascend_env=False) \
#             -> tuple[Option, dict[str, Record]]: ...
#     def select_by_pattern(self, patt, default=None, ascend_env=False) -> Option | None: ...
#     def select_by_name(self, name: str, ascend_env=True) -> Option | None: ...
#     def select_by_value(self, value: tuple[Record, ...], ascend_env=True) -> Option | None: ...
#     def call(self, *key: Record, copy_option=True, ascend=False) -> Record: ...
#     def deref(self, name: str, ascend_env=True) -> Function: ...
#     def init(self, pattern: Pattern, key: list[Record], parent: Function = None, copy=True) -> Function: ...
#     def instanceof(self, prototype: Function) -> float: ...
#     def clone(self) -> Function: ...

class Record:
    name: str | None
    table: Table
    # _filters: set[PredicateSlice] | None
    # filters: set[PredicateSlice]
    # slices: list[Slice]
    data: list[Record]
    key: Record
    index: int | None
    truthy: bool = True
    mro: tuple[Trait, ...]
    def __init__(self, table, *data_tuple: Record, **data_dict: Record): ...
    def get(self, name: str, *default) -> Record: ...
    # def get_by_index(self, index: int) -> Record: ...
    def set(self, name: str, value: Record): ...
    # def set_by_index(self, index: int, value: Record): ...
    def call(self, *key: Record) -> Record: ...

    def select(self, *key: Record) -> tuple[Option | None, dict | None]: ...
    # def update_slices(self, *slices: PredicateSlice): ...
    def hashable(self) -> bool: ...
    def to_string(self) -> PyValue[str]: ...
    def __index__(self) -> int: ...


T = TypeVar('T', None, bool, int, Fraction, float, str, tuple, frozenset, list, dict)
A = TypeVar('A')
class PyValue(Record, Generic[T]):
    value: T
    def __init__(self, table: Table, value: T): ...

# class PyValue(Record):
#     value: None | bool | int | Fraction | float | str | tuple | frozenset
#     def __init__(self, table: Table, value: None | bool | int | Fraction | float | str | tuple | frozenset): ...

class PyObj(Record, Generic[A]):
    obj: any
    def __init__(self, obj): ...

List = py_value
# class List(Record):
#     records: list
#     _type: Matcher = AnyMatcher()
#     def __init__(self, initial_list: list = None, type: Matcher = None): ...
#     def __getitem__(self, index: PyValue[int]) -> Record: ...
#     def __setitem__(self, index: PyValue[int], value: Record): ...
#     def insert(self, index: PyValue[int], value: Record): ...
#     def slice(self, i: PyValue[int], j: PyValue[int]): ...
#     def splice(self, idx: PyValue[int], seq: List, del_count: int = 0):...

def py_value(value: T) -> PyValue: ...
def piliize(value: any) -> Record: ...

class Function(Record):
    trait: Trait
    slot_dict: dict[str, Record]
    formula_dict: dict[str, Function]
    setter_dict: dict[str, Function]
    def __init__(self,
                 options: dict[FlexiPatt, opt_resolution] = None,
                 *fields: Field,
                 name: str = None,
                 table_name: str = 'Function'): ...
    # def add_option(self, pattern: FlexiPatt, resolution: opt_resolution = None) -> Option: ...
    # def assign_option(self, pattern: FlexiPatt, resolution: opt_resolution = None) -> Option: ...

class Trait(Function):
    options: list[Option]
    hashed_options: dict[tuple[Record, ...], Option]
    field_ids: dict[str, int]
    fields: list[Field]
    trait: Trait | None = None  # its own trait, since traits can be treated like functions
    def __init__(self, options: dict[FlexiPatt, opt_resolution] = None, *fields: Field, name: str = None, own_trait: Trait = None): ...
    def get_field(self, rec: Record, index: int): ...
    def set_field(self, rec: Record, index: int, value: Record): ...
    def upsert_field(self, field: Field): ...
    def add_option(self, pattern: FlexiPatt, resolution: opt_resolution = None) -> Option: ...
    def assign_option(self, pattern: FlexiPatt, resolution: opt_resolution) -> Option: ...
    def remove_option(self, pattern: FlexiPatt): ...
    def select_and_bind(self, key: list[Record] | tuple[Record], walk_prototype_chain=True, ascend_env=False) \
            -> tuple[Option, dict[str, Record]] | tuple[None, None]: ...
    def select_by_pattern(self, patt, default=None, ascend_env=False) -> Option | None: ...
    # def add_own_option(self, pattern: FlexiPatt, resolution: opt_resolution = None) -> Option: ...
    def select_by_value(self, value: tuple[Record, ...], ascend_env=True) -> Option | None: ...

class Table(Function):
    records: list[Record] | dict[Record, Record] | set[Record] | None
    # sub_slices: set[Slice]
    # prototype: Trait  # contains all the heritable options
    traits: tuple[Trait, ...]  # the traits to be used by instances.  traits[0] is defined in the table body block
    getters: dict[str, int | Function]
    setters: dict[str, int | Function]
    defaults: tuple[Record, ...]  # this is for instantiation of Records
    def __init__(self, *traits: Trait, fields: list[Field] = (), name: str = None): ...
    def integrate_traits(self): ...
    def get_field(self, name: str) -> Field | None: ...
    def __getitem__(self, key: Record) -> Record | None: ...
    def __setitem__(self, key: Record, value: Record): ...
    def __contains__(self, item: Record): ...
    def upsert_field(self, field: Field): ...
    def add_record(self, record: Record): ...
    def _add_record(self, record: Record): ...

class ListTable(Table):
    records: list[Record]
    # def __init__(self, *traits: Trait, fields: list[Field], name: str = None): ...
    def __getitem__(self, key: PyValue[int]): ...

class MetaTable(ListTable): ...
class BootstrapTraitTable(ListTable): ...

class DictTable(Table):
    records: dict[Record, Record]
    key_field: int
    def __init__(self, key_field: int = 0, *field_tuple, **fields): ...

class SetTable(Table):
    records: set[Record]
    # def __init__(self, *fields, name=None): ...

class VirtTable(SetTable):
    records = None
    # def __init__(self, *fields, name=None): ...

# class Slice(Table):
#     parent: Table
#     def __init__(self, parent: Table): ...
#
# class PredicateSlice(Slice, VirtTable):
#     predicate: Function | PyFunction
#     def __init__(self, parent: Table, predicate: Function | PyFunction): ...
#
# class VirtSlice(Slice, VirtTable): ...
#
# class ListSlice(Slice, ListTable): ...
#
# class DictSlice(Slice, DictTable): ...
#
# class SetSlice(Slice, SetTable): ...

class Field(Record):
    name: str
    type: Matcher = None
    def __init__(self, name: str,
                 type: Matcher = None,
                 default: Function | None = py_value(None),
                 formula: Function = py_value(None)): ...

    def get_data(self, rec: Record, idx: int): ...
    def set_data(self, rec: Record, idx: int, value): ...

class Slot(Field):
    default: Function | None
    def __init__(self, name: str, type: Matcher, default: Record = None): ...
    def get_data(self, rec: Record, idx: int): ...
    def set_data(self, rec: Record, idx: int, value): ...

class Formula(Field):
    formula: Function
    def __init__(self, name: str, type: Matcher, formula: Function): ...
    def get_data(self, rec: Record, idx: int): ...

class Setter(Field):
    fn: Function
    def __init__(self, name: str, fn: Function): ...
    def set_data(self, rec: Record, idx: int, value): ...

class Matcher:
    name: str | None
    invert: int = 0
    guard: Function | PyFunction = None
    rank: tuple[int, int, ...]  # for sorting
    def __init__(self, name: str = None, guard: Function | PyFunction = None, inverse=False): ...
    def issubset(self, other: Matcher) -> bool: ...
    def equivalent(self, other: Matcher) -> bool: ...
    def match_score(self, arg: Record) -> bool | float: ...
    def basic_score(self, arg) -> bool | float: ...
    def call_guard(self, arg: Record) -> bool: ...

class TableMatcher(Matcher):
    table: Table
    def __init__(self, table: Table, name: str = None, guard: Function | PyFunction = None, inverse=False): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class TraitMatcher(Matcher):
    trait: Trait
    def __init__(self, trait: Trait, name: str = None, guard: Function | PyFunction = None, inverse=False): ...

# class SliceMatcher(Matcher):
#     slices: tuple[Slice, ...]
#     def __init__(self, *slices: Slice, name=None, guard=None, inverse=False): ...
class ValueMatcher(Matcher):
    value: Record
    def __init__(self, value: Record, name: str = None, guard: Function | PyFunction = None, inverse=False): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class FieldMatcher(Matcher):
    fields: dict[str, Matcher]
    def __init__(self, fields: dict[str, Matcher], name=None, guard=None, inverse=False): ...

class UnionMatcher(Matcher):
    matchers: frozenset[Matcher]
    # params: set[Parameter]  # this would make it more powerful, but not worth it for the added complexity
    # examples;
        # int+ | str
        # list[int] | int+
    def __init__(self, *monads: Matcher, name: str = None, guard: Function | PyFunction = None, inverse=False): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class Intersection(Matcher):
    matchers: frozenset[Matcher]
    def __init__(self, *monads: Matcher, name: str = None, guard: Function | PyFunction = None, inverse=False): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class AnyMatcher(Matcher):
    def __eq__(self, other): ...
    def __hash__(self): ...

class Parameter:
    matcher: Matcher | None
    name: str | None
    quantifier: str  #  "+" | "*" | "?" | ""
    count: tuple[int, int | float]
    optional: bool
    multi: bool

    def __init__(self, matcher: Matcher | None, name: str = None, quantifier: str = ""): ...
    def issubset(self, other: Parameter) -> bool: ...
    def try_get_matchers(self) -> tuple[Matcher, ...] | list[Matcher]: ...
    def compare_quantifier(self, other: Parameter) -> int: ...
    def match_score(self, value: Record) -> int | float: ...

class UnionParam(Parameter):
    parameters: tuple[Parameter, ...]
    def __init__(self, *parameters: Parameter, name: str = None, quantifier: str = ""): ...

class AnyParam(Parameter):
    pass

def copy_bindings(saves: dict[str, Function]) -> dict[str, Function]: ...

class Pattern(Record):
    parameters: tuple[Parameter]
    def __init__(self, *parameters: Parameter): ...
    def match_score(self, *values: Record) -> int | float: ...
    def issubset(self, other: Pattern) -> bool: ...
    def min_len(self) -> int: ...
    def max_len(self) -> int | float: ...
    # def match_zip_recursive(self, args: list = None, i_inst: int = 0, i_arg: int = 0, score: float = 0, sub_score: float = 0, saves: dict[str, Record] = None) \
    #         -> 0 | tuple[int|float, dict[str, Record]]: ...
    def match_zip_recursive(self, state: MatchState) -> 0 | tuple[int|float, dict[str, Record]]: ...
    def match_zip(self, args: tuple[Record, ...] = None) -> tuple[float|int, dict[str, Record]] | 0: ...
    def try_get_params(self) -> tuple[Parameter, ...]: ...
    def try_get_matchers(self) -> tuple[Matcher, ...] | list[Matcher]: ...
    def compare(self, other: Pattern): ...
    def __len__(self): ...
    def __getitem__(self, item): ...
    def __eq__(self, other): ...
    def __hash__(self): ...

class MatchState:
    parameters: list[Parameter]
    args: list[Record]
    i_param: int
    i_arg: int
    score: int | float
    param_score: int | float
    bindings: dict[str, Record | list[Record]]
    def __init__(self, parameters, args, i_param=0, i_arg=0, score=0, param_score=0, bindings=None): ...

# class SubPattern:
#     patterns: tuple[Pattern, ...]
#     def __init__(self, *patterns: Pattern): ...
#
# class UnionPattern(SubPattern): ...
#
# class IntersectionPattern(SubPattern): ...

def patternize(value: Record) -> Pattern: ...

class Args(Record):
    positional_arguments: list[Record] | tuple[Record]
    named_arguments: dict[str, Record]
    flags: set[str]
    def __init__(self, *args: Record, flags: set[str], **kwargs: Record): ...


class CodeBlock:
    exprs: list[Expression]
    scope: Closure | None
    native: PyFunction
    def __init__(self, block: Block | PyFunction): ...
    def execute(self, args: tuple[Record, ...] = None,
                caller: Record = None,
                bindings: dict[str, Record] = None,
                *, fn: Function = None): ...

class Native(CodeBlock):
    def __init__(self, fn: PyFunction): ...

class Closure:
    names: dict[str, Record]
    code_block: CodeBlock
    scope: Closure
    args: tuple[Record, ...]
    caller: Record
    fn: Function
    return_value: Record = None
    def __init__(self, code_block: CodeBlock | None,
                 args: tuple[Record, ...] = None,
                 caller: Record = None,
                 bindings: dict[str, Record] = None,
                 fn: Function = None): ...
    def assign(self, name: str, value: Record): ...

class TopNamespace(Closure):
    code_block = None
    scope = None
    args = None
    caller = None
    def __init__(self, bindings: dict[str, Record]): ...

opt_resolution = Record | CodeBlock | PyFunction | None
class Option(Record):
    env: Closure
    pattern: Pattern
    resolution: opt_resolution
    value: Record
    block: CodeBlock
    fn: PyFunction
    alias: Option
    dot_option: bool
    def __init__(self, pattern: FlexiPatt, resolution: opt_resolution = None): ...
    def is_null(self) -> bool: ...
    def not_null(self) -> bool: ...
    def nullify(self): ...
    def assign(self, val_or_block: opt_resolution): ...
    def resolve(self, args: tuple[Record, ...], caller: Record = None, bindings: dict[str|int, Record] = None) \
            -> Record: ...


class Operator:
    text: str
    prefix: int | None
    postfix: int | None
    binop: int | None
    ternary: str | None
    associativity: str
    fn: Function
    chainable: bool
    static: bool | PyFunction
    def __init__(self, text, fn:Function=None, prefix:int=None, postfix:int=None, binop:int=None,
                 ternary:str=None, associativity='left', chainable:bool=False, static=False): ...
    def eval_args(self, lhs, rhs) -> list[Record]: ...
