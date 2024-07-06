import Env
from Env import *
from tables import *
from BuiltIns import *
from Syntax import *

print(f"loading module: {__name__} ...")

identity = lambda x: x

Op[';'].fn = Function({AnyPlusPattern: lambda *args: args[-1]})

def eval_assign_args(lhs: Node, rhs: Node) -> Args:
    if isinstance(rhs, Block):
        val = Closure(rhs)
    else:
        val = rhs.evaluate()

    match lhs:
        case Token(type=TokenType.Name, source_text=name):
            """ name := value """
            return Args(py_value(name), val)  # str, any
        case ListNode():
            """ [key] := value """
            return Args(lhs.evaluate(), val)
        case OpExpr(op=Operator(text='.'),
                    terms=(Node() as fn_node, Token(type=TokenType.Name, source_text=name))):
            """ foo.bar := value """
            location = fn_node.evaluate()
            return Args(location, py_value(name), val)  # any, str, any
        case OpExpr(op=Operator(text='.'),
                    terms=(Node() as fn_node, ListNode(list_type=ListType.Args) as args)):
            """ foo[key] := value """
            location = fn_node.evaluate()  # note: if location is not a function, a custom option must be added to =
            key: Args = args.evaluate()
            return Args(location, key, val)  # fn, args, any
        case OpExpr(op=Operator(text=','), terms=keys):
            keys
        case OpExpr(op=Operator(text=',')):
            raise SyntaxErr(f"Line {lhs.line}: Invalid LHS for assignment.  If you want to assign to a key, use either "
                            f"`[key1, key2] = value` or `key1, key2: value`")
        case _:
            raise SyntaxErr(f"Line {lhs.line}: Invalid lhs for assignment: {lhs}")

def set_or_assign_option(*args, operation: Function = None):
    match args:
        case Function() as fn, Args() as args, Record() as val:
            if operation:
                val = operation.call(fn.call(args), val)
            fn.assign_option(args, val)
        case Record() as rec, PyValue(value=str() as name), Record() as val:
            if operation:
                val = operation.call(rec.get(name), val)
            rec.set(name, val)
        case _:
            raise ValueError("Incorrect value types for assignment")
    return val

Op['='].eval_args = eval_assign_args
Op['='].fn = Function({ArgsMatcher(StringParam, AnyParam): lambda name, val: Context.env.assign(name.value, val),
                       ArgsMatcher(FunctionParam, AnyParam, AnyParam): set_or_assign_option,
                       ArgsMatcher(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val)
                       }, name='=')

def null_assign(rec_or_name: Record | PyValue, name_or_val: PyValue | Record, val_or_none: Record = None):
    if val_or_none is None:
        name = rec_or_name.value
        val = name_or_val
        get = Context.deref
        set = Context.env.assign
    else:
        rec = rec_or_name
        name = name_or_val.value
        val = val_or_none
        get = rec.get
        set = rec.set
    existing_value = get(name, BuiltIns['blank'])
    if existing_value is BuiltIns['blank']:
        set(name, val)
        return val
    else:
        return existing_value

def null_assign_rec(rec, name, val):
    name = name.value
    existing_value = rec.get(name, BuiltIns['blank'])
    if existing_value is BuiltIns['blank']:
        rec.set(name, val)
        return val
    else:
        return existing_value


Op['??='].fn = Function({ArgsMatcher(StringParam, AnyParam): null_assign,
                         ArgsMatcher(FunctionParam, AnyParam, AnyParam):
                             lambda fn, args, val: fn.assign_option(args, val, no_clobber=True).value
                                                or BuiltIns['blank'],
                         ArgsMatcher(AnyParam, StringParam, AnyParam): null_assign
                         }, name='??=')
Op['??='].eval_args = eval_assign_args

def eval_colon_args(lhs: Node, rhs: Node) -> Args:
    if isinstance(rhs, Block):
        pattern = params = None
        match lhs:
            case OpExpr(op=Operator(text='.'), terms=[fn_node, ParamsNode() as params]):
                #
                match fn_node:
                    case Token(type=TokenType.Name, source_text=name):
                        fn = Context.deref(name, None)
                        if fn is None:
                            fn = Function(name=name)
                            Context.env.locals[name] = fn
                    case _:
                        fn = fn_node.evaluate()
                if not isinstance(fn, Function):
                    raise TypeErr(f'Line {fn_node.line}: Cannot assign option to {fn.table} {fn}')
            case ParamsNode() as params:
                if (fn := Context.env.fn) is None:
                    raise SyntaxErr(f"Line {lhs.line}: this syntax is not available in this context.  "
                                    f"Try within a function, trait, or table block.")
            case OpExpr(op=Operator(text='.'), terms=[Token(type=TokenType.Name, source_text=name)]):
                """ .foo: ... """
                fn = Context.deref(name, None)
                if fn is None:
                    fn = Function(name=name)
                    Context.env.locals[name] = fn
                if not isinstance(fn, Function):
                    raise TypeErr(f'Line {lhs.line}: Cannot assign option to {fn.table} {fn}')
                match Context.env.fn:
                    case Table() | Trait() as t:
                        pattern = ArgsMatcher(Parameter(t, 'self'))
                    case Function():
                        pattern = ArgsMatcher()
                    case None:
                        raise SyntaxErr(f"Line {lhs.line}: this syntax is not available in this context.  "
                                        f"Try within a trait, or table block.")
            case OpExpr(op=Operator(text='.'), terms=[fn_node, Token(type=TokenType.Name, source_text=name)]):
                ''' foo.bar: ... '''
                t = fn_node.evaluate()
                fn = t.get(name)
                match t:
                    case Table() | Trait():
                        pattern = ArgsMatcher(Parameter(t, 'self'))
                    case Function():
                        pattern = ArgsMatcher()
                    case None:
                        raise RuntimeErr(f"Line {lhs.line}: leftmost container term must be a table, trait, or function."
                                         f"\nIe, for `foo.bar: ...` then foo must be a table, trait, or function.")
            case _:
                raise SyntaxErr(f'Line {lhs.line}: option syntax is: `<fn>[<params>]: <block or expression>`.  '
                                f'Eg `foo[int n]: n + 2`')
        if pattern is None:
            pattern = params.evaluate()
        code_block = Closure(rhs)
        return Args(fn, pattern, code_block)
    else:
        match lhs:
            case ListNode(nodes=key_nodes) | OpExpr(op=Operator(text=','), terms=key_nodes):
                pass
            case ParamsNode():
                raise NotImplementedError
            case _:
                key_nodes = [lhs]

        key = Args(*(n.evaluate() for n in key_nodes))
        value = rhs.evaluate()
        return Args(key, value)


def eval_colon_args(lhs: Node, rhs: Node) -> Args:
    if isinstance(rhs, Block):
        resolution = Closure(rhs)
    else:
        resolution = rhs.evaluate()

    match lhs:
        case ParamsNode():
            """ [params]: ... """
            return Args(lhs.evaluate(), resolution)
        case OpExpr(op=Operator(text='.'), terms=terms):
            match terms:
                case [fn_node, ParamsNode() as params]:
                    """ foo[key]: ... """
                    match fn_node:
                        case Token(type=TokenType.Name, source_text=name):
                            location = Context.deref(name, None)
                            if location is None:
                                location = Function(name=name)
                                Context.env.locals[name] = location
                        case _:
                            location = fn_node.evaluate()
                    key = params.evaluate()
                    return Args(location, key, resolution)  # fn, args, any
                case [Token(type=TokenType.Name, source_text=name)]:
                    """ .foo: ... """
                    location = Context.env.fn
                    if location is None:
                        raise EnvironmentError(f"Line {Context.line}: illegal .dot option found")
                    fn = Context.deref(name, None)
                    if fn is None:
                        fn = Function(name=name)
                        Context.env.locals[name] = fn
                    key = ArgsMatcher(Parameter(location, 'self'))
                    # maybe this should have no self parameter if in Function context?
                    return Args(fn, key, resolution)
                case [OpExpr(op=Operator(text='.'), terms=[Token(type=TokenType.Name, source_text=name), ParamsNode() as params])]:
                    """ .foo[params] """
                    """ Is this actually the correct pattern to match .foo[params]?"""
                    raise NotImplementedError
                case [table_or_trait, Token(type=TokenType.Name, source_text=name)]:
                    ''' Foo.bar: ... '''
                    t = table_or_trait.evaluate()
                    fn = t.get(name)
                    match t:
                        case Table() | Trait():
                            pattern = ArgsMatcher(Parameter(t, 'self'))
                        case Function():
                            pattern = ArgsMatcher()
                        case None:
                            raise RuntimeErr(f"Line {lhs.line}: leftmost container term must be a table, trait, or function."
                                             f"\nIe, for `foo.bar: ...` then foo must be a table, trait, or function.")
                    return Args(fn, pattern, resolution)
                case _:
                    raise SyntaxErr(f"Line {Context.line}: Unrecognized syntax for LHS of assignment.")
        case OpExpr(op=Operator(text=','), terms=keys):
            """ key1, key2: ... """
            key = Args(*(n.evaluate() for n in keys))
            return Args(key, resolution)
        case _:
            """ key: ... """
            key = Args(lhs.evaluate())
            return Args(key, resolution)


def assign_option(*args):
    match args:
        case fn, pattern, resolution:
            fn.assign_option(pattern, resolution)
        case Args() | ArgsMatcher() as key, resolution:
            if not isinstance(Context.env.fn, Function):
                raise EnvironmentError(f"Line {Context.line}: Cannot assign key-value option in this context.  "
                                       f"Must be within a definition of a function, table, or trait.")
            Context.env.fn.assign_option(key, resolution)
        case _:
            raise RuntimeErr(f"Line {Context.line}: wrong arguments for colon function.")
    return BuiltIns['blank']



Op[':'].eval_args = eval_colon_args
Op[':'].fn = Function({AnyPlusPattern: assign_option})


def eval_dot_args(lhs: Node, rhs: Node) -> Args:
    if not rhs:
        raise NotImplementedError(f"Line {Context.line}: dot functions not implemented yet.")
    if rhs.type == TokenType.Name:
        right_arg = py_value(rhs.source_text)
    else:
        right_arg = rhs.evaluate()  # should evaluate to Args

    match lhs, rhs:
        case (OpExpr(op=Operator(text='.'),
                     terms=(left_term, Token(type=TokenType.Name, source_text=name))),
              ListNode(list_type=ListType.Args) as args_node):
            # case left_term.name[args_node]
            # is name in left_term?
            left = left_term.evaluate()
            prop = left.get(name, None, search_table_frame_too=False)
            if prop is not None:
                # the name resolved a slot in left
                return Args(prop, right_arg, caller=left)
            # else: no slot named "name" in fn
            args = Args(left) + right_arg
            fn = left.table.get(name, Context.deref(name, None))
            if fn is None:
                raise KeyErr(f"Line {Context.line}: {left.table} {left} has no slot '{name}' and no variable with that "
                             f"name found in current scope either.")
            return Args(fn, args)
        case _:
            pass

    return Args(lhs.evaluate(), right_arg)

    # if not TraitMatcher(SeqTrait).match_score(right_arg):
    #     return Args(lhs.evaluate(), right_arg)
    # args = right_arg
    # if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
    #     name = lhs[-1].source_text
    #     a = expressionize(lhs[:-2]).evaluate()
    #     fn = a.get(name, None)
    #     if fn is None:
    #         fn = Context.deref(name, None)
    #         if fn is None:
    #             raise KeyErr(f"Line {Context.line}: {a.table} {a} has no slot '{name}' and no variable with that name "
    #                          f"found in current scope either.")
    #         if isinstance(args, Args):
    #             args.positional_arguments = (a, *args.positional_arguments)
    #         else:
    #             args = List([a] + args.value)
    # else:
    #     fn = expressionize(lhs).evaluate()
    # return [fn, args]


def dot_fn(a: Record, b: Record, *, caller=None):
    match b:
        case Args() as args:
            return a.call(args, caller=caller)
        case PyValue(value=str() as name):
            prop = a.get(name, None)
            if prop is not None:
                return prop
            fn = a.table.get(name, Context.deref(name, None))
            return fn.call(a, caller=caller)
        # case PyValue(value=tuple() | list() as args):
        #     return a.call(*args)
        case _:
            print(f"WARNING: Line {Context.line}: "
                  f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
            return a.call(b)
    # raise OperatorError(f"Line {Context.line}: "
    #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")


def py_dot(a: PyObj, b: PyValue):
    obj = a.obj
    match b.value:
        case str() as name:
            return piliize(getattr(obj, name))
        case list() | tuple() as args:
            return piliize(a.obj(*[arg.value for arg in args]))


caller_patt = ArgsMatcher(AnyParam, AnyParam, named_params={'caller': Parameter(AnyMatcher(), 'caller', '?')})

Op['.'].fn = Function({caller_patt: dot_fn,
                       # ArgsMatcher(AnyParam, AnyParam, caller=AnyParam): temp_new_dot_fn,
                       StringParam: lambda a: Context.deref(a.value),
                       ArgsMatcher(Parameter(TableMatcher(BuiltIns['PythonObject'])),
                                   Parameter(UnionMatcher(TraitMatcher(FuncTrait), TableMatcher(BuiltIns['Table'])))):
                           py_dot,  # I don't remember why the second parameter for the pydot is func|table ???
                       ArgsMatcher(Parameter(TableMatcher(BuiltIns['PythonObject'])), AnyParam): py_dot
                       })

Op['.?'].fn = Function({caller_patt:
                            lambda a, b: BuiltIns['.'].call(a,b)
                                         if BuiltIns['has'].call(a, b).value else py_value(None),
                        ArgsMatcher(StringParam):
                            lambda a: BuiltIns['.'].call(a)
                                      if BuiltIns['has'].call(a).value else py_value(None),
                        })
# map-dot / swizzle operator
Op['..'].fn = Function({ArgsMatcher(SeqParam, StringParam): lambda ls, name: List([dot_fn(el, name) for el in ls.value]),
                       ArgsMatcher(NumericParam, NumericParam): lambda a, b: piliize(range(a.value, b.value))},
                       name='..')
Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = eval_dot_args
BuiltIns['call'] = Op['.'].fn

Op[','].fn = Function({AnyPlusPattern: lambda *args: py_value(args)},
                      name=',')

def eval_nullish_args(*nodes: Node):
    for node in nodes:
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

Op['in'].fn = Function({AnyBinopPattern:
                            lambda a, b: py_value(any(a == opt.value for opt in b.options)),
                        ArgsMatcher(AnyParam, IterParam):
                            lambda a, b: py_value(a in b.value)},
                       name='in')

Op['=='].fn = Function({AnyBinopPattern: lambda a, b: py_value(a == b)},
                       name='==')

Op['!='].fn = Function({AnyBinopPattern: lambda a, b: py_value(not BuiltIns['=='].call(a, b).value)},
                       name='!=')
Op['~'].fn = Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))},
                      name='~')
Op['!~'].fn = Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))},
                       name='!~')
Op['is'].fn = Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))},
                       name='is')
Op['is not'].fn = Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))},
                           name='is not')
Op['|'].fn = Function({AnyBinopPattern: lambda *args: UnionMatcher(*map(patternize, args))},
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
Op['+'].fn = Function({NormalBinopPattern: lambda a, b: py_value(a.value + b.value),
                   ArgsMatcher(AnyParam): lambda a: BuiltIns['num'].call(a),
                   ArgsMatcher(StringParam, StringParam): lambda a, b: py_value(a.value + b.value),
                   ArgsMatcher(ListParam, ListParam): lambda a, b: py_value(a.value + b.value),
                   ArgsMatcher(*(Parameter(TraitMatcher(TupTrait)),) * 2): lambda a, b: py_value(a.value + b.value),
                   ArgsMatcher(SeqParam, SeqParam): lambda a, b: py_value(a.value + b.value)
                    },
                   name='+')
Op['-'].fn = Function({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
                    ArgsMatcher(AnyParam): lambda a: py_value(-BuiltIns['num'].call(a).value)},
                   name='-')
Op['*'].fn = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value * b.value),
                    ArgsMatcher(StringParam, IntegralParam): lambda a, b: py_value(a.value * b.value)},
                   name='*')
Op['/'].fn = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value / b.value),
                    ArgsMatcher(RationalParam, RationalParam): lambda a, b:
                    py_value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))},
                   name='/')
Op['//'].fn = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value // b.value)},
                    name='//')
Op['%'].fn = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)},
                   name='%')
Op['**'].fn = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)},
                    name='**')
Op['^'].fn = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)},
                   name='^')
Op['?'].fn = Function({AnyParam: lambda p: UnionMatcher(patternize(p), ValueMatcher(BuiltIns['blank']))},
                      name='?')
def has_option(fn: Record, arg: Record = None) -> PyValue:
    if arg is None:
        fn, arg = None, fn

    match fn, arg:
        case None, PyValue(value=str() as name):
            return py_value(Context.deref(name, None) is not None)
        case None, _:
            raise TypeErr(f"Line {Context.line}: When used as a prefix, "
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
            raise TypeErr(f"Line {Context.line}: "
                          f"The right-hand term of the `has` operator must be a string or sequence of arguments.")


Op['has'].fn = Function({ArgsMatcher(AnyParam, NonStrSeqParam): has_option,
                      AnyBinopPattern: has_option,
                      ArgsMatcher(StringParam): lambda s: py_value(Context.deref(s, None) is not None),
                      ArgsMatcher(NormalParam): has_option},
                     name='has')
Op['+'].fn = Function({NormalBinopPattern: lambda a, b: py_value(a.value + b.value),
                       ArgsMatcher(AnyParam): lambda a: BuiltIns['num'].call(a),
                       ArgsMatcher(StringParam, StringParam): lambda a, b: py_value(a.value + b.value),
                       ArgsMatcher(ListParam, ListParam): lambda a, b: py_value(a.value + b.value),
                       ArgsMatcher(*(Parameter(TraitMatcher(TupTrait)),) * 2): lambda a, b: py_value(a.value + b.value),
                       # ArgsMatcher(SeqParam, SeqParam): lambda a, b: py_value(a.value + b.value),
                       },
                      name='+')





# if False == "False":
#     # ********** Operators ***********
#     # noinspection PyShadowingNames
#     def eval_set_args(lhs: list[Node], rhs: list[Node]) -> (tuple[PyValue[str], Record]
#                                                             | tuple[Function, List, Record]
#                                                             | tuple[Record, PyValue[str], Record]):
#         rec: Function | Record | None = None
#         name: str | None = None
#         key: PyValue[str | list]
#         match lhs:
#             case [Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#             case [ListNode(nodes=statements)]:
#                 key = List(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
#             case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#                 rec = expressionize(fn_nodes).evaluate()
#             case [*fn_nodes, Token(source_text='.' | '.['), ListNode() as list_node]:
#                 key = expressionize([list_node]).evaluate()
#                 rec = expressionize(fn_nodes).evaluate()
#             case _:
#                 raise SyntaxErr(f'Line {Context.line}: '
#                                 f'Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
#         match rhs:
#             case []:
#                 raise SyntaxErr(f'Line {Context.line}: Missing right-hand side for assignment.')
#             case [Block() as blk]:
#                 value = CodeBlock(blk).execute(fn=Function(name=name))
#             case _:
#                 value = expressionize(rhs).evaluate()
#         if isinstance(getattr(key, 'value', None), str) and isinstance(value, Function) and value.name is None:
#             value.name = key.value
#         if rec is None:
#             return key, value
#         return rec, key, value
#
#     def eval_alias_args(lhs: list[Node], rhs: list[Node]) -> tuple[Record]:
#         left = eval_set_args(lhs, rhs)
#         right = eval_set_args(rhs, rhs)
#         match right[:-1]:
#             case (PyValue(value=str() as name),):
#                 closure = Context.env
#                 # option = Option(ArgsMatcher(), lambda : closure.)
#         right_key = right[-2]
#         if len(right) == 3:
#             right_fn = right[-3]
#             ascend = False
#         else:
#             right_fn = Context.env
#             ascend = True
#         # right_fn = right[-3] if len(right)==3 else Context.env
#         match right_key.value:  # right_key.value
#             case str() as name:
#                 option = right_fn.select_by_name(name, ascend)
#             case list() as args:
#                 option, _ = right_fn.select_and_bind(args, ascend_env=ascend)
#             case _:
#                 raise TypeErr(f"Line {Context.line}: Sorry, I'm not sure how to alias {right_key}")
#         # option = right_fn.select_by_name(right_key.value, ascend)
#         return (*left[:-1], option)
#
#
#     def assign_option(fn: Function, args: PyValue[tuple] | Args, val: Record):
#         params = (Parameter(ValueMatcher(rec)) for rec in args)
#         fn.assign_option(ArgsMatcher(*params), val)
#         return val
#
#
#     def augment_assign_fn(op: str):
#         def aug_assign(key: PyValue[str], val: Record):
#             initial = Context.deref(key.value)
#             new = BuiltIns[op].call(initial, val)
#             # WARNING: this may potentially create a shadow variable
#             return Context.env.assign(key.value, new)
#         return aug_assign
#
#
#     # Operator('=', binop=1, static=True, associativity='right')
#     Operator(';',
#              Function({AnyBinopPattern: lambda x, y: y}),
#              binop=1)
#     def assign_fn(fn: Function, patt: ArgsMatcher, block: CodeBlock, dot_option: PyValue[bool]) -> PyValue:
#         option = fn.select_by_pattern(patt)
#         if option is None:
#             option = fn.assign_option(patt, block)
#         else:
#             option.resolution = block
#         option.dot_option = dot_option.value
#         return py_value(None)
#     Operator(':',
#              Function({ArgsMatcher(AnyParam, PatternParam, AnyParam, BoolParam): assign_fn}),
#              binop=2, associativity='right')
#     def eval_assign_fn_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         blk_nd = rhs[0]
#         if len(rhs) == 1 and isinstance(blk_nd, Block):
#             block: Block = blk_nd
#         else:
#             return_statement = Expression([Token('return')] + rhs)  # noqa
#             block = Block([return_statement])
#         fn, patt, dot_option = read_option(lhs)
#         return [fn, patt, CodeBlock(block), py_value(dot_option)]
#     Op[':'].eval_args = eval_assign_fn_args
#
#     Operator(':=', binop=2, associativity='right')
#     Operator('=',
#              Function({ArgsMatcher(StringParam, AnyParam): lambda name, val: Context.env.assign(name, val),
#                        ArgsMatcher(FunctionParam, NonStrSeqParam, AnyParam): assign_option,
#                        ArgsMatcher(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val)}),
#              binop=2, associativity='right')
#     Op['='].eval_args = eval_set_args
#     Op[':='].fn = Op['='].fn
#     for op in ('+', '-', '*', '/', '//', '**', '%'):
#         Operator(op+'=', Function({AnyBinopPattern: augment_assign_fn(op),
#                                   ArgsMatcher(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(*args)}),
#                  binop=2, associativity='right')
#         Op[op+'='].eval_args = eval_set_args
#     Op[':='].eval_args = eval_alias_args
#
#     def eval_null_assign_args(lhs: list[Node], rhs: list[Node]) -> tuple[Record, ...]:
#         fn = None
#         existing = None
#         rec = None
#         match lhs:
#             case [Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#                 existing = Context.deref(name, None)
#             case [ListNode(nodes=statements)]:
#                 key = List(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
#                 fn = Context.env.caller
#             case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#                 rec = expressionize(fn_nodes).evaluate()
#                 existing = rec.get(name, None)
#             case [*fn_nodes, Token(source_text='.' | '.['), ListNode() as list_node]:
#                 key = expressionize([list_node]).evaluate()
#                 rec = expressionize(fn_nodes).evaluate()
#                 fn = rec
#             case _:
#                 raise SyntaxErr(
#                     f'Line {Context.line}: Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
#         if fn:
#             opt, _ = fn.select(Args(*key.value))
#             if opt and opt.value and opt.value != BuiltIns['blank']:
#                 existing = opt.value
#         if existing is None or existing == BuiltIns['blank']:
#             value = expressionize(rhs).evaluate()
#         else:
#             return existing,
#         if isinstance(getattr(key, 'value', None), str) and isinstance(value, Function) and value.name is None:
#             value.name = key.value
#         if rec is None:
#             return key, value
#         return rec, key, value
#
#     def null_assign(key: PyValue[str], val: Record):
#         initial = Context.deref(key.value, None)
#         if initial is not None:
#             return initial
#         return Context.env.assign(key.value, val)
#
#     def null_assign_option(fn: Function, key: PyValue[list], val: Record):
#         params = (Parameter(ValueMatcher(rec)) for rec in key.value)
#         fn.assign_option(ArgsMatcher(*params), val)
#         return val
#
#
#     Operator('??=',
#              Function({ArgsMatcher(StringParam, AnyParam): lambda name, val: Context.env.assign(name, val),
#                        ArgsMatcher(FunctionParam, NonStrSeqParam, AnyParam): assign_option,
#                        ArgsMatcher(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val),
#                        ArgsMatcher(AnyParam): lambda x: x}),
#              binop=2, associativity='right')
#     Op['??='].eval_args = eval_set_args
#     Operator('=>',
#              Function({ArgsMatcher(AnyParam, FunctionParam): lambda x, fn: BuiltIns['call'](Args(fn, x))}),
#              binop=2)
#     Operator(',',
#              Function({ArgsMatcher(Parameter(AnyMatcher(), quantifier='+')): lambda *args: py_value(tuple(args)),
#                       AnyParam: lambda x: py_value((x,))}),
#              binop=2, postfix=2, associativity='right')
#     def eval_tuple_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         left = expressionize(lhs).evaluate()
#         if not rhs:
#             return [left]
#         right_expr = expressionize(rhs)
#         if getattr(right_expr, 'op', None) == Op[',']:
#             return [left, *eval_tuple_args(right_expr.lhs, right_expr.rhs)]
#         return [left, right_expr.evaluate()]
#     Op[','].eval_args = eval_tuple_args
#     def eval_if_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         for i in reversed(range(len(rhs))):
#             if rhs[i].source_text == 'else':
#                 condition = expressionize(rhs[:i])
#                 rhs = rhs[i + 1:]
#                 condition = condition.evaluate()
#                 break
#         else:
#             raise SyntaxErr(f"Line {Context.line}: If statement with no else clause")
#         if BuiltIns['bool'].call(condition).value:
#             return [expressionize(lhs).evaluate(), py_value(True), py_value(None)]
#         else:
#             return [py_value(None), py_value(False), expressionize(rhs).evaluate()]
#     Operator('if',
#              Function({ArgsMatcher(AnyParam, AnyParam, AnyParam):
#                       lambda consequent, condition, alt: consequent if condition.value else alt}),
#              binop=3, ternary='else')
#     Op['if'].eval_args = eval_if_args
#
#     def nullish_or(*args: Record):
#         for arg in args[:-1]:
#             if arg != BuiltIns['blank']:
#                 return arg
#         return args[-1]
#
#
#     Operator('??',
#              Function({ArgsMatcher(Parameter(AnyMatcher(), "", "+")): nullish_or}),
#              binop=4)
#     def eval_nullish_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         first = expressionize(lhs).evaluate()
#         if first != BuiltIns['blank']:
#             return [first]
#         return [expressionize(rhs).evaluate()]
#     Op['??'].eval_args = eval_nullish_args
#
#     def or_fn(*args: Record) -> Record:
#         i = 0
#         for i in range(len(args)-1):
#             if BuiltIns['bool'].call(args[i]).value:
#                 return args[i]
#         return args[i]
#     Operator('or',
#              Function({AnyPlusPattern: or_fn}),
#              binop=5)
#     def eval_or_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         condition = expressionize(lhs).evaluate()
#         return [condition] if BuiltIns['bool'].call(condition).value else [expressionize(rhs).evaluate()]
#     Op['or'].eval_args = eval_or_args
#
#     def and_fn(*args: Record) -> Record:
#         i = 0
#         for i in range(len(args)-1):
#             if not BuiltIns['bool'].call(args[i]).value:
#                 return args[i]
#         return args[i]
#     Operator('and',
#              Function({AnyPlusPattern: and_fn}),
#              binop=6)
#     def eval_and_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         condition = expressionize(lhs).evaluate()
#         return [condition] if not BuiltIns['bool'].call(condition).value else [expressionize(rhs).evaluate()]
#     Op['and'].eval_args = eval_and_args
#
#     Operator('not',
#              Function({ArgsMatcher(AnyParam): lambda a: py_value(not BuiltIns['bool'].call(a).value)}),
#              prefix=7)
#     Operator('in',
#              Function({AnyBinopPattern: lambda a, b: py_value(a in (opt.value for opt in b.options if hasattr(opt, 'value'))),
#                        ArgsMatcher(AnyParam, IterParam): lambda a, b: py_value(a in b.value)}),
#              binop=8)
#     Operator('==',
#              Function({AnyBinopPattern: lambda a, b: py_value(a == b)}),
#              binop=9)
#     Operator('!=',
#              Function({AnyBinopPattern: lambda a, b: py_value(not BuiltIns['=='].call(a, b).value)}),
#              binop=9)
#     Operator('~',
#              Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))}),
#              binop=9, chainable=False)
#     Operator('!~',
#              Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))}),
#              binop=9, chainable=False)
#     Operator('is',
#              Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))}),
#              binop=9, chainable=False)
#     Operator('is not',
#              Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))}),
#              binop=9, chainable=False)
#     # def union_patterns(*values_or_patterns: Record):
#     #     patts = map(patternize, values_or_patterns)
#     #     params = (param for patt in patts for param in patt.try_get_params())
#     #     try:
#     #         matchers = (m for p in params for m in p.try_get_matchers())
#     #         return ArgsMatcher(Parameter(Union(*matchers)))
#     #     except TypeError:
#     #         return ArgsMatcher(UnionParam(*params))
#     def union_patterns(*values_or_patterns: Record):
#         patterns = map(patternize, values_or_patterns)
#         params = []
#         for patt in patterns:
#             if len(patt.parameters) != 1:
#                 raise TypeErr(f"Line {Context.line}: Cannot get union of patterns with multiple parameters.")
#             param = patt.parameters[0]
#             if isinstance(param, UnionParam):
#                 params.extend(param.parameters)
#             else:
#                 params.append(param)
#         matchers = []
#         for param in params:
#             if param.quantifier or param.name is not None:
#                 return ArgsMatcher(UnionParam(*params))
#             if isinstance(param.matcher, UnionMatcher):
#                 matchers.extend(param.matcher.matchers)
#             else:
#                 matchers.append(param.matcher)
#         return ArgsMatcher(Parameter(UnionMatcher(*matchers)))
#
#
#     Operator('|',
#              Function({AnyBinopPattern: union_patterns}),
#              binop=10, chainable=True)
#     Operator('<',
#              Function({NormalBinopPattern: lambda a, b: py_value(a.value < b.value)}),
#              binop=11, chainable=True)
#     Operator('>',
#              Function({NormalBinopPattern: lambda a, b: py_value(a.value > b.value)}),
#              binop=11, chainable=True)
#     Operator('<=',
#              Function({AnyBinopPattern:
#                       lambda a, b: py_value(BuiltIns['<'].call(a, b).value or BuiltIns['=='].call(a, b).value)}),
#              binop=11, chainable=True)
#     Operator('>=',
#              Function({AnyBinopPattern:
#                       lambda a, b: py_value(BuiltIns['>'].call(a, b).value or BuiltIns['=='].call(a, b).value)}),
#              binop=11, chainable=True)
#     Operator('+',
#              Function({NormalBinopPattern: lambda a, b: py_value(a.value + b.value),
#                        ArgsMatcher(AnyParam): lambda a: BuiltIns['num'].call(a),
#                        ArgsMatcher(StringParam, StringParam): lambda a, b: py_value(a.value + b.value),
#                        ArgsMatcher(ListParam, ListParam): lambda a, b: py_value(a.value + b.value),
#                        ArgsMatcher(*(Parameter(TraitMatcher(TupTrait)),) * 2): lambda a, b: py_value(a.value + b.value),
#                        # ArgsMatcher(SeqParam, SeqParam): lambda a, b: py_value(a.value + b.value)
#                        }),
#              binop=12, prefix=14)
#     Operator('-',
#              Function({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
#                       ArgsMatcher(AnyParam): lambda a: py_value(-BuiltIns['num'].call(a).value)}),
#              binop=12, chainable=False, prefix=14)
#     Operator('*',
#              Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value * b.value),
#                       ArgsMatcher(StringParam, IntegralParam): lambda a, b: py_value(a.value * b.value)}),
#              binop=13)
#     Operator('/',
#              Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value / b.value),
#                       ArgsMatcher(RationalParam, RationalParam): lambda a, b:
#                       py_value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))}),
#              binop=13, chainable=False)
#     Operator('//',
#              Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value // b.value)}),
#              binop=13, chainable=False)
#     Operator('%',
#              Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)}),
#              binop=13, chainable=False)
#     Operator('**',
#              Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)}),
#              binop=14, chainable=False, associativity='right')
#     Operator('^',
#              Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)}),
#              binop=14, chainable=False, associativity='right')
#     Operator('?',
#              Function({AnyParam: lambda p: union_patterns(p, BuiltIns['blank'])}),
#              postfix=15, static=True)
#     def has_option(fn: Record, arg: Record = None) -> PyValue:
#         if arg is None:
#             fn, arg = None, fn
#
#         match fn, arg:
#             case None, PyValue(value=str() as name):
#                 return py_value(Context.deref(name, None) is not None)
#             case None, _:
#                 raise TypeErr(f"Line {Context.line}: When used as a prefix, "
#                               f"the right-hand term of the `has` operator must be a string, not {arg.table}")
#             case Record(), PyValue(value=str() as name):
#                 return py_value(fn.get(name, None) is not None)
#             # case Record(), List(records=args) | PyValue(value=tuple() as args):
#             case Record(), Args() as args:
#                 option, _ = fn.select(args)
#                 return py_value(option is not None)
#             case Record(), PyValue(value=tuple() | list() as args):
#                 args = Args(*args)
#                 option, _ = fn.select(args)
#                 return py_value(option is not None)
#             case _:
#                 raise TypeErr(f"Line {Context.line}: "
#                               f"The right-hand term of the `has` operator must be a string or sequence of arguments.")
#     Operator('has',
#              Function({ArgsMatcher(AnyParam, NonStrSeqParam): has_option,
#                        AnyBinopPattern: has_option,
#                        ArgsMatcher(NormalParam): has_option}),
#              binop=15, prefix=15)
#     def add_guard_fn(fn: Record, guard: Function):
#         patt = patternize(fn)
#         patt.guard = guard
#         return patt
#     def add_guard_expr(fn: Record, expr: Expression):
#         patt = patternize(fn)
#         patt.exprs.append(expr)  # noqa
#         return patt
#     Operator('&',
#              Function({ArgsMatcher(AnyParam, AnyParam): add_guard_expr,
#                        ArgsMatcher(FunctionParam, AnyParam): add_guard_expr}),
#              binop=15)
#     # def eval_patt_guard_args(lhs: list[Node], rhs: list[Node]) -> [Record, Expression]:
#     #     return [expressionize(lhs).evaluate(), expressionize(rhs)]
#     Op['&'].eval_args = lambda lhs, rhs: [expressionize(lhs).evaluate(), expressionize(rhs)]
#     Operator('@',
#              Function(),
#              prefix=16)
#
#     def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         if len(rhs) != 1:
#             raise SyntaxErr(f'Line {Context.line}: missing args')
#         if rhs[0].type == TokenType.Name:
#             right_arg = py_value(rhs[0].source_text)
#         else:
#             right_arg = expressionize(rhs).evaluate()
#         if not lhs:
#             return [right_arg]
#         if not TraitMatcher(SeqTrait).match_score(right_arg):
#             return [expressionize(lhs).evaluate(), right_arg]
#         args = right_arg
#         if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
#             name = lhs[-1].source_text
#             a = expressionize(lhs[:-2]).evaluate()
#             fn = a.get(name, None)
#             if fn is None:
#                 fn = Context.deref(name, None)
#                 if fn is None:
#                     raise KeyErr(f"Line {Context.line}: {a.table} {a} has no slot '{name}' and no variable with that name "
#                                  f"found in current scope either.")
#                 if isinstance(args, Args):
#                     args.positional_arguments = (a, *args.positional_arguments)
#                 else:
#                     args = List([a] + args.value)
#         else:
#             fn = expressionize(lhs).evaluate()
#         return [fn, args]
#     def dot_fn(a: Record, b: Record):
#         match b:
#             case Args() as args:
#                 return a(args)
#             case PyValue(value=str() as name):
#                 try:
#                     # return a.deref(name, ascend_env=False)
#                     return a.get(name)
#                 except SlotErr as e:
#                     pass
#                 fn = Context.deref(name)
#                 # if not fn.instanceof(BuiltIns['func']):
#                 #     raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
#                 # assert isinstance(fn.value, Record)
#                 return fn.call(a)
#             # case List(records=args) | PyValue(value=tuple() as args):
#             case PyValue(value=tuple() | list() as args):
#                 return a.call(*args)
#             case _:
#                 print(f"WARNING: Line {Context.line}: "
#                       f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
#                 return a.call(b)
#         # raise OperatorError(f"Line {Context.line}: "
#         #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
#
#     def py_dot(a: PyObj, b: PyValue):
#         obj = a.obj
#         match b.value:
#             case str() as name:
#                 return piliize(getattr(obj, name))
#             case list() | tuple() as args:
#                 return piliize(a.obj(*[arg.value for arg in args]))
#
#
#     Operator('.',
#              Function({ArgsMatcher(AnyParam, NonStrSeqParam): dot_fn,
#                        AnyBinopPattern: dot_fn,
#                        StringParam: lambda a: Context.deref(a.value),
#                        ArgsMatcher(Parameter(TableMatcher(BuiltIns['PythonObject'])),
#                                Parameter(UnionMatcher(TraitMatcher(FuncTrait), TableMatcher(BuiltIns['Table'])))):
#                            py_dot}),
#              binop=16, prefix=16)
#     BuiltIns['call'] = BuiltIns['.']
#     Operator('.?',
#              Function({AnyBinopPattern: lambda a, b: BuiltIns['.'].call(a,b) if BuiltIns['has'].call(a, b).value else py_value(None),
#                       ArgsMatcher(StringParam): lambda a: BuiltIns['.'].call(a) if BuiltIns['has'].call(a).value else py_value(None),}),
#              binop=16, prefix=16)
#     # map-dot / swizzle operator
#     Operator('..',
#              Function({ArgsMatcher(SeqParam, StringParam): lambda ls, name: List([dot_fn(el, name) for el in ls.value]),
#                        ArgsMatcher(NumericParam, NumericParam): lambda a, b: piliize(range(a.value, b.value))}),
#              binop=16, prefix=16)
#     Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = eval_call_args
#
#
#     # pattern generator options for int, str, float, etc
#     def make_lambda_guard(type_name: str):
#         if type_name == 'str':
#             return lambda a, b: ArgsMatcher(Parameter(TableMatcher(BuiltIns[type_name], guard=lambda x: py_value(a.value <= len(x.value) <= b.value))))
#         else:
#             return lambda a, b: ArgsMatcher(Parameter(TableMatcher(BuiltIns[type_name], guard=lambda x: py_value(a.value <= x.value <= b.value))))
#
#
#     for type_name in ('num', 'ratio', 'float', 'int', 'str'):
#         if type_name == 'int':
#             pass
#         pass # BuiltIns[type_name].add_option(ArgsMatcher(NumericParam, NumericParam), make_lambda_guard(type_name))
#
#
#     # BuiltIns['str'].add_option(ArgsMatcher(StringParam),
#     #                            lambda regex: ArgsMatcher(Parameter(TraitMatcher(
#     #                                BuiltIns['str'], guard=lambda s: py_value(bool(re.fullmatch(regex.value, s.value)))))))
#     # BuiltIns['num'].add_option(ArgsMatcher(NumericParam, NumericParam), lambda a, b: py_value(TableMatcher(BuiltIns['num'], guard=lambda x: py_value(a.value <= x.value <= b.value))))
#
#     # Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
#     def number_guard(op_sym: str):
#         # assert a.value == b.type
#         return lambda t, n: ArgsMatcher(Parameter(TraitMatcher(t, guard=lambda x: Op[op_sym].fn.call(x, n))))
#
#     # generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
#     def string_guard(op_sym: str):
#         # assert a.value == BuiltIns['str'] and b.type in (BuiltIns['int'], BuiltIns['float'])
#         return lambda t, n: ArgsMatcher(Parameter(TraitMatcher(t, guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), n))))
#         # def guard(x, y):
#         #     return ArgsMatcher(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), b))))
#         # # return guard
#         # return ArgsMatcher(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), b))))
#
#     # def add_guards(op_sym: str):
#     #     Op[op_sym].fn.add_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['int'])), NumericParam),
#     #                                 number_guard(op_sym))
#     #     Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['float'])), NumericParam),
#     #                                 lambda a, b: number_guard(a, b, op_sym))
#     #     Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['ratio'])), NumericParam),
#     #                                 lambda a, b: number_guard(a, b, op_sym))
#     #     Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
#     #                                 lambda a, b: string_guard(a, b, op_sym))
#
#
#     for op_sym in ('>', '<', '>=', '<='):
#         for type_name in ('int', 'ratio', 'float', 'num'):
#             Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[type_name])), NumericParam),
#                                         number_guard(op_sym))
#         Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
#                                     string_guard(op_sym))
#         # add_guards(op)
#
#
#     # noinspection PyShadowingNames
#     def eval_set_args(lhs: list[Node], rhs: list[Node]) -> (tuple[PyValue[str], Record]
#                                                             | tuple[Function, List, Record]
#                                                             | tuple[Record, PyValue[str], Record]):
#         rec: Function | Record | None = None
#         name: str | None = None
#         key: PyValue[str | list]
#         match lhs:
#             case [Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#             case [ListNode(nodes=statements)]:
#                 key = List(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
#             case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#                 rec = expressionize(fn_nodes).evaluate()
#             case [*fn_nodes, Token(source_text='.' | '.['), ListNode() as list_node]:
#                 key = expressionize([list_node]).evaluate()
#                 rec = expressionize(fn_nodes).evaluate()
#             case _:
#                 raise SyntaxErr(f'Line {Context.line}: '
#                                 f'Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
#         match rhs:
#             case []:
#                 raise SyntaxErr(f'Line {Context.line}: Missing right-hand side for assignment.')
#             case [Block() as blk]:
#                 value = CodeBlock(blk).execute(fn=Function(name=name))
#             case _:
#                 value = expressionize(rhs).evaluate()
#         if isinstance(getattr(key, 'value', None), str) and isinstance(value, Function) and value.name is None:
#             value.name = key.value
#         if rec is None:
#             return key, value
#         return rec, key, value
#
#     def eval_alias_args(lhs: list[Node], rhs: list[Node]) -> tuple[Record]:
#         left = eval_set_args(lhs, rhs)
#         right = eval_set_args(rhs, rhs)
#         match right[:-1]:
#             case (PyValue(value=str() as name),):
#                 closure = Context.env
#                 # option = Option(ArgsMatcher(), lambda : closure.)
#         right_key = right[-2]
#         if len(right) == 3:
#             right_fn = right[-3]
#             ascend = False
#         else:
#             right_fn = Context.env
#             ascend = True
#         # right_fn = right[-3] if len(right)==3 else Context.env
#         match right_key.value:  # right_key.value
#             case str() as name:
#                 option = right_fn.select_by_name(name, ascend)
#             case list() as args:
#                 option, _ = right_fn.select_and_bind(args, ascend_env=ascend)
#             case _:
#                 raise TypeErr(f"Line {Context.line}: Sorry, I'm not sure how to alias {right_key}")
#         # option = right_fn.select_by_name(right_key.value, ascend)
#         return (*left[:-1], option)
#
#
#     def assign_option(fn: Function, args: PyValue[tuple] | Args, val: Record):
#         params = (Parameter(ValueMatcher(rec)) for rec in args)
#         fn.assign_option(ArgsMatcher(*params), val)
#         return val
#
#
#     def augment_assign_fn(op: str):
#         def aug_assign(key: PyValue[str], val: Record):
#             initial = Context.deref(key.value)
#             new = BuiltIns[op].call(initial, val)
#             # WARNING: this may potentially create a shadow variable
#             return Context.env.assign(key.value, new)
#         return aug_assign
#
#
#     # Operator('=', binop=1, static=True, associativity='right')
#     Op[';'] = Function({AnyBinopPattern: lambda x, y: y})
#
#     def assign_fn(fn: Function, patt: ArgsMatcher, block: CodeBlock, dot_option: PyValue[bool]) -> PyValue:
#         option = fn.select_by_pattern(patt)
#         if option is None:
#             option = fn.assign_option(patt, block)
#         else:
#             option.resolution = block
#         option.dot_option = dot_option.value
#         return py_value(None)
#
#
#     Op[':'] = Function({ArgsMatcher(AnyParam, PatternParam, AnyParam, BoolParam): assign_fn})
#
#     def eval_assign_fn_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         blk_nd = rhs[0]
#         if len(rhs) == 1 and isinstance(blk_nd, Block):
#             block: Block = blk_nd
#         else:
#             return_statement = Expression([Token('return')] + rhs)  # noqa
#             block = Block([return_statement])
#         fn, patt, dot_option = read_option(lhs)
#         return [fn, patt, CodeBlock(block), py_value(dot_option)]
#     Op[':'].eval_args = eval_assign_fn_args
#
#     Operator(':=', binop=2, associativity='right')
#     Op['='] = Function({ArgsMatcher(StringParam, AnyParam): lambda name, val: Context.env.assign(name, val),
#                        ArgsMatcher(FunctionParam, NonStrSeqParam, AnyParam): assign_option,
#                        ArgsMatcher(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val)})
#     Op['='].eval_args = eval_set_args
#     Op[':='].fn = Op['='].fn
#     for op in ('+', '-', '*', '/', '//', '**', '%'):
#         Op[op+'='] = Function({AnyBinopPattern: augment_assign_fn(op),
#                                   ArgsMatcher(AnyParam, AnyParam, AnyParam): lambda *args: BuiltIns['set'].call(*args)})
#         Op[op+'='].eval_args = eval_set_args
#     Op[':='].eval_args = eval_alias_args
#
#     def eval_null_assign_args(lhs: list[Node], rhs: list[Node]) -> tuple[Record, ...]:
#         fn = None
#         existing = None
#         rec = None
#         match lhs:
#             case [Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#                 existing = Context.deref(name, None)
#             case [ListNode(nodes=statements)]:
#                 key = List(list(map(lambda s: expressionize(s.nodes).evaluate(), statements)))
#                 fn = Context.env.caller
#             case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
#                 key = py_value(name)
#                 rec = expressionize(fn_nodes).evaluate()
#                 existing = rec.get(name, None)
#             case [*fn_nodes, Token(source_text='.' | '.['), ListNode() as list_node]:
#                 key = expressionize([list_node]).evaluate()
#                 rec = expressionize(fn_nodes).evaluate()
#                 fn = rec
#             case _:
#                 raise SyntaxErr(
#                     f'Line {Context.line}: Invalid left-hand-side for = assignment: {" ".join(n.source_text for n in lhs)}')
#         if fn:
#             opt, _ = fn.select(Args(*key.value))
#             if opt and opt.value and opt.value != BuiltIns['blank']:
#                 existing = opt.value
#         if existing is None or existing == BuiltIns['blank']:
#             value = expressionize(rhs).evaluate()
#         else:
#             return existing,
#         if isinstance(getattr(key, 'value', None), str) and isinstance(value, Function) and value.name is None:
#             value.name = key.value
#         if rec is None:
#             return key, value
#         return rec, key, value
#
#     def null_assign(key: PyValue[str], val: Record):
#         initial = Context.deref(key.value, None)
#         if initial is not None:
#             return initial
#         return Context.env.assign(key.value, val)
#
#     def null_assign_option(fn: Function, key: PyValue[list], val: Record):
#         params = (Parameter(ValueMatcher(rec)) for rec in key.value)
#         fn.assign_option(ArgsMatcher(*params), val)
#         return val
#
#
#     Op['??='] = Function({ArgsMatcher(StringParam, AnyParam): lambda name, val: Context.env.assign(name, val),
#                        ArgsMatcher(FunctionParam, NonStrSeqParam, AnyParam): assign_option,
#                        ArgsMatcher(AnyParam, StringParam, AnyParam): lambda rec, name, val: rec.set(name.value, val),
#                        ArgsMatcher(AnyParam): lambda x: x})
#     Op['??='].eval_args = eval_set_args
#     Op['=>'] = Function({ArgsMatcher(AnyParam, FunctionParam): lambda x, fn: BuiltIns['call'](Args(fn, x))})
#     Op[','] = Function({ArgsMatcher(Parameter(AnyMatcher(), quantifier='+')): lambda *args: py_value(tuple(args)),
#                       AnyParam: lambda x: py_value((x,))})
#     def eval_tuple_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         left = expressionize(lhs).evaluate()
#         if not rhs:
#             return [left]
#         right_expr = expressionize(rhs)
#         if getattr(right_expr, 'op', None) == Op[',']:
#             return [left, *eval_tuple_args(right_expr.lhs, right_expr.rhs)]
#         return [left, right_expr.evaluate()]
#     Op[','].eval_args = eval_tuple_args
#     def eval_if_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         for i in reversed(range(len(rhs))):
#             if rhs[i].source_text == 'else':
#                 condition = expressionize(rhs[:i])
#                 rhs = rhs[i + 1:]
#                 condition = condition.evaluate()
#                 break
#         else:
#             raise SyntaxErr(f"Line {Context.line}: If statement with no else clause")
#         if BuiltIns['bool'].call(condition).value:
#             return [expressionize(lhs).evaluate(), py_value(True), py_value(None)]
#         else:
#             return [py_value(None), py_value(False), expressionize(rhs).evaluate()]
#     Op['if'] = Function({ArgsMatcher(AnyParam, AnyParam, AnyParam):
#                       lambda consequent, condition, alt: consequent if condition.value else alt})
#     Op['if'].eval_args = eval_if_args
#
#     def nullish_or(*args: Record):
#         for arg in args[:-1]:
#             if arg != BuiltIns['blank']:
#                 return arg
#         return args[-1]
#
#
#     Op['??'] = Function({ArgsMatcher(Parameter(AnyMatcher(), "", "+")): nullish_or})
#     def eval_nullish_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         first = expressionize(lhs).evaluate()
#         if first != BuiltIns['blank']:
#             return [first]
#         return [expressionize(rhs).evaluate()]
#     Op['??'].eval_args = eval_nullish_args
#
#     def or_fn(*args: Record) -> Record:
#         i = 0
#         for i in range(len(args)-1):
#             if BuiltIns['bool'].call(args[i]).value:
#                 return args[i]
#         return args[i]
#     Op['or'] = Function({AnyPlusPattern: or_fn})
#     def eval_or_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         condition = expressionize(lhs).evaluate()
#         return [condition] if BuiltIns['bool'].call(condition).value else [expressionize(rhs).evaluate()]
#     Op['or'].eval_args = eval_or_args
#
#     def and_fn(*args: Record) -> Record:
#         i = 0
#         for i in range(len(args)-1):
#             if not BuiltIns['bool'].call(args[i]).value:
#                 return args[i]
#         return args[i]
#     Op['and'] = Function({AnyPlusPattern: and_fn})
#     def eval_and_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         condition = expressionize(lhs).evaluate()
#         return [condition] if not BuiltIns['bool'].call(condition).value else [expressionize(rhs).evaluate()]
#     Op['and'].eval_args = eval_and_args
#
#     Op['not'] = Function({ArgsMatcher(AnyParam): lambda a: py_value(not BuiltIns['bool'].call(a).value)})
#     Op['in'] = Function({AnyBinopPattern: lambda a, b: py_value(a in (opt.value for opt in b.options if hasattr(opt, 'value'))),
#                        ArgsMatcher(AnyParam, IterParam): lambda a, b: py_value(a in b.value)})
#     Op['=='] = Function({AnyBinopPattern: lambda a, b: py_value(a == b)})
#     Op['!='] = Function({AnyBinopPattern: lambda a, b: py_value(not BuiltIns['=='].call(a, b).value)})
#     Op['~'] = Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))})
#     Op['!~'] = Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))})
#     Op['is'] = Function({AnyBinopPattern: lambda a, b: py_value(bool(patternize(b).match_score(a)))})
#     Op['is not'] = Function({AnyBinopPattern: lambda a, b: py_value(not patternize(b).match_score(a))})
#     # def union_patterns(*values_or_patterns: Record):
#     #     patts = map(patternize, values_or_patterns)
#     #     params = (param for patt in patts for param in patt.try_get_params())
#     #     try:
#     #         matchers = (m for p in params for m in p.try_get_matchers())
#     #         return ArgsMatcher(Parameter(Union(*matchers)))
#     #     except TypeError:
#     #         return ArgsMatcher(UnionParam(*params))
#     def union_patterns(*values_or_patterns: Record):
#         patterns = map(patternize, values_or_patterns)
#         params = []
#         for patt in patterns:
#             if len(patt.parameters) != 1:
#                 raise TypeErr(f"Line {Context.line}: Cannot get union of patterns with multiple parameters.")
#             param = patt.parameters[0]
#             if isinstance(param, UnionParam):
#                 params.extend(param.parameters)
#             else:
#                 params.append(param)
#         matchers = []
#         for param in params:
#             if param.quantifier or param.name is not None:
#                 return ArgsMatcher(UnionParam(*params))
#             if isinstance(param.matcher, UnionMatcher):
#                 matchers.extend(param.matcher.matchers)
#             else:
#                 matchers.append(param.matcher)
#         return ArgsMatcher(Parameter(UnionMatcher(*matchers)))
#
#
#     Op['|'] = Function({AnyBinopPattern: union_patterns})
#     Op['<'] = Function({NormalBinopPattern: lambda a, b: py_value(a.value < b.value)})
#     Op['>'] = Function({NormalBinopPattern: lambda a, b: py_value(a.value > b.value)})
#     Op['<='] = Function({AnyBinopPattern:
#                       lambda a, b: py_value(BuiltIns['<'].call(a, b).value or BuiltIns['=='].call(a, b).value)})
#     Op['>='] = Function({AnyBinopPattern:
#                       lambda a, b: py_value(BuiltIns['>'].call(a, b).value or BuiltIns['=='].call(a, b).value)})
#     Op['+'] = Function({NormalBinopPattern: lambda a, b: py_value(a.value + b.value),
#                        ArgsMatcher(AnyParam): lambda a: BuiltIns['num'].call(a),
#                        ArgsMatcher(StringParam, StringParam): lambda a, b: py_value(a.value + b.value),
#                        ArgsMatcher(ListParam, ListParam): lambda a, b: py_value(a.value + b.value),
#                        ArgsMatcher(*(Parameter(TraitMatcher(TupTrait)),) * 2): lambda a, b: py_value(a.value + b.value),
#                        # ArgsMatcher(SeqParam, SeqParam): lambda a, b: py_value(a.value + b.value)
#                        })
#     Op['-'] = Function({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
#                       ArgsMatcher(AnyParam): lambda a: py_value(-BuiltIns['num'].call(a).value)})
#     Op['*'] = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value * b.value),
#                       ArgsMatcher(StringParam, IntegralParam): lambda a, b: py_value(a.value * b.value)})
#     Op['/'] = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value / b.value),
#                       ArgsMatcher(RationalParam, RationalParam): lambda a, b:
#                       py_value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator))})
#     Op['//'] = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value // b.value)})
#     Op['%'] = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)})
#     Op['**'] = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)})
#     Op['^'] = Function({ArgsMatcher(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)})
#     Op['?'] = Function({AnyParam: lambda p: union_patterns(p, BuiltIns['blank'])})
#     def has_option(fn: Record, arg: Record = None) -> PyValue:
#         if arg is None:
#             fn, arg = None, fn
#
#         match fn, arg:
#             case None, PyValue(value=str() as name):
#                 return py_value(Context.deref(name, None) is not None)
#             case None, _:
#                 raise TypeErr(f"Line {Context.line}: When used as a prefix, "
#                               f"the right-hand term of the `has` operator must be a string, not {arg.table}")
#             case Record(), PyValue(value=str() as name):
#                 return py_value(fn.get(name, None) is not None)
#             # case Record(), List(records=args) | PyValue(value=tuple() as args):
#             case Record(), Args() as args:
#                 option, _ = fn.select(args)
#                 return py_value(option is not None)
#             case Record(), PyValue(value=tuple() | list() as args):
#                 args = Args(*args)
#                 option, _ = fn.select(args)
#                 return py_value(option is not None)
#             case _:
#                 raise TypeErr(f"Line {Context.line}: "
#                               f"The right-hand term of the `has` operator must be a string or sequence of arguments.")
#     Op['has'] = Function({ArgsMatcher(AnyParam, NonStrSeqParam): has_option,
#                        AnyBinopPattern: has_option,
#                        ArgsMatcher(NormalParam): has_option})
#     def add_guard_fn(fn: Record, guard: Function):
#         patt = patternize(fn)
#         patt.guard = guard
#         return patt
#     def add_guard_expr(fn: Record, expr: Expression):
#         patt = patternize(fn)
#         patt.exprs.append(expr)  # noqa
#         return patt
#     Op['&'] = Function({ArgsMatcher(AnyParam, AnyParam): add_guard_expr,
#                        ArgsMatcher(FunctionParam, AnyParam): add_guard_expr})
#     # def eval_patt_guard_args(lhs: list[Node], rhs: list[Node]) -> [Record, Expression]:
#     #     return [expressionize(lhs).evaluate(), expressionize(rhs)]
#     Op['&'].eval_args = lambda lhs, rhs: [expressionize(lhs).evaluate(), expressionize(rhs)]
#     Op['@'] = Function()
#
#     def eval_call_args(lhs: list[Node], rhs: list[Node]) -> list[Record]:
#         if len(rhs) != 1:
#             raise SyntaxErr(f'Line {Context.line}: missing args')
#         if rhs[0].type == TokenType.Name:
#             right_arg = py_value(rhs[0].source_text)
#         else:
#             right_arg = expressionize(rhs).evaluate()
#         if not lhs:
#             return [right_arg]
#         if not TraitMatcher(SeqTrait).match_score(right_arg):
#             return [expressionize(lhs).evaluate(), right_arg]
#         args = right_arg
#         if len(lhs) > 2 and lhs[-1].type == TokenType.Name and lhs[-2].source_text in ('.', '..', '.?'):
#             name = lhs[-1].source_text
#             a = expressionize(lhs[:-2]).evaluate()
#             fn = a.get(name, None)
#             if fn is None:
#                 fn = Context.deref(name, None)
#                 if fn is None:
#                     raise KeyErr(f"Line {Context.line}: {a.table} {a} has no slot '{name}' and no variable with that name "
#                                  f"found in current scope either.")
#                 if isinstance(args, Args):
#                     args.positional_arguments = (a, *args.positional_arguments)
#                 else:
#                     args = List([a] + args.value)
#         else:
#             fn = expressionize(lhs).evaluate()
#         return [fn, args]
#     def dot_fn(a: Record, b: Record):
#         match b:
#             case Args() as args:
#                 return a(args)
#             case PyValue(value=str() as name):
#                 try:
#                     # return a.deref(name, ascend_env=False)
#                     return a.get(name)
#                 except SlotErr as e:
#                     pass
#                 fn = Context.deref(name)
#                 # if not fn.instanceof(BuiltIns['func']):
#                 #     raise OperatorError(f"Line {Context.line}: '{name}' is not an option or function.")
#                 # assert isinstance(fn.value, Record)
#                 return fn.call(a)
#             # case List(records=args) | PyValue(value=tuple() as args):
#             case PyValue(value=tuple() | list() as args):
#                 return a.call(*args)
#             case _:
#                 print(f"WARNING: Line {Context.line}: "
#                       f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
#                 return a.call(b)
#         # raise OperatorError(f"Line {Context.line}: "
#         #                     f"right-hand operand of dot operator should be string or list of arguments.  Found {b}.")
#
#     def py_dot(a: PyObj, b: PyValue):
#         obj = a.obj
#         match b.value:
#             case str() as name:
#                 return piliize(getattr(obj, name))
#             case list() | tuple() as args:
#                 return piliize(a.obj(*[arg.value for arg in args]))
#
#
#     Op['.'] = Function({ArgsMatcher(AnyParam, NonStrSeqParam): dot_fn,
#                        AnyBinopPattern: dot_fn,
#                        StringParam: lambda a: Context.deref(a.value),
#                        ArgsMatcher(Parameter(TableMatcher(BuiltIns['PythonObject'])),
#                                Parameter(UnionMatcher(TraitMatcher(FuncTrait), TableMatcher(BuiltIns['Table'])))):
#                            py_dot})
#     BuiltIns['call'] = BuiltIns['.']
#     Op['.?'] = Function({AnyBinopPattern: lambda a, b: BuiltIns['.'].call(a,b) if BuiltIns['has'].call(a, b).value else py_value(None),
#                       ArgsMatcher(StringParam): lambda a: BuiltIns['.'].call(a) if BuiltIns['has'].call(a).value else py_value(None),})
#     # map-dot / swizzle operator
#     Op['..'] = Function({ArgsMatcher(SeqParam, StringParam): lambda ls, name: List([dot_fn(el, name) for el in ls.value]),
#                        ArgsMatcher(NumericParam, NumericParam): lambda a, b: piliize(range(a.value, b.value))})
#     Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = eval_call_args
#
#
#     # pattern generator options for int, str, float, etc
#     def make_lambda_guard(type_name: str):
#         if type_name == 'str':
#             return lambda a, b: ArgsMatcher(Parameter(TableMatcher(BuiltIns[type_name], guard=lambda x: py_value(a.value <= len(x.value) <= b.value))))
#         else:
#             return lambda a, b: ArgsMatcher(Parameter(TableMatcher(BuiltIns[type_name], guard=lambda x: py_value(a.value <= x.value <= b.value))))
#
#
#     for type_name in ('num', 'ratio', 'float', 'int', 'str'):
#         if type_name == 'int':
#             pass
#         pass # BuiltIns[type_name].add_option(ArgsMatcher(NumericParam, NumericParam), make_lambda_guard(type_name))
#
#
#     # BuiltIns['str'].add_option(ArgsMatcher(StringParam),
#     #                            lambda regex: ArgsMatcher(Parameter(TraitMatcher(
#     #                                BuiltIns['str'], guard=lambda s: py_value(bool(re.fullmatch(regex.value, s.value)))))))
#     # BuiltIns['num'].add_option(ArgsMatcher(NumericParam, NumericParam), lambda a, b: py_value(TableMatcher(BuiltIns['num'], guard=lambda x: py_value(a.value <= x.value <= b.value))))
#
#     # Add shortcut syntax for adding function guards to type checks.  Eg `int > 0` or `float < 1.0`
#     def number_guard(op_sym: str):
#         # assert a.value == b.type
#         return lambda t, n: ArgsMatcher(Parameter(TraitMatcher(t, guard=lambda x: Op[op_sym].fn.call(x, n))))
#
#     # generating functions with syntax like `str > 5` => `[str x]: len(x) > 5`
#     def string_guard(op_sym: str):
#         # assert a.value == BuiltIns['str'] and b.type in (BuiltIns['int'], BuiltIns['float'])
#         return lambda t, n: ArgsMatcher(Parameter(TraitMatcher(t, guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), n))))
#         # def guard(x, y):
#         #     return ArgsMatcher(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), b))))
#         # # return guard
#         # return ArgsMatcher(Parameter(TableMatcher(BuiltIns['str'], guard=lambda s: Op[op_sym].fn.call(py_value(len(s.value)), b))))
#
#     # def add_guards(op_sym: str):
#     #     Op[op_sym].fn.add_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['int'])), NumericParam),
#     #                                 number_guard(op_sym))
#     #     Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['float'])), NumericParam),
#     #                                 lambda a, b: number_guard(a, b, op_sym))
#     #     Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['ratio'])), NumericParam),
#     #                                 lambda a, b: number_guard(a, b, op_sym))
#     #     Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
#     #                                 lambda a, b: string_guard(a, b, op_sym))
#
#
#     for op_sym in ('>', '<', '>=', '<='):
#         for type_name in ('int', 'ratio', 'float', 'num'):
#             Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[type_name])), NumericParam),
#                                         number_guard(op_sym))
#         Op[op_sym].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns['str'])), NumericParam),
#                                     string_guard(op_sym))
#         # add_guards(op)

def make_op_equals_functions(sym: str):
    op_fn = Op[sym].fn
    op_name = sym + '='
    Op[op_name].fn = Function({ArgsMatcher(StringParam, AnyParam):
                                   lambda name, val: Context.env.assign(name.value,
                                                                        op_fn.call(Context.deref(name), val)),
                               ArgsMatcher(FunctionParam, AnyParam, AnyParam):
                                   lambda *args: set_or_assign_option(*args, operation=op_fn),
                               ArgsMatcher(AnyParam, StringParam, AnyParam):
                                   lambda rec, name, val: rec.set(name.value, op_fn.call(rec.get(name.value), val))
                               }, name=op_name)
    Op[op_name].eval_args = eval_assign_args


for sym in ('+', '-', '*', '/', '//', '**', '%', '??', '&', '|'):
    make_op_equals_functions(sym)

""" This is an option that matches (1) value int (the trait itself) and (2) any int value.  Eg, `int < 2`.
    The resolution of the option is a function that returns a pattern.  The pattern is an int trait matcher, with a
    specification via Lambda Matcher that said int ought to be less than the value in the pattern expression. 
"""
for trait in ('int', 'ratio', 'float', 'num'):
    Op['<'].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                         Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: y.value < x.value))
                             )
    Op['<='].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                          Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: y.value <= x.value))
                              )
    Op['>='].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                          Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: y.value >= x.value))
                              )
    Op['>'].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                         Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: y.value > x.value))
                             )

for trait in ('str', 'list'):
    Op['<'].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                         Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: len(y.value) < x.value))
                             )
    Op['<='].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                          Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: len(y.value) <= x.value))
                              )
    Op['>='].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                          Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: len(y.value) >= x.value))
                              )
    Op['>'].fn.assign_option(ArgsMatcher(Parameter(ValueMatcher(BuiltIns[trait])),
                                         Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: len(y.value) > x.value))
                             )
