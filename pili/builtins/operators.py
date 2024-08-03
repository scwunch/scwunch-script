from pili.syntax import default_op_fn
from .base import *
from ..utils import limit_str
from operator import lt, le, gt, ge

print(f'loading {__name__}.py')

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
        rhs: Token
        right_arg = py_value(rhs.text)
    else:
        right_arg = rhs.evaluate()
    return Args(lhs.evaluate(), right_arg)


# caller_patt = ParamSet(AnyParam, AnyParam, named_params={'caller': Parameter(AnyMatcher(), 'caller', '?')})
# note: caller_patt should be (FunctionParam, ArgsParam), but I just made it any, any for a slight speed boost

# Op['.'].fn = Function({caller_patt: dot_call_fn,
#                        StringParam: lambda a: state.deref(a.value),
#                        ParamSet(AnyParam, StringParam): dot_call_fn,
#                        ParamSet(SeqParam, ArgsParam): list_get,
#                        ParamSet(Parameter(TableMatcher(BuiltIns['PythonObject'])),
#                                 Parameter(UnionMatcher(TraitMatcher(FuncTrait), TableMatcher(BuiltIns['Table'])))):
#                            py_dot,  # I don't remember why the second parameter for the pydot is func|table ???
#                        ParamSet(Parameter(TableMatcher(BuiltIns['PythonObject'])), AnyParam): py_dot
#                        })
Op['.'].fn = Function({ParamSet(AnyParam, StringParam): dot_call_fn}, name='.')
# Op['.?'].fn = Function({caller_patt:
#                             lambda a, b: BuiltIns['.'].call(a,b)
#                                          if BuiltIns['has'].call(a, b).value else py_value(None),
#                         ParamSet(StringParam):
#                             lambda a: BuiltIns['.'].call(a)
#                                       if BuiltIns['has'].call(a).value else py_value(None),
#                         })
Op['.?'].fn = Function({ParamSet(AnyParam, StringParam):
                            lambda a, b: dot_call_fn(a, b, safe_get=True)},
                                         # if BuiltIns['has'].call(a, b).value else BuiltIns['blank']},
                       name='.?')
Op['..'].fn = Function({ParamSet(IterParam, FunctionParam):
                            lambda it, fn: py_value([fn.call(el) for el in it]),
                        ParamSet(IterParam, StringParam):
                            lambda it, name: py_value([dot_call_fn(el, name) for el in it]),
                        }, name='..')
Op['..?'].fn = Function({ParamSet(IterParam, FunctionParam):
                            lambda it, fn: py_value([fn.call(el, safe_call=True) for el in it]),
                        ParamSet(IterParam, StringParam):
                            lambda it, name: py_value([dot_call_fn(el, name, safe_get=True) for el in it]),
                        }, name='..')
Op['.'].eval_args = Op['.?'].eval_args = Op['..'].eval_args = Op['..?'].eval_args = eval_dot_args

def eval_call_args(lhs: Node, rhs: Node) -> Args:
    args = rhs.evaluate()
    match lhs:
        case OpExpr('.'|'.?'|'..'|'..?' as op, [loc_node, Token(TokenType.Name, text=name)]):
            rec = loc_node.evaluate()
            args = Args(rec, py_value(name), args)
            if op.startswith('..'):
                args.named_arguments['swizzle'] = BuiltIns['true']
            if op.endswith('?'):
                args.named_arguments['safe_get'] = BuiltIns['true']
            return args
        case _:
            return Args(lhs.evaluate(), args)


Op['['].eval_args = Op['call?'].eval_args = eval_call_args
Op['['].fn = Function({ParamSet(AnyParam, Parameter(AnyMatcher(), None, '?'), ArgsParam,
                                named_params=make_flags('swizzle', 'safe_get')): dot_call_fn,
                       }, name='call')

# def safe_call_fn
Op['call?'].fn = Function({ParamSet(AnyParam, Parameter(AnyMatcher(), None, '?'), ArgsParam,
                                    named_params=make_flags('swizzle', 'safe_get')):
                               lambda *args, **kwargs: dot_call_fn(*args, **kwargs, safe_call=True),
                           }, name='call?')
# Op['['].fn = Function({ParamSet(AnyParam, ArgsParam): Record.call,  # lambda rec, args: rec.call(args),
#                        ParamSet(SeqParam, ArgsParam): list_get,
#                        ParamSet(Parameter(TableMatcher(BuiltIns['PythonObject'])), ArgsParam): call_py_obj,
#                        ParamSet(Parameter(TraitMatcher(IterTrait)), StringParam, ArgsParam):
#                             lambda it, s, args: None
#                        }, name='call')
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

Op['=='].fn = Function({AnyPlusPattern: lambda a, *args: py_value(all(a == b for b in args))},
                       name='==')

def neq(*args: Record):
    if len(args) <= 1:
        return BuiltIns['false']
    if not Op['=='].fn.call(Args(*args)).truthy:
        return BuiltIns['true']
    return neq(*args[1:])
Op['!='].fn = Function({AnyPlusPattern: neq},
                       name='!=')
def eval_is_op_args(lhs: Node, rhs: Node) -> Args:
    # if rhs.type is TokenType.Name:
    #     rhs = BindExpr(rhs)
    return Args(lhs.evaluate(), rhs.eval_pattern())
Op['is'].eval_args = Op['is not'].eval_args = eval_is_op_args
Op['is'].fn = Function({AnyBinopPattern: lambda a, b: py_value(b.match(a) is not None)},
                       name='is')
Op['is not'].fn = Function({AnyBinopPattern: lambda a, b: py_value(b.match(a) is None)},
                           name='is not')
def make_comp_fn(opfn: PyFunction):
    def inner(*args):
        if len(args) < 2:
            raise ValueError(f'Line {state.line}: called comparative function with less than two arguments.')
        for i in range(1, len(args)):
            if not opfn(args[i-1].value, args[i].value):
                return BuiltIns['false']
        return BuiltIns['true']
    return inner

Op['<'].fn = Function({AnyPlusPattern: make_comp_fn(lt)},
                      name='<')
Op['>'].fn = Function({AnyPlusPattern: make_comp_fn(gt)},
                      name='>')
Op['<='].fn = Function({AnyBinopPattern:
                         lambda a, b: py_value(BuiltIns['<'].call(a, b).value or BuiltIns['=='].call(a, b).value),
                        AnyPlusPattern: make_comp_fn(le)},
                       name='<=')
Op['>='].fn = Function({AnyBinopPattern:
                         lambda a, b: py_value(BuiltIns['>'].call(a, b).value or BuiltIns['=='].call(a, b).value),
                        AnyPlusPattern: make_comp_fn(ge)},
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
def optionalize(arg: Record) -> Parameter:
    param: Parameter = patternize(arg)
    patt, binding, default, quantifier = param.pattern, param.binding, param.default, param.quantifier
    if not isinstance(patt, Matcher):
        raise NotImplementedError
    if quantifier in ('+', '*'):
        quantifier = '*'
    else:
        quantifier = '?'
    return Parameter(UnionMatcher(patt, ValueMatcher(BuiltIns['blank'])), binding, quantifier, default)
Op['?'].fn = Function({AnyParam: optionalize},
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
def eval_args_as_pattern(*nodes: Node) -> Args:
    return Args(*(node.eval_pattern() for node in nodes))
Op['|'].eval_args = Op['&'].eval_args = Op['~'].eval_args = eval_args_as_pattern
def extract_matchers(params: tuple[Pattern, ...]):
    for param in params:
        param = patternize(param)
        if not isinstance(param, Parameter) or param.binding or param.quantifier or param.default:
            raise NotImplementedError("Not yet implemented UnionParams / UnionPatts")
        yield param.pattern
Op['|'].fn = Function({AnyPlusPattern: lambda *args: Parameter(UnionMatcher(*extract_matchers(args)))},
                      name='|')
Op['&'].fn = Function({AnyPlusPattern: lambda *args: Parameter(IntersectionMatcher(*extract_matchers(args)))},
                      name='&')

def invert_pattern(rec: Record):
    match patternize(rec):
        case Parameter(pattern=Matcher() as patt, binding=b, quantifier=q, default=d):
            if q and q[0] in "+*":
                raise NotImplementedError
            return Parameter(NotMatcher(patt), b, q, d)
    raise NotImplementedError
Op['~'].fn = Function({AnyParam: invert_pattern,
                       AnyBinopPattern: lambda a, b:
                                        Parameter(IntersectionMatcher(*extract_matchers((a, invert_pattern(b)))))},
                      name='~')
Op['@'].fn = Function({AnyParam: lambda rec: Parameter(ValueMatcher(rec))})


# Op['!'].fn = Function({AnyParam: lambda rec: Parameter(ValueMatcher(rec))})

def eval_declaration_arg(_, arg: Node) -> Args:
    match arg:
        case Token(type=TokenType.Name, text=name):
            return Args(py_value(name))
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
