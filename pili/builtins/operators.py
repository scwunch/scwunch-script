from pili.syntax import default_op_fn
from .base import *
from ..utils import limit_str
from operator import lt, le, gt, ge

print(f'loading {__name__}.py')

Operator.fn = Map({AnyPattern: default_op_fn})

def identity(x): return x

Op[';'].fn = Map({AnyPlusPattern: lambda *args: args[-1]})

def eval_eq_args(lhs: Node, *val_nodes: Node) -> Args:
    values = (Closure(node) if isinstance(node, Block) else node.evaluate() for node in val_nodes)
    match lhs:
        # case Token(TokenType.Name, text=name):
        #     patt = Parameter(AnyMatcher(), name)
        case OpExpr('.', [loc_node, Token(TokenType.Name, text=name)]):
            patt = BindPropertyPattern(loc_node.evaluate(), name)
            # rec = loc_node.evaluate()
            # return Args(rec, py_value(name), *values)
        case OpExpr('[', [loc_node, args]):
            patt = BindKeyPattern(loc_node.evaluate(), args.evaluate())
            # match loc_node:
            #     case OpExpr('.', [rec_node, Token(TokenType.Name, text=name)]):
            #         return Args(rec_node.evaluate(), py_value(name), args.evaluate(), *values)
            # return Args(loc_node.evaluate(), args.evaluate(), *values)
        case _:
            patt = lhs.eval_pattern(as_param=True)
    return Args(patt, *values)

def set_with_fn(operation: Map = None):
    def inner(patt: Pattern, left: Record, right: Record):
        val = operation.call(Args(left, right))
        return patt.match_and_bind(val)
    return inner


Op['='].eval_args = eval_eq_args
Op['='].fn = Map({ParamSet(PatternParam, AnyParam):
                           lambda patt, val: patt.match_and_bind(val),
                  # ParamSet(AnyParam, Parameter(AnyMatcher(), quantifier="+")): dot_set_fn,
                  AnyParam: identity},
                 name='=')
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
            patt = BindPropertyPattern(rec, name)
            # return Args(rec, py_value(name), rhs.evaluate())
        case OpExpr('[', terms):
            rec, args = [t.evaluate() for t in terms]
            exists = BuiltIns['has'].call(Args(rec, args)).value
            existing = lhs.evaluate() if exists else BuiltIns['blank']
            if existing != BuiltIns['blank']:
                return Args(existing)
            patt = BindKeyPattern(rec, args)
            # match loc_node:
            #     case OpExpr('.', [rec_node, Token(TokenType.Name, text=name)]):
            #         return Args(rec_node.evaluate(), py_value(name), args.evaluate(), *values)
            # return Args(loc_node.evaluate(), args.evaluate(), *values)
        case _:
            raise SyntaxErr(f'Line {lhs.line} in "{self.file}": "{lhs.source_text}" is invalid syntax for left-hand-side of "??=".')
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
            return Args(params.evaluate(), resolution)
        case OpExpr('[', [fn_node, ParamsNode() as params]):
            """ foo[params]: ... """
            match fn_node:
                case Token(TokenType.Name, text=name):
                    fn = state.deref(name, None)
                    if fn is None:
                        fn = Map(name=name)
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

Op[':='].fn = BuiltIns['transmogrify']

def assign_option(*args):
    match args:
        case fn, pattern, resolution:
            fn.assign_option(pattern, resolution, prioritize=True)
        case Args() | ParamSet() as key, resolution:
            if not isinstance(state.env.fn, Map):
                raise EnvironmentError(f"Line {state.line}: Cannot assign key-value option in this context.  "
                                       f"Must be within a definition of a map, class, or trait.")
            state.env.fn.assign_option(key, resolution)
        case _:
            raise RuntimeErr(f"Line {state.line}: wrong arguments for colon function.")
    return BuiltIns['blank']

Op[':'].eval_args = eval_colon_args
Op[':'].fn = Map({AnyPlusPattern: assign_option})


def eval_dot_args(lhs: Node, rhs: Node) -> Args:
    if rhs.type == TokenType.Name:
        rhs: Token
        right_arg = py_value(rhs.text)
    else:
        right_arg = rhs.evaluate()
    return Args(lhs.evaluate(), right_arg)


Op['.'].fn = Map({ParamSet(AnyParam, StringParam): dot_call_fn}, name='.')
Op['.?'].fn = Map({ParamSet(AnyParam, StringParam):
                            lambda a, b: dot_call_fn(a, b, safe_get=True)},
                  name='.?')
Op['..'].fn = Map({ParamSet(IterParam, MapParam):
                            lambda it, fn: py_value([fn.call(el) for el in it]),
                   ParamSet(IterParam, StringParam):
                            lambda it, name: py_value([dot_call_fn(el, name) for el in it]),
                   }, name='..')
Op['..?'].fn = Map({ParamSet(IterParam, MapParam):
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
Op['['].fn = Map({ParamSet(AnyParam, Parameter(AnyMatcher(), None, '?'), ArgsParam,
                           named_params=make_flags('swizzle', 'safe_get')): dot_call_fn,
                  }, name='call')

Op['call?'].fn = Map({ParamSet(AnyParam, Parameter(AnyMatcher(), None, '?'), ArgsParam,
                               named_params=make_flags('swizzle', 'safe_get')):
                               lambda *args, **kwargs: dot_call_fn(*args, **kwargs, safe_call=True),
                      }, name='call?')
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
Op['=>'].fn = Map({AnyBinopPattern: lambda params, block: Map({params: block})},
                  name='=>')
def eval_comma_args(*nodes) -> Args:
    return Args(*eval_list_nodes(nodes))
Op[','].eval_args = eval_comma_args
Op[','].fn = Map({AnyPlusPattern: lambda *args: py_value(args)},
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
Op['??'].fn = Map({AnyPlusPattern: nullish})

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
Op['or'].fn = Map({AnyPlusPattern: make_or_fn(False)},
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
Op['and'].fn = Map({AnyPlusPattern: make_and_fn(False)},
                   name='and')

Op['not'].fn = Map({AnyParam: lambda x: py_value(not x.truthy)},
                   name='not')

Op['in'].fn = Map({ParamSet(AnyParam, MapParam):
                            lambda a, b: py_value(Args(a) in b.op_map),
                   ParamSet(AnyParam, Parameter(UnionMatcher(NonStrSeqParam, SetParam))):
                            lambda a, b: py_value(a in b.value),
                   ParamSet(AnyParam, StringParam):
                            lambda a, b: py_value(a.value in b.value)},
                  name='in')

Op['=='].fn = Map({AnyPlusPattern: lambda a, *args: py_value(all(a == b for b in args))},
                  name='==')

def neq(*args: Record):
    if len(args) <= 1:
        return BuiltIns['false']
    if not Op['=='].fn.call(Args(*args)).truthy:
        return BuiltIns['true']
    return neq(*args[1:])
Op['!='].fn = Map({AnyPlusPattern: neq},
                  name='!=')
def eval_is_op_args(lhs: Node, rhs: Node) -> Args:
    # if rhs.type is TokenType.Name:
    #     rhs = BindExpr(rhs)
    return Args(lhs.evaluate(), rhs.eval_pattern())
Op['is'].eval_args = Op['is not'].eval_args = eval_is_op_args
Op['is'].fn = Map({AnyBinopPattern: lambda a, b: py_value(b.match(a) is not None)},
                  name='is')
Op['is not'].fn = Map({AnyBinopPattern: lambda a, b: py_value(b.match(a) is None)},
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

Op['<'].fn = Map({AnyPlusPattern: make_comp_fn(lt)},
                 name='<')
Op['>'].fn = Map({AnyPlusPattern: make_comp_fn(gt)},
                 name='>')
Op['<='].fn = Map({AnyBinopPattern:
                         lambda a, b: py_value(BuiltIns['<'].call(a, b).value or BuiltIns['=='].call(a, b).value),
                   AnyPlusPattern: make_comp_fn(le)},
                  name='<=')
Op['>='].fn = Map({AnyBinopPattern:
                         lambda a, b: py_value(BuiltIns['>'].call(a, b).value or BuiltIns['=='].call(a, b).value),
                   AnyPlusPattern: make_comp_fn(ge)},
                  name='>=')

Op['to'].fn = Map({ParamSet(*[Parameter(UnionMatcher(TraitMatcher(NumTrait), ValueMatcher(BuiltIns['blank'])))] * 2):
                            lambda *args: Range(*args)},
                  name='to')
Op['>>'].fn = Op['to'].fn
Op['>>'].eval_args = Op['to'].eval_args = lambda *nodes: Args(*(n.evaluate() for n in nodes))
Op['by'].fn = Map({ParamSet(Parameter(TraitMatcher(RangeTrait)), NumericParam):
                            lambda r, step: Range(*r.data[:2], step),
                   ParamSet(SeqParam, NumericParam): lambda seq, step: (v for v in seq[::step.value])},
                  name='by')
Op['+'].fn = Map({Parameter(TraitMatcher(NumTrait), quantifier='+'):
                           lambda *args: py_value(sum(n.value for n in args)),
                  Parameter(TraitMatcher(StrTrait), quantifier='+'):
                           lambda *args: py_value(''.join(n.value for n in args)),
                  # ParamSet(AnyParam): lambda a: BuiltIns['num'].call(a),
                  Parameter(TraitMatcher(ListTrait), quantifier='+'):
                           lambda *args: py_value(sum((n.value for n in args), [])),
                  Parameter(TraitMatcher(TupTrait), quantifier='+'):
                           lambda *args: py_value(sum((n.value for n in args), ())),
                  }, name='+')
Op['-'].fn = Map({NormalBinopPattern: lambda a, b: py_value(a.value - b.value),
                  ParamSet(AnyParam): lambda a: py_value(-a.value)},
                 name='-')
def product(*args: PyValue):
    acc = args[0].value
    for n in args[1:]:
        if acc == 0:
            return py_value(0)
        acc *= n.value
    return py_value(acc)

Op['*'].fn = Map({Parameter(TraitMatcher(NumTrait), quantifier='+'): product,
                  ParamSet(SeqParam, IntegralParam): lambda a, b: py_value(a.value * b.value)},
                 name='*')
Op['/'].fn = Map({ParamSet(RationalParam, RationalParam): lambda a, b:
                       py_value(Fraction(a.value.numerator * b.value.denominator, a.value.denominator * b.value.numerator)),
                  ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value / b.value)},
                 name='/')
Op['//'].fn = Map({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value // b.value)},
                  name='//')
Op['%'].fn = Map({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value % b.value)},
                 name='%')
Op['**'].fn = Map({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)},
                  name='**')
Op['^'].fn = Map({ParamSet(NumericParam, NumericParam): lambda a, b: py_value(a.value ** b.value)},
                 name='^')

def optionalize(arg: Pattern) -> Pattern:
    match patternize(arg):
        case BindingPattern() as patt:
            return patt.merge(default=BuiltIns['blank'])
        case patt:
            return UnionMatcher(patt, ValueMatcher(BuiltIns['blank']))
Op['?'].fn = Map({AnyParam: optionalize},
                 name='?')

def has_option(fn: Record, arg: Record = None) -> PyValue:
    if arg is None:
        fn, arg = None, fn

    match fn, arg:
        case None, PyValue(value=str() as name):
            return py_value(state.deref(name, None) is not None)
        case None, _:
            raise TypeErr(f"Line {state.line}: When used as a prefix, "
                          f"the right-hand term of the `has` operator must be a string, not {arg.cls}")
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


Op['has'].fn = Map({ParamSet(AnyParam, NonStrSeqParam): has_option,
                    ParamSet(StringParam): lambda s: py_value(state.deref(s, None) is not None),
                    ParamSet(NormalParam): has_option,
                    AnyBinopPattern: has_option},
                   name='has')
def eval_args_as_pattern(*nodes: Node) -> Args:
    return Args(*(node.eval_pattern() for node in nodes))
Op['|'].eval_args = Op['&'].eval_args = Op['~'].eval_args = Op['?'].eval_args = Op['@'].eval_args = eval_args_as_pattern
# def extract_matchers(params: tuple[Pattern, ...]):
#     for param in params:
#         param = patternize(param)
#         if not isinstance(param, Parameter) or param.quantifier or param.default:
#             raise NotImplementedError("Not yet implemented union/intersection of multi-parameter patterns.")
#         if param.binding:
#             yield BindingMatcher(param.pattern, param.binding)
#         else:
#             yield param.pattern
Op['|'].fn = Map({AnyPlusPattern: lambda *args: UnionMatcher(*map(patternize, args))},
                 name='|')
Op['&'].fn = Map({AnyPlusPattern: lambda *args: IntersectionMatcher(*map(patternize, args))},
                 name='&')

def invert_pattern(rec: Record):
    match patternize(rec):
        case Parameter(pattern=Matcher() as patt, binding=b, quantifier=q, default=d):
            if q and q[0] in "+*":
                raise NotImplementedError
            return Parameter(NotMatcher(patt), b, q, d)
    raise NotImplementedError
Op['~'].fn = Map({AnyParam: lambda arg: NotMatcher(patternize(arg)),
                  AnyBinopPattern: lambda a, b:
                                        IntersectionMatcher(patternize(a), NotMatcher(patternize(b)))},
                 name='~')
Op['@'].fn = Map({AnyParam: lambda x: patternize(x)})


# Op['!'].fn = Map({AnyParam: lambda rec: Parameter(ValueMatcher(rec))})

def eval_declaration_arg(arg: Node) -> Args:
    match arg:
        case Token(type=TokenType.Name, text=name):
            return Args(py_value(name))
    raise AssertionError


Op['var'].eval_args = Op['local'].eval_args = eval_declaration_arg
Op['var'].fn = Map({StringParam: lambda x: VarPatt(x.value)})
Op['local'].fn = Map({StringParam: lambda x: LocalPatt(x.value)})

def make_op_equals_functions(sym: str):
    match sym:
        case '&&':
            op_fn = Op['and'].fn
        case '||':
            op_fn = Op['or'].fn
        case _:
            op_fn = Op[sym].fn
    op_name = sym + '='
    Op[op_name].fn = Map({ParamSet(StringParam, AnyParam):
                                   lambda name, val: state.env.assign(name.value,
                                                                        op_fn.call(state.deref(name), val)),
                          ParamSet(MapParam, AnyParam, AnyParam):
                                   lambda *args: set_or_assign_option(*args, operation=op_fn),
                          ParamSet(AnyParam, StringParam, AnyParam):
                                   lambda rec, name, val: rec.set(name.value, op_fn.call(rec.get(name.value), val))
                          }, name=op_name)
    Op[op_name].fn = Map({ParamSet(PatternParam, AnyParam, AnyParam): set_with_fn(op_fn)
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
    Op[op_name].fn = Map({ParamSet(PatternParam, AnyParam, AnyParam): set_with_fn(op_fn)
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
                                                              LambdaMatcher(lambda y: t(y) < x.value)),
                             prioritize=True)
    Op['<='].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                       Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: t(y) <= x.value)),
                              prioritize=True)
    Op['>='].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                       Parameter(TraitMatcher(BuiltIns['num']))),
                              lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                               LambdaMatcher(lambda y: t(y) >= x.value)),
                              prioritize=True)
    Op['>'].fn.assign_option(ParamSet(Parameter(ValueMatcher(BuiltIns[trait])),
                                      Parameter(TraitMatcher(BuiltIns['num']))),
                             lambda _, x: IntersectionMatcher(TraitMatcher(BuiltIns[trait]),
                                                              LambdaMatcher(lambda y: t(y) > x.value)),
                             prioritize=True)


for trait in ('int', 'ratio', 'float', 'num', 'str', 'list', 'tuple', 'set', 'frozenset'):
    make_comp_opt(trait)
