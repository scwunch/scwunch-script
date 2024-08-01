import importlib
from . import state
# from runtime import FieldMatcher
from .state import Op, BuiltIns
from .utils import read_number, ContextErr, SyntaxErr, RuntimeErr, TypeErr
from .runtime import *
from .syntax import Node, TokenType, Token, Position, Operator, SINGLETONS, ListNode, Block

print(f'loading {__name__}.py')

Node.eval_pattern = lambda self, name_as_any=False: patternize(self.evaluate())

def eval_token(self: Token):
    s = self.text
    match self.type:
        case TokenType.Singleton:
            return py_value(SINGLETONS[s])
        case TokenType.Number:
            return py_value(read_number(s, state.settings['base']))
        case TokenType.StringLiteral:
            return py_value(s.strip("`"))
        case TokenType.Name:
            if s == 'self':
                return state.deref(s, state.env.fn)
            return state.deref(s)
    raise NotImplementedError(f"Line {self.line}: Could not evaluate token", self)
Token.evaluate = eval_token

def eval_token_pattern(self: Token, name_as_any=False):
    if name_as_any and self.type is TokenType.Name:
        return Parameter(AnyMatcher(), self.text)
    return patternize(self.evaluate())
Token.eval_pattern = eval_token_pattern

def execute_block(self: Block) -> Record:
    line = state.line
    val = BuiltIns['blank']
    for tbl in self.table_names:
        # noinspection PyTypeChecker
        state.env.locals[tbl] = ListTable(name=tbl, uninitialized=True)
    for trait in self.trait_names:
        # noinspection PyTypeChecker
        state.env.locals[trait] = Trait(name=trait, uninitialized=True)
    for fn in self.function_names:
        # noinspection PyTypeChecker
        state.env.locals[fn] = Function(name=fn, uninitialized=True)
    for expr in self.statements:
        state.line = expr.line
        val = expr.evaluate()
        if state.env.return_value:
            break
        if state.break_loop or state.continue_:
            break
    state.line = line
    return val

class StringNode(ListNode):
    def __repr__(self):
        return self.source_text

    def evaluate(self):
        def eval_string_parts(nodes: list[Node]) -> str:
            for node in nodes:
                if node.type != TokenType.StringPart:
                    yield BuiltIns['str'].call(node.evaluate()).value
                    continue
                text = node.source_text
                start = i = 0
                while i < len(text):
                    if text[i] == '\\':
                        c = text[i+1]
                        if c in "nrtbf'\"{\\":
                            yield text[start:i]
                            i += 2
                            start = i
                            match c:
                                case 'n':
                                    yield '\n'
                                case 'r':
                                    yield '\r'
                                case 't':
                                    yield '\t'
                                case 'b':
                                    yield '\b'
                                case 'f':
                                    yield '\f'
                                case _:
                                    # assume char is one of: ', ", {, \
                                    yield c
                        else:
                            raise SyntaxErr(f"Unrecognized escape character '{c}' @{node.pos}")
                    i += 1
                yield text[start:]

        return py_value(''.join(eval_string_parts(self.nodes)))


class Block(ListNode):
    """
    a container for executables (statements, e
    representing the lines of code to put into a function
    """
    empty = frozenset()
    statements: list[Node]
    table_names: set[str]
    trait_names: set[str]
    function_names: set[str]

    def __init__(self, nodes: list[Node],
                 table_names: set[str] = empty, trait_names: set[str] = empty, func_names: set[str] = empty,
                 pos: Position = None):
        super().__init__(nodes, pos)
        self.statements = nodes
        self.table_names = table_names
        self.trait_names = trait_names
        self. function_names = func_names

    def evaluate(self):
        # raise NotImplementedError("Use Block.execute instead.")
        self.execute()
        return BuiltIns['blank']

    def execute(self):
        line = state.line
        val = BuiltIns['blank']
        for tbl in self.table_names:
            # noinspection PyTypeChecker
            state.env.locals[tbl] = ListTable(name=tbl, uninitialized=True)
        for trait in self.trait_names:
            # noinspection PyTypeChecker
            state.env.locals[trait] = Trait(name=trait, uninitialized=True)
        for fn in self.function_names:
            # noinspection PyTypeChecker
            state.env.locals[fn] = Function(name=fn, uninitialized=True)
        for expr in self.statements:
            state.line = expr.line
            val = expr.evaluate()
            if state.env.return_value:
                break
            if state.break_loop or state.continue_:
                break
        state.line = line
        return val

    def __repr__(self):
        if not self.statements:
            return 'Block[empty]'
        elif len(self.statements) == 1:
            return f"Block[{repr(self.statements[0])}]"
        else:
            return f"Block[{len(self.statements)} statements]"


class ListLiteral(ListNode):
    def evaluate(self):
        return py_value(list(eval_list_nodes(self.nodes)))

    def eval_pattern(self, name_as_any=False) -> Pattern:
        return ParamSet(*(item.eval_pattern() for item in self.nodes))

    def __repr__(self):
        res = map(str, self.nodes)
        return f"{self.__class__.__name__}[{', '.join(res)}]"


class TupleLiteral(ListLiteral):
    def evaluate(self):
        return py_value(tuple(eval_list_nodes(self.nodes)))


class ArgsNode(ListNode):
    def evaluate(self):
        named_args = {}
        flags = set()

        def generate_args(nodes: list[Node]):
            for node in nodes:
                match node:
                    case OpExpr('=', [Token(type=TokenType.Name, source_text=name), val_node]):
                        named_args[name] = val_node.evaluate()
                    case OpExpr('!', [EmptyExpr(), Token(type=TokenType.Name, source_text=name)]):
                        flags.add(name)
                    case OpExpr('*', [EmptyExpr(), iter_node]):
                        yield from BuiltIns['iter'].call(iter_node.evaluate())
                    case _:
                        yield node.evaluate()

        return Args(*generate_args(self.nodes), flags=flags, named_arguments=named_args)

    def eval_pattern(self, name_as_any=False) -> Pattern:
        raise NotImplementedError

    def __repr__(self):
        return f"ArgsNode{self.nodes}"


class ParamsNode(ListNode):
    named_params: list[Node]
    any_kwargs: bool | str = False
    def __init__(self, nodes: list[Node], named_params: list[Node], pos: Position = None):
        super().__init__(nodes, pos)
        if named_params and named_params[-1].source_text == '*':
            named_params.pop()
            self.any_kwargs = True
        self.named_params = named_params

    def evaluate(self):
        def gen_params(nodes) -> tuple[str, Parameter]:
            for node in nodes:
                match node:
                    case OpExpr('!', [EmptyExpr(), Token(TokenType.Name, text=name)]):
                        param = Parameter(TraitMatcher(BuiltIns['bool']), name, '?', BuiltIns['blank'])
                    case _:
                        param = node.eval_pattern(name_as_any=True)
                yield param.binding, param

        return ParamSet(*(p[1] for p in gen_params(self.nodes)),
                        named_params=dict(tuple(gen_params(self.named_params))),
                        kwargs=self.any_kwargs)

    eval_pattern = evaluate

    def __repr__(self):
        return f'ParamsNode({', '.join(map(str, self.nodes))}; {', '.join(map(str, self.named_params))})'


class FieldMatcherNode(ListNode):
    def evaluate(self):
        field_dict = {}

        def generate_matchers(nodes: list[Node]):
            for node in nodes:
                match node:
                    case OpExpr(':', [Token(type=TokenType.Name, text=name), patt_node]):
                        field_dict[name] = patt_node.eval_pattern(name_as_any=True)
                    case _:
                        yield node.eval_pattern()

        return FieldMatcher(tuple(generate_matchers(self.nodes)), field_dict)

    def eval_pattern(self, name_as_any=False) -> Pattern:
        return Parameter(self.evaluate())


class FunctionLiteral(ListNode):
    def evaluate(self):
        fn = Function()
        Closure(Block(self.nodes)).execute(link_frame=fn)
        return fn

    def eval_pattern(self, name_as_any=False) -> Pattern:
        raise NotImplementedError


class OpExpr(Node):
    op: Operator
    terms: tuple[Node, ...]
    fixity: str  # prefix|postfix
    def __init__(self, op: Operator, *terms: Node, pos: Position = None, fixity: str = None):
        assert terms
        self.pos = pos
        self.op = op
        if op.text == '=>':
            pos = terms[1].pos
            terms = terms[0], Block([CommandWithExpr('return', terms[1], pos)], pos=pos)
        if op.text == ':' and isinstance(terms[1], Block):
            match terms[0]:
                case OpExpr('[') | OpExpr('.', [EmptyExpr(), _]) | ParamsNode():
                    pass
                case _:
                    raise SyntaxErr(f'Line {self.line}: invalid syntax: "{terms[0].source_text}".  A colon followed by '
                                    f'a block must be a `.dot[method]:`, `[param list]:`, or `option[def]`:')
        self.terms = terms

    @property
    def sym(self):
        return self.op.text

    __match_args__ = ('sym', 'terms')

    def evaluate(self):
        args: Args = self.op.eval_args(*self.terms)
        return self.op.fn.call(args)

    def eval_pattern(self, name_as_any=False) -> Pattern:
        match self.op.text:
            case ',':
                return ParamSet(*(t.eval_pattern(name_as_any=True) for t in self.terms))
            case ':':
                lhs, rhs = self.terms
                if not (isinstance(lhs, Token) and lhs.type == TokenType.Name):
                    raise SyntaxErr(f'Line {self.line}: could not patternize "{self.source_text}"; '
                                    f'left-hand-side of colon must be a name.')
                    # TODO: but at some point I will make this work for options too like Foo(["key"]: str value)
                    # probably in the context of braces like {"key": str value}
                field_name = lhs.text
                return Parameter(FieldMatcher((), {field_name: rhs.eval_pattern(name_as_any=True)}))
            case '=':
                lhs, rhs = self.terms
                default = rhs.evaluate()
                left = lhs.eval_pattern(name_as_any=True)
                return Parameter(left, default=default)
            case '+' if len(self.terms) == 1:
                param = self.terms[0].eval_pattern()
                if not isinstance(param, Parameter):
                    return Parameter(param, quantifier='+')
            case _:
                pass

        return patternize(self.evaluate())

    def __repr__(self):
        if self.op.text == '[':
            return f"{self.terms[0]}[{self.terms[1]}]"
        return f"{self.op}({', '.join(map(str, self.terms))})"


class BindExpr(Node):
    node: Node
    name: str | None
    quantifier: str
    def __init__(self, node: Node, name: str = None, quantifier: str = '', pos: Position = None):
        self.node = node
        self.name = name
        self.quantifier = quantifier
        self.pos = pos

    def evaluate(self, name_as_any=False):
        if self.node.type is TokenType.Name:
            return Parameter(patternize(self.node.evaluate()), self.name, self.quantifier)
        return Parameter(self.node.eval_pattern(), self.name, self.quantifier)

    eval_pattern = evaluate

    def __repr__(self):
        return f"Bind({self.node} {self.name}{self.quantifier})"


class IfElse(Node):
    do: Node
    condition: Node
    alt: Node
    def __init__(self, do: Node, cond: Node, alt: Node, pos: Position = None):
        self.do = do
        self.condition = cond
        self.alt = alt
        if pos is None:
            self.pos = self.do.pos + self.alt.pos
        else:
            self.pos = pos

    def evaluate(self):
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.do.evaluate()
        else:
            return self.alt.evaluate()

    def eval_pattern(self, name_as_any=False) -> Pattern:
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.do.eval_pattern()
        else:
            return self.alt.eval_pattern()

    def __repr__(self):
        return f"IfElse({self.do} if {self.condition} else {self.alt})"

class ExprWithBlock(Node):
    header: Node
    block: Block | None
    alt: Node | Block | None = None
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        if pos is None:
            self.pos = header.pos + (blk_nodes[-1].pos if blk_nodes else header.pos)
        else:
            self.pos = pos
        self.header = header
        match blk_nodes:
            case [Block() as blk]:
                self.block = blk
            case [Block() as blk, Block() as alt]:
                self.block = blk
                self.alt = alt
            case [Block() as blk, elif_header, *alt_expr_nodes]:
                self.block = blk
                pos = elif_header.pos + alt_expr_nodes[-1].pos
                self.alt = Conditional(elif_header, alt_expr_nodes, pos)
            case []:
                self.block = None
            case _:
                raise NotImplementedError(f"Line {blk_nodes[0].line}: process else statement...")

    def __repr__(self):
        alt = ' else ' + repr(self.alt) if self.alt else ''
        return f"{self.__class__.__name__}({self.header} => {self.block}{alt})"


class Conditional(ExprWithBlock):
    condition: Node
    consequent: Block

    def evaluate(self):
        condition = self.header.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.block.evaluate()
        elif self.alt:
            return self.alt.evaluate()
        else:
            return BuiltIns['blank']

class ForLoop(ExprWithBlock):
    var: Node
    iterable: Node
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        super().__init__(header, blk_nodes, pos)
        match header:
            case OpExpr('in', terms):
                self.var, self.iterable = terms
        # for i, node in enumerate(header_nodes):
        #     if node.source_text == 'in':
        #         self.var = mathological(header_nodes[:i])
        #         self.iterable = mathological(header_nodes[i+1:])
        #         break
        # else:
            case _:
                raise SyntaxErr(f"Line {self.line}: For loop expression expected 'in' keyword.")

    def evaluate(self):
        iterator = BuiltIns['iter'].call(self.iterable.evaluate())
        match self.var:
            case Token(type=TokenType.Name, source_text=var_name):
                pass
            case _:
                raise NotImplementedError

        var_patt: Pattern = Op['='].eval_args(self.var)[0]

        for val in iterator:
            var_patt.match_and_bind(val)
            # if (bindings := var_patt.match(val)) is None:
            #     raise MatchErr(f"Line {self.line}: "
            #                    f"pattern '{var_patt}' did not match value {val} in {self.iterable.source_text}")
            # for k, v in bindings.items():
            #     state.env.assign(k, v)
            # state.env.locals[var_name] = val
            self.block.execute()
            if state.break_loop:
                state.break_loop -= 1
                break
            elif state.continue_:
                state.continue_ -= 1
                if state.continue_:
                    break
            elif state.env.return_value:
                break
        return BuiltIns['blank']

class WhileLoop(ExprWithBlock):
    def evaluate(self):
        result = py_value(None)
        for i in range(6 ** 6):
            if state.break_loop:
                state.break_loop -= 1
                return py_value(None)
            condition_value = self.header.evaluate()
            if BuiltIns['bool'].call(condition_value).value:
                result = self.block.evaluate()
            else:
                return result
            if state.break_loop:
                state.break_loop -= 1
                return py_value(None)
        raise RuntimeErr(f"Line {self.line or state.line}: Loop exceeded limit of 46656 executions.")


class Command(Node):
    command: str
    def __init__(self, cmd: str, pos: Position = None):
        self.pos = pos
        self.command = cmd


class CommandWithExpr(Command):
    expr: Node
    def __init__(self, cmd: str, expr: Node, pos: Position = None):
        super().__init__(cmd, pos)
        self.expr = expr

    def evaluate(self):
        match self.command:
            case 'exit':
                print('Exiting now')
                exit()
            case 'debug':
                state.debug = True
                print('Start debugging...')
                result = self.expr.evaluate()
                return result
            case 'return':
                result = self.expr.evaluate()
                state.env.return_value = result
                return result
            case 'print':
                # print('!@>', BuiltIns['string'].call(self.expr.evaluate()).value)
                print(BuiltIns['str'].call(self.expr.evaluate()).value)
                return BuiltIns['blank']
            case 'break':
                result = self.expr.evaluate()
                match result.value:
                    case None:
                        levels = 1
                    case int() as levels:
                        pass
                    case _:
                        raise RuntimeErr(f"Line {state.line}: "
                                         f"break expression should evaluate to non-negative integer.  Found {result}.")
                state.break_loop += levels
            case 'continue':
                result = self.expr.evaluate()
                match result.value:
                    case None:
                        levels = 1
                    case int() as levels:
                        pass
                    case _:
                        raise RuntimeErr(f"Line {state.line}: "
                                         f"break expression should evaluate to non-negative integer.  Found {result}.")
                state.continue_ += levels
            case 'import':
                module_name, _, var_name = self.expr.source_text.partition(' as ')
                mod = importlib.import_module(module_name)
                globals()[var_name or module_name] = mod
                # state.env.assign_option(var_name or module_name, piliize(a))
                state.env.locals[var_name or module_name] = py_value(mod)
                return py_value(mod)
            case 'label':
                state.env.name = BuiltIns['str'].call(self.expr.evaluate()).value
            case _:
                raise SyntaxErr(f"Line {state.line}: Unhandled command {self.command}")

    def eval_pattern(self, name_as_any=False) -> Pattern:
        match self.command:
            case 'debug':
                state.debug = True
                print('Start debugging...')
                result = self.expr.eval_pattern(name_as_any)
                return result

    def __repr__(self):
        return f"Cmd:{self.command}({self.expr})"


# this function needs to be defined in this module so that it has access to the imports processed by importlib
# opt: Option = BuiltIns['python'].op_list[0]
def run_python_code(code: PyValue[str], direct=None, execute=None):
    """  direct:  return the value without wrapping it in PyValue or PyObj
        execute:  handle statements like def, class, assignment, etc.  Returns blank
                  without this flag, will only evaluate an expression and return the value
    """
    if direct is None:
        direct = BuiltIns['blank']
    if execute is None:
        execute = BuiltIns['blank']
    if direct.truthy and execute.truthy:
        raise RuntimeErr(f'Line {state.line}: "direct" and "execute" flags are incompatible.')
    if execute.truthy:
        exec(code.value)
        return BuiltIns['blank']
    value = eval(code.value)
    return value if direct.truthy else py_value(value)
# opt.fn = run_python_code


class NamedExpr(Command):
    name: str
    expr: Node
    def __init__(self, cmd: str, name: str, pos: Position = None):
        super().__init__(cmd, pos)
        self.name = name

class FunctionExpr(ExprWithBlock):
    fn_name: str
    body: Block
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        if len(blk_nodes) > 1:
            raise SyntaxErr(f"Line {blk_nodes[1].line}: function blocks cannot have else statements.")
        super().__init__(header, blk_nodes, pos)
        match header:
            case Token(type=TokenType.Name, text=name):
                self.fn_name = name
            case _:
                raise SyntaxErr(f"Line {header.line}: "
                                f"Function/Trait syntax should be `function|trait <fn_name> <block>`.")
        self.body = self.block
        # match nodes.nodes:
        #     case [Token(type=TokenType.Name, text=name), Block() as blk]:
        #         self.fn_name = name
        #         self.body = blk
        #     case _:
        #         raise SyntaxErr(f"Line {self.line}: Function syntax should be `function <fn_name> <block>`.")

    def evaluate(self):
        fn = state.deref(self.fn_name)
        if isinstance(fn, Function):
            del fn.uninitialized
        else:
            fn = Function(name=self.fn_name)
            state.env.assign(self.fn_name, fn)
        if self.body is not None:
            Closure(self.body).execute(link_frame=fn)
        return fn

    def __repr__(self):
        return f'Function({self.fn_name}, {self.body})'

class TraitExpr(FunctionExpr):
    # trait_name: str
    # body: Block
    # def __init__(self, cmd: str, nodes: Concrete, pos: Position = None):
    #     super().__init__(cmd, pos)
    #     match nodes.nodes:
    #         case [Token(type=TokenType.Name, text=name), Block() as blk]:
    #             self.trait_name = name
    #             self.body = blk
    #         case _:
    #             raise SyntaxErr(f"Line {self.line}: Trait syntax should be `trait <trait_name> <block>`.")

    def evaluate(self):
        trait = state.deref(self.fn_name)
        if isinstance(trait, Trait) and hasattr(trait, 'uninitialized'):
            del trait.uninitialized
        else:
            trait = Trait(name=self.fn_name)
            state.env.assign(self.fn_name, trait)
        if self.body is not None:
            Closure(self.body).execute(link_frame=trait)
        return trait

    def __repr__(self):
        return f'Trait({self.fn_name}, {self.body})'

class TableExpr(FunctionExpr):
    table_name: str
    traits: list[Node]
    body: Block = None
    def __init__(self, header: Node, blk_nodes: list[Node], pos: Position = None):
        if len(blk_nodes) > 1:
            raise SyntaxErr(f"Line {blk_nodes[1].line}: function blocks cannot have else statements.")
        ExprWithBlock.__init__(self, header, blk_nodes, pos)
        self.body = self.block
        match header:
            case Token(type=TokenType.Name, text=name):
                self.table_name = name
                self.traits = []
            case OpExpr('&', [Token(type=TokenType.Name, text=name), FieldMatcherNode(nodes=trait_nodes)]):
                self.table_name = name
                self.traits = list(trait_nodes)
            case _:
                raise SyntaxErr(f"Line {self.line}: Table syntax should be `table <table_name> (<trait>)* <block>`.\n"
                                f"Eg, `table MyList(list, seq, iter)`")

    def evaluate(self):
        table = state.deref(self.table_name)
        if isinstance(table, Table):
            del table.uninitialized
        else:
            table = ListTable(name=self.table_name)
            state.env.assign(self.table_name, table)

        def gen_traits():
            for node in self.traits:
                trait = node.evaluate()
                if not isinstance(trait, Trait):
                    raise TypeErr(f"Line: {state.line}: expected trait, but '{node}' is {repr(trait)}")
                yield trait

        table.traits = (*table.traits, *gen_traits())
        if self.body is not None:
            Closure(self.body).execute(link_frame=table)
        table.integrate_traits()
        return table

    def __repr__(self):
        return f'Table({self.table_name} with {", ".join(map(str, self.traits))}, {self.body})'


class SlotExpr(NamedExpr):
    # slot_name: str
    # slot_type: Node
    default: None | Node | Block | str = None
    def __init__(self, cmd: str, field_name: str, node: Node, pos: Position = None):
        super().__init__(cmd, field_name, pos)
        # self.field_type = BindExpr(node, '')
        match node:
            case OpExpr('='|':', terms):
                self.field_type, self.default = terms
            case _:
                self.field_type = node

    def evaluate(self):
        # slot_type = patternize(self.slot_type.evaluate())
        slot_type = self.field_type.eval_pattern()
        # match patternize(self.slot_type.evaluate()):
        #     case ParamSet(parameters=(Parameter(matcher=slot_type),)):
        #         pass
        #     case _:
        #         raise TypeErr(f"Line {state.line}: Invalid type: {self.slot_type.evaluate()}.  "
        #                       f"Expected value, table, trait, or single-parameter pattern.")
        match self.default:
            case None:
                default = getattr(slot_type, 'default', None)
            case Block() as blk:
                default = Closure(blk)
            case Node():
                default = self.default.evaluate()
                # default = Function({Parameter(AnyMatcher(), 'self'): default_value})
            case _:
                raise ValueError("Unexpected default")
        if default is not None:
            default = Function({Parameter(AnyMatcher(), 'self'): default})
        slot = Slot(self.name, slot_type, default)
        match state.env.fn:
            case Trait() as trait:
                trait.fields.append(slot)
            case Table(traits=(Trait() as trait, *_)):
                trait.fields.append(slot)
            case Function() as fn:
                fn.update_field(slot)
            case _:
                raise AssertionError

        return BuiltIns['blank']

    def __repr__(self):
        default = f", {self.default}" or ''
        return f"Slot({self.name}, {self.field_type}{default})"

class FormulaExpr(NamedExpr):
    formula_name: str
    formula_type: Node
    block: Block
    def __init__(self, cmd: str, field_name: str, node: Node, pos: Position = None):
        raise DeprecationWarning('FormulaExpr (invoked by `formula field_name return_type: ...` is being deprecated'
                                 'in favour of dot methods.')
        # super().__init__(cmd, field_name, pos)
        # match node:
        #     case OpExpr(':', terms):
        #         self.field_type, self.block = terms
        #     case _:
        #         self.field_type = node

    def evaluate(self):
        formula_type = self.formula_type.eval_pattern()

        patt = ParamSet(Parameter(state.env.fn, binding='self'))
        formula_fn = Function({patt: Closure(self.block)})
        formula = Formula(self.formula_name, formula_type, formula_fn)
        match state.env.fn:
            case Trait() as trait:
                trait.fields.append(formula)
            case Table(traits=(Trait() as trait, *_)):
                trait.fields.append(formula)
            case Function() as fn:
                fn.update_field(formula)
            case _:
                raise AssertionError
        return BuiltIns['blank']

    def __repr__(self):
        return f"Formula({self.formula_name}, {self.formula_type}, {self.block})"

class SetterExpr(NamedExpr):
    field_name: str
    param_nodes: ParamsNode
    block: Block
    def __init__(self, cmd: str, field_name: str, node: Node, pos: Position = None):
        super().__init__(cmd, field_name, pos)
        self.field_name = field_name
        match node:
            case OpExpr(':', [ParamsNode(nodes=param_nodes) as params, Block() as blk]):
                param_nodes.insert(0, BindExpr(Token('any'), 'self'))
                self.params = params
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Setter syntax is: `setter <name>[<value parameter>]: <block>`."
                                f"Eg, `setter description[str desc]: self._description = trim[desc]`")

    def evaluate(self):
        # fn = Function({ParamSet(Parameter(AnyMatcher(), 'self'), make_param(self.param_nodes)):
        #                    CodeBlock(self.block)})
        params: ParamSet = self.params.evaluate()
        # params.parameters = (Parameter(AnyMatcher(), 'self'),) + params.parameters
        assert len(params) == 2
        fn = Function({params: Closure(self.block)})
        setter = Setter(self.field_name, fn)
        match state.env.fn:
            case Trait() as trait:
                trait.fields.append(setter)
            case Table(traits=(Trait() as trait, *_)) as table:
                trait.fields.append(setter)
            case Function() as fn:
                fn.update_field(setter)
            case _:
                raise AssertionError

        return BuiltIns['blank']

    def __repr__(self):
        return f"Setter({self.field_name}, {self.params}, {self.block})"

class OptExpr(Command):
    params: ParamsNode
    return_type: Node | None
    block: Block
    def __init__(self, cmd: str, node: Node, pos: Position = None):
        super().__init__(cmd, pos)
        match node:
            case OpExpr(':', [ParamsNode() as params, Block() as blk]):
                self.params = params
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Opt syntax is: `opt [param1, param2, ...]: <block>`."
                                f"Eg, `opt [int i, int j]: ...`")

    def evaluate(self):
        # self.return_type not yet implemented
        # match self.return_type.eval_pattern():
        #     case ParamSet(parameters=(Parameter(pattern=Matcher() as return_type))):
        #         pass
        #     case Matcher() as return_type:
        #         pass
        #     case _:
        #         raise TypeErr(f"Line {state.line}: Invalid return type: {self.return_type.evaluate()}.  "
        #                       f"Expected value, table, trait, or single-parameter pattern.")
        pattern = self.params.evaluate()
        match state.env.fn:
            case Trait() as trait:
                pass
            case Table(traits=(Trait() as trait, *_)):
                pass
            case Function():
                raise ContextErr(f'Line {self.line}: To add options to functions, omit the "opt" keyword.')
            case _:
                raise AssertionError
        trait.assign_option(pattern, Closure(self.block))
        return BuiltIns['blank']

    def __repr__(self):
        ret_type = f', {self.return_type}' if self.return_type else ''
        return f"Opt({self.params}{ret_type}, {self.block})"

class Declaration(NamedExpr):
    var_name: str
    # context_expr: None | Expression = None
    value_expr: None | Node = None

    def __init__(self, cmd: str, var_name: str, node: Node, pos: Position = None):
        super().__init__(cmd, var_name, pos)
        self.value_expr = node

    def __repr__(self):
        return f"{self.__class__.__name__}({self.var_name}={self.value_expr})"

class LocalExpr(Declaration):
    def evaluate(self):
        value = self.value_expr.evaluate()
        state.env.locals[self.var_name] = value
        return value

    def eval_pattern(self, name_as_any=False) -> Pattern:
        value = self.value_expr.evaluate()
        state.env.locals[self.var_name] = value
        return Parameter(AnyMatcher(), self.var_name, default=value)

class VarExpr(Declaration):
    def evaluate(self):
        value = self.value_expr.evaluate()
        state.env.vars[self.var_name] = value
        return value

    def eval_pattern(self, name_as_any=False) -> Pattern:
        value = self.value_expr.evaluate()
        state.env.vars[self.var_name] = value
        return Parameter(AnyMatcher(), self.var_name, default=value)

class EmptyExpr(Node):
    def __init__(self, pos: tuple[int, int]):
        self.pos = Position(pos)

    def evaluate(self):
        return BuiltIns['blank']

    def __repr__(self):
        return "EmptyExpr()"


Operator.eval_args = lambda op, *terms: Args(*(t.evaluate() for t in terms if not isinstance(t, EmptyExpr)))

def eval_list_nodes(nodes):
    for n in nodes:
        match n:
            case OpExpr('*', [EmptyExpr(), iter_node]):
                yield from BuiltIns['iter'].call(iter_node.evaluate())
            case EmptyExpr():
                continue
            case _:
                yield n.evaluate()


EXPRMAP = {
    'if': Conditional,
    'for': ForLoop,
    'while': WhileLoop,
    'trait': TraitExpr,
    'table': TableExpr,
    'function': FunctionExpr,
    'slot': SlotExpr,
    'formula': FormulaExpr,
    'opt': OptExpr,
    'setter': SetterExpr,
    'local': LocalExpr,
    'var': VarExpr,
}
