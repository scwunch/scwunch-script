import importlib
from Syntax import *
# from DataStructures import *
from tables import *
from Env import *


# def expressionize_OLD(nodes: list[Node] | Statement):
#     if isinstance(nodes, Statement):
#         line = nodes.pos[0]
#         src = nodes.source_text
#         nodes = nodes.nodes
#     else:
#         line = None
#         src = " ".join(n.source_text for n in nodes)
#     if not nodes:
#         return EmptyExpr()
#     if nodes[0].type == TokenType.Command:
#         return Command(nodes[0].source_text, nodes[1:], line, src)
#     if len(nodes) == 1:
#         return SingleNode(nodes[0], line, src)
#     match nodes[0].source_text:
#         case 'if':
#             return Conditional(nodes, line, src)
#         case 'for':
#             return ForLoop(nodes, line, src)
#         case 'while':
#             return WhileLoop(nodes, line, src)
#         # case 'try':
#         #     return TryCatch(nodes, line, src)
#
#     return Mathological(nodes, line, src)

def expressionize(nodes: list[Node] | Statement):
    if isinstance(nodes, Statement):
        line = nodes.pos[0]
        src = nodes.source_text
        nodes = nodes.nodes
    else:
        line = None
        src = " ".join(n.source_text for n in nodes)

    match nodes:
        case []:
            return EmptyExpr()
        case [Token(type=TokenType.Command, source_text=key_word), *other_nodes]:
            return expressions.get(key_word, CommandWithExpr)(key_word, other_nodes, line, src)
            # return Command(cmd, other_nodes, line, src)
        case [node]:
            return SingleNode(node, line, src)
        case [Token(type=TokenType.Keyword, source_text=key_word), *_]:
            return expressions[key_word](nodes, line, src)
            # match word:
            #     case 'if':
            #         return Conditional(nodes, line, src)
            #     case 'for':
            #         return ForLoop(nodes, line, src)
            #     case 'while':
            #         return WhileLoop(nodes, line, src)
        case _:
            return Mathological(nodes, line, src)

Context.make_expr = expressionize

class Expression:
    line: int = None
    nodes: list[Node] = None
    source: str = ""
    def __init__(self, line: int | None, source: str):
        self.line = line
        self.source = source

    def evaluate(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.nodes) if self.nodes else 0

    def __repr__(self):
        if self.line:
            return f"Line {self.line}: {self.source}"
        return self.source

class Mathological(Expression):
    op_idx: int
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)
        self.nodes = nodes
        op_idx = None
        min_precedence = math.inf
        if nodes[0].type == TokenType.Operator:
            pre_op = Op[nodes[0].source_text]
            if pre_op.prefix:
                min_precedence = pre_op.prefix
                op_idx = 0
        if nodes[-1].type == TokenType.Operator:
            post_op = Op[nodes[-1].source_text]
            if post_op.postfix and post_op.postfix < min_precedence:
                min_precedence = post_op.postfix
                op_idx = len(nodes) - 1

        def get_op(idx: int) -> Operator | None:
            if nodes[idx].type != TokenType.Operator:
                return None
            return Op[nodes[idx].source_text]
        prev_op = get_op(0)
        for i in range(1, len(nodes) - 1):
            op = get_op(i)
            if op and op.binop and op.binop < min_precedence + (op.associativity == 'left'):
                if op.prefix:
                    prev_op = get_op(i-1)
                    if prev_op and prev_op.binop and not prev_op.postfix:
                        continue
                if op.postfix:
                    next_op = get_op(i+1)
                    if next_op and next_op.binop and not next_op.prefix:
                        continue
                op_idx = i
                min_precedence = op.binop

        if op_idx is None:
            raise OperatorErr(f'Line {self.line}: No operator found in expression: {nodes}')
        self.op = Op[nodes[op_idx].source_text]
        self.lhs = nodes[:op_idx]
        self.rhs = nodes[op_idx+1:]

    def evaluate(self):
        # if self.op.static:
        #     return self.op.static(self.lhs, self.rhs)
        args = self.op.eval_args(self.lhs, self.rhs)
        return self.op.fn.call(*args)


class ExprWithBlock(Expression):
    header: list[Node]
    block: Block
    alt: list[Node] | CodeBlock
    # def __init__OLD(self, nodes: list[Node], line: int | None, source: str):
    #     super().__init__(line, source)
    #     for i, node in enumerate(nodes):
    #         if isinstance(node, Block):
    #             if nodes[i-1].source_text == ':':
    #                 raise SyntaxErr(f"Line ({self.line}): "
    #                                 f"Pili does not use colons for control blocks like if and for.")
    #             self.block_index = i
    #             self.header = nodes[:i]
    #             self.block = Block(node)
    #             if i+1 == len(nodes):
    #                 self.alt = []
    #                 break
    #             if not (nodes[i + 1].source_text == 'else' and i+2 < len(nodes)):
    #                 raise SyntaxErr(f"Line {nodes[i+1].pos[0]}: Expected else followed by block.  Got {nodes[i+1]}")
    #
    #             node = nodes[i+2]
    #             if isinstance(node, Block):
    #                 self.alt = CodeBlock(node)
    #             elif node.source_text == 'if':
    #                 self.alt = nodes[i+2:]
    #             elif node.source_text == ':':
    #                 raise SyntaxErr(f"Line ({node.pos[0]}): "
    #                                 f"Pili does not use colons for control blocks like if and for.")
    #             else:
    #                 raise SyntaxErr(f"Line {node.pos[0]}: "
    #                                 f"Expected 'else' block or 'else if' after if block.  Got \n\t{nodes[i+1:]}")
    #             break
    #     else:
    #         raise SyntaxErr(f"Line {self.line}: no block found after {nodes[0].source_text} statement.")

    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(line, source)
        try:
            i = next(i for i, node in enumerate(nodes) if isinstance(node, Block))
        except StopIteration:
            raise SyntaxErr(f"Line {self.line}: missing block after {nodes[0].source_text} statement.")
        if nodes[i - 1].source_text == ':':
            raise SyntaxErr(f"Line ({self.line}): "
                            f"Pili does not use colons for control blocks like if and for.")
        self.header = nodes[1:i]
        self.block = nodes[i]  # noqa  nodes[i]: Block
        match nodes[i+1:]:
            case []:
                self.alt = []
            case [Token(source_text='else'), *other_nodes]:
                self.alt = other_nodes
            case _:
                raise SyntaxErr(f"Line {nodes[i+1].pos[0]}: "
                                f"Expected else followed by statement or block.  Got {nodes[i+1:]}")


class Conditional(ExprWithBlock):
    condition: Expression
    consequent: CodeBlock
    alt: list[Node] | CodeBlock
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(nodes, line, source)
        self.condition = expressionize(self.header)
        self.consequent = CodeBlock(self.block)

    def evaluate(self):
        condition = self.condition.evaluate()
        condition = BuiltIns['bool'].call(condition).value
        if condition:
            return self.consequent.execute()
        elif isinstance(self.alt, CodeBlock):
            return self.alt.execute()
        else:
            return expressionize(self.alt).evaluate()


class ForLoop(ExprWithBlock):
    var: list[Node]
    iterable: Expression
    block: CodeBlock
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(nodes, line, source)
        self.block = CodeBlock(self.block)
        for i, node in enumerate(self.header):
            if node.source_text == 'in':
                self.var = self.header[:i]
                self.iterable = expressionize(self.header[i+1:])
                break
        else:
            raise SyntaxErr(f"Line {self.line}: For loop expression expected 'in' keyword.")

    def evaluate(self):
        iterator = get_iterable(self.iterable.evaluate())
        if iterator is None:
            raise TypeErr(f"Line {Context.line}: {self.iterable} is not iterable.")
        if len(self.var) == 1:
            var_node = self.var[0]
            if var_node.type == TokenType.Name:
                var_name = var_node.source_text
                # var_val = py_value(var_node.source_text)
            else:
                # var_val = eval_node(var_node)
                raise NotImplementedError
        else:
            raise NotImplementedError
            # var_val = expressionize(self.var).evaluate()
        # patt = patternize(var_val)
        # variable = Context.env.assign_option(patt, py_value(None))
        # variable = Option
        for val in iterator:
            # variable.assign(val)
            Context.env.names[var_name] = val
            self.block.execute()
            if Context.break_loop:
                Context.break_loop -= 1
                break
            elif Context.continue_:
                Context.continue_ -= 1
                if Context.continue_:
                    break
            elif Context.env.return_value:
                break
        return py_value(None)

class WhileLoop(ExprWithBlock):
    condition: Expression
    block: CodeBlock
    def __init__(self, nodes: list[Node], line: int | None, source: str):
        super().__init__(nodes, line, source)
        self.condition = expressionize(self.header)
        self.block = CodeBlock(self.block)

    def evaluate(self):
        result = py_value(None)
        for i in range(6 ** 6):
            if Context.break_loop:
                Context.break_loop -= 1
                return py_value(None)
            condition_value = self.condition.evaluate()
            if BuiltIns['bool'].call(condition_value).value:
                result = self.block.execute()
            else:
                return result
            if Context.break_loop:
                Context.break_loop -= 1
                return py_value(None)
        raise RuntimeErr(f"Line {self.line or Context.line}: Loop exceeded limit of 46656 executions.")


class Command(Expression):
    command: str
    def __init__(self, cmd: str, line: int | None, source: str):
        super().__init__(line, source)
        self.command = cmd

class CommandWithExpr(Command):
    expr: Expression
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        self.expr = expressionize(nodes)

    def evaluate(self):
        match self.command:
            case 'exit':
                print('Exiting now')
                exit()
            case 'debug':
                Context.debug = True
                result = self.expr.evaluate()
                return result
            case 'return':
                result = self.expr.evaluate()
                Context.env.return_value = result
                return result
            case 'print':
                # print('!@>', BuiltIns['string'].call(self.expr.evaluate()).value)
                print(BuiltIns['str'].call(self.expr.evaluate()).value)
                return py_value(None)
            case 'break':
                result = self.expr.evaluate()
                match result.value:
                    case None:
                        levels = 1
                    case int() as levels:
                        pass
                    case _:
                        raise RuntimeErr(f"Line {Context.line}: "
                                         f"break expression should evaluate to non-negative integer.  Found {result}.")
                Context.break_loop += levels
            case 'continue':
                result = self.expr.evaluate()
                match result.value:
                    case None:
                        levels = 1
                    case int() as levels:
                        pass
                    case _:
                        raise RuntimeErr(f"Line {Context.line}: "
                                         f"break expression should evaluate to non-negative integer.  Found {result}.")
                Context.continue_ += levels
            case 'import':
                module_name, _, var_name = self.expr.source.partition(' as ')
                a = importlib.import_module(module_name)
                globals()[var_name or module_name] = a
                # Context.env.assign_option(var_name or module_name, piliize(a))
                Context.env.names[var_name or module_name] = piliize(a)
                return piliize(a)
            # case 'inherit':
            #     result = self.expr.evaluate()
            #     types = result.value if result.instanceof(BuiltIns['tuple']) else (result,)
            #     Context.env.mro += types
            #     return py_value(Context.env.mro)
            case 'label':
                Context.env.name = BuiltIns['str'].call(self.expr.evaluate()).value
            case _:
                raise SyntaxErr(f"Line {Context.line}: Unhandled command {self.command}")

class TraitExpr(Command):
    trait_name: str
    body: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), Block() as blk]:
                self.trait_name = name
                self.body = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Trait syntax should be `trait <trait_name> <block>`.")

    def evaluate(self):
        trait = Trait(name=self.trait_name)
        Context.env.assign(self.trait_name, trait)
        if self.body is not None:
            CodeBlock(self.body).execute(fn=trait)
        return trait

class TableExpr(Command):
    table_name: str
    traits: list[str]
    body: Block = None
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), *trait_nodes, Block() as blk]:
                self.body = blk
            case [Token(type=TokenType.Name, source_text=name), *trait_nodes]:
                pass
            case _:
                raise SyntaxErr(f"Line {self.line}: Table syntax should be `table <table_name> (@<trait>)* <block>`.")
        self.table_name = name
        self.traits = []
        for i in range(0, len(trait_nodes), 2):
            match trait_nodes[i:i+2]:
                case [Token(source_text='@'), Token(type=TokenType.Name, source_text=name)]:
                    self.traits.append(name)
                case _:
                    raise SyntaxErr(f"Line {self.line}: "
                                    f"Table traits must be listed like `@trait_name` separated only by whitespace.")

    def evaluate(self):
        table = ListTable(name=self.table_name)
        Context.env.assign(self.table_name, table)
        table.traits += tuple(Context.deref(tname) for tname in self.traits)
        for trait, name in zip(table.traits, self.traits):
            if not isinstance(trait, Trait):
                raise TypeErr(f"Line: {Context.line}: '{name}' is not a Trait, it is {repr(trait)}")
        if self.body is not None:
            CodeBlock(self.body).execute(fn=table)
        return table

class SlotExpr(Command):
    slot_name: str
    slot_type: Expression
    default: None | Expression | Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), *other_nodes]:
                self.slot_name = name
            case _:
                raise SyntaxErr(f"Line {self.line}: Slot is missing name.")
        for i, node in enumerate(other_nodes):
            if node.source_text == '=':
                self.slot_type = expressionize(other_nodes[:i])
                self.default = expressionize(other_nodes[i+1:])
                break
            if node.source_text == ':':
                self.slot_type = expressionize(other_nodes[:i])
                if i+2 != len(other_nodes):
                    raise SyntaxErr(f"Line {self.line}: Slot is missing default block after `:`.")
                self.default = other_nodes[i+1]
                break
        else:
            self.slot_type = expressionize(other_nodes)
            self.default = None

    def evaluate(self):
        match patternize(self.slot_type.evaluate()):
            case Pattern(parameters=(Parameter(matcher=slot_type),)):
                pass
            case _:
                raise TypeErr(f"Line {Context.line}: Invalid type: {self.slot_type.evaluate()}.  "
                              f"Expected value, table, trait, or single-parameter pattern.")
        match self.default:
            case None:
                default = None
            case Expression():
                default_value = self.default.evaluate()
                default = Function({Parameter(AnyMatcher(), 'self'): default_value})
            case Block() as blk:
                default = Function({Parameter(AnyMatcher(), 'self'): CodeBlock(blk)})
            case _:
                raise ValueError("Unexpected default")
        slot = Slot(self.slot_name, slot_type, default)
        match Context.env.fn:
            case Trait() as trait:
                pass
            case Table(traits=(Trait() as trait, *_)) as table:
                table.getters[self.slot_name] = len(table.fields), slot
                table.fields.append(slot)
                table.setters[self.slot_name] = len(table.fields), slot
                table.fields.append(slot)
            case Function(trait=Trait() as trait):
                pass
            case _:
                raise AssertionError
        trait.upsert_field(slot)
        return BuiltIns['blank']

class FormulaExpr(Command):
    formula_name: str
    formula_type: Expression
    block: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), *patt_nodes, Token(source_text=':'), Block() as blk]:
                self.formula_name = name
                self.formula_type = expressionize(patt_nodes)
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Formula syntax is: `formula <name> <type_expr>?: <block>`."
                                f"Eg, `formula count int: len[self.items]`")

    def evaluate(self):
        match patternize(self.formula_type.evaluate()):
            case Pattern(parameters=(Parameter(matcher=formula_type), )):
                pass
            case _:
                raise TypeErr(f"Line {Context.line}: Invalid type: {self.formula_type.evaluate()}.  "
                              f"Expected value, table, trait, or single-parameter pattern.")
        formula_fn = Function({Pattern(): CodeBlock(self.block)})
        formula = Formula(self.formula_name, formula_type, formula_fn)
        match Context.env.fn:
            case Trait() as trait:
                pass
            case Table(traits=(Trait() as trait, *_)) as table:
                table.getters[self.formula_name] = len(table.fields), formula
                table.fields.append(formula)
            case Function(trait=Trait() as trait):
                pass
            case _:
                raise AssertionError
        trait.upsert_field(formula)
        return BuiltIns['blank']
class SetterExpr(Command):
    field_name: str
    param_nodes: list[Node]
    block: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [Token(type=TokenType.Name, source_text=name), Token(source_text='.'),
                  ListNode(items=[Statement() as stmt]),
                  Token(source_text=':'), Block() as blk]:
                self.field_name = name
                self.param_nodes = stmt.nodes
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Setter syntax is: `setter <name>[<value parameter>]: <block>`."
                                f"Eg, `setter description[str desc]: self._description = trim[desc]`")

    def evaluate(self):
        fn = Function({Pattern(Parameter(AnyMatcher(), 'self'), make_param(self.param_nodes)):
                           CodeBlock(self.block)})
        setter = Setter(self.field_name, fn)
        match Context.env.fn:
            case Trait() as trait:
                pass
            case Table(traits=(Trait() as trait, *_)) as table:
                table.setters[self.field_name] = len(table.fields), setter
                table.fields.append(setter)
            case Function(trait=Trait() as trait):
                pass
            case _:
                raise AssertionError

        # try:
        #     fid = trait.field_ids[self.field_name]
        #     field = trait.fields[fid]
        #     field.setter = fn
        # except KeyError:
        #     raise SlotErr(f"Line {Context.line}: No formula with name '{self.field_name}' to add setter to.")
        #     # DONE: make this work for adding setters to *other* traits applied (not just the trait in scope)
        return BuiltIns['blank']

class OptExpr(Command):
    params: list[Statement]
    return_type: Expression | None
    block: Block
    def __init__(self, cmd: str, nodes: list[Node], line: int | None, source: str):
        super().__init__(cmd, line, source)
        match nodes:
            case [ListNode(nodes=params), *patt_nodes, Token(source_text=':'), Block() as blk]:
                self.params = params
                self.return_type = expressionize(patt_nodes)
                self.block = blk
            case _:
                raise SyntaxErr(f"Line {self.line}: Opt syntax is: `opt [param1, param2, ...] <type_expr>?: <block>`."
                                f"Eg, `opt [int i, int j] str | blank : ...`")

    def evaluate(self):
        match patternize(self.return_type.evaluate()):
            case Pattern(parameters=(Parameter(matcher=return_type), )):
                pass
            case _:
                raise TypeErr(f"Line {Context.line}: Invalid type: {self.return_type.evaluate()}.  "
                              f"Expected value, table, trait, or single-parameter pattern.")
        pattern = Pattern(*map(make_param, self.params))
        assert isinstance(Context.env.fn, Table)
        trait: Trait = Context.env.fn.traits[0]
        trait.add_option(pattern, CodeBlock(self.block))
        return BuiltIns['blank']


expressions = {
    'if': Conditional,
    'for': ForLoop,
    'while': WhileLoop,
    'trait': TraitExpr,
    'table': TableExpr,
    'slot': SlotExpr,
    'formula': FormulaExpr,
    'opt': OptExpr,
    'setter': SetterExpr
}

def piliize(val):
    if isinstance(val, list | tuple):
        # gen = (py_value(v) for v in val)
        records = map(py_value, val)
        if isinstance(val, tuple):
            return py_value(tuple(records))
        if isinstance(val, list):
            return List(list(records))
    return py_value(val)

def py_eval(code):
    return piliize(eval(code.value))

class EmptyExpr(Expression):
    def __init__(self):
        pass
    def evaluate(self):
        return py_value(None)

class SingleNode(Expression):
    def __init__(self, node: Node, line: int | None, source: str):
        super().__init__(line, source)
        self.node = node
    def evaluate(self):
        return eval_node(self.node)


def eval_node(node: Node) -> Record:
    match node:
        case Statement() as statement:
            return expressionize(statement).evaluate()
        case Token() as tok:
            return eval_token(tok)
        case Block() | ListNode(list_type=ListType.Function) as block:
            return CodeBlock(block).execute(())
        case ListNode(nodes=nodes):
            return List(list(map(eval_node, nodes)))
        case StringNode(nodes=nodes):
            return py_value(''.join(map(eval_string_part, nodes)))
    raise ValueError(f'Could not evaluate node {node} at line: {node.pos}')

def eval_string_part(node: Node) -> str:
    if node.type == TokenType.StringPart:
        return eval_string(node.source_text)
    if isinstance(node, Statement):
        return BuiltIns['str'].call(expressionize(node).evaluate()).value
    raise ValueError('invalid string part')


def precook_args(op: Operator, lhs, rhs) -> list[Record]:
    if op.binop and lhs and rhs:
        args = [expressionize(lhs).evaluate(), expressionize(rhs).evaluate()]
    elif op.prefix and rhs or op.postfix and lhs:
        args = [expressionize(rhs or lhs).evaluate()]
    else:
        raise ArithmeticError("Mismatch between operator type and operand positions.")
    return args


Operator.eval_args = precook_args


def eval_token(tok: Token) -> Record:
    s = tok.source_text
    match tok.type:
        case TokenType.Singleton:
            return py_value(singletons[s])
        case TokenType.Number:
            return py_value(read_number(s, Context.settings['base']))
        case TokenType.StringLiteral:
            return py_value(s.strip("`"))
        case TokenType.Name:
            if s == 'self':
                return Context.deref(s, Context.env.caller)
            return Context.deref(s)
        case _:
            raise Exception("Could not evaluate token", tok)


def eval_string(text: str):
    value = ""
    for i in range(len(text)):
        if i and text[i-1] == '\\':
            continue
        if text[i] == '\\':
            match text[i+1]:
                case 'n':
                    value += '\n'
                case 'r':
                    value += '\r'
                case 't':
                    value += '\t'
                case 'b':
                    value += '\b'
                case 'f':
                    value += '\f'
                case c:
                    # assume char is one of: ', ", {, \
                    value += c
        else:
            value += text[i]
    return value


def get_iterable(val: Record):
    match val:
        case PyValue(value=tuple() | list() | set() | frozenset() as iterable):
            return iterable
        case Table(records=iterable):
            return iterable
        case PyValue(value=str() as string):
            return (py_value(c) for c in string)
    return None


def make_value_param(param_nodes: list[Node]) -> Parameter:
    name = None
    match param_nodes:
        case []:
            raise SyntaxErr(f"Expected function parameter on line {Context.line}.")
        case [Token(type=TokenType.Name, source_text=name)]:
            value = py_value(name)
        case [Token(type=TokenType.Name, source_text=name)]:
            value = py_value(name)
        case _:
            value = expressionize(param_nodes).evaluate()
    return Parameter(ValueMatcher(value), name)


def make_param(param_nodes: list[Node] | Statement) -> Parameter:
    if isinstance(param_nodes, Statement):
        param_nodes = param_nodes.nodes
    if not param_nodes:
        raise SyntaxErr(f"Expected function parameter on line {Context.line}; no nodes found.")
    quantifier = ""
    match param_nodes:
        case [*_, Token(type=TokenType.Operator, source_text=op)] if op in ('?', '+', '*', '??', '+?', '*?'):
                quantifier = op
                param_nodes = param_nodes[:-1]
        case [Token(type=TokenType.Name, source_text=dot_name)]:
            return Parameter(dot_name)
        case []:
            raise SyntaxErr(f"Expected function parameter on line {Context.line}; no nodes found.")
    name = None
    match param_nodes:
        case [node]:
            pattern_nodes = param_nodes
            # return Parameter(patternize(eval_node(node)), quantifier=quantifier)
        case [*pattern_nodes, Token(type=TokenType.Name, source_text=name)]:
            last_op = Op.get(param_nodes[-2].source_text, None)
            if last_op and last_op.binop:
                if last_op.postfix:
                    print("WARNING: ambiguous pattern")
                else:
                    pattern_nodes = param_nodes
        case [*pattern_nodes]:
            pass
        case _:
            raise SyntaxErr(f"Expected function parameter on line {Context.line}; found only quantifier {quantifier}")
    try:
        expr_val = expressionize(pattern_nodes).evaluate()
    except NoMatchingOptionError as e:
        print(f"Line {Context.line}: Warning: did you try to assign a bare function name without defining a type?")
        raise e
    param = patternize(expr_val).parameters[0]
    param.name = name
    param.quantifier = quantifier
    return param


def split(nodes: list[Node], splitter: TokenType) -> list[list[Node]]:
    start = 0
    groups: list[list[Node]] = []
    for i, node in enumerate(nodes):
        if isinstance(node, Token) and node.type == splitter:
            groups.append(nodes[start:i])
            start = i+1
    if start < len(nodes):
        groups.append(nodes[start:])
    return groups


# def read_option_OLD(nodes: list[Node], is_value=False) -> Option:
#     dot_option = nodes[0].source_text == '.'
#     match nodes:
#         # .[].[]
#         case[Token(source_text='.'), ListNode() as opt, Token(source_text='.'), ListNode() as param_list]:
#             fn_nodes = [Token('pili'), Token('.'), opt]
#             param_list = [item.nodes for item in param_list.nodes]
#         # .fn_nodes.[]
#         case [Token(source_text='.'), *fn_nodes, Token(source_text='.'), ListNode() as param_list]:
#             param_list = [item.nodes for item in param_list.nodes]
#         case [Token(source_text='.'), *fn_nodes]:
#             param_list = []
#         case [*fn_nodes, ListNode() as param_list]:
#             if fn_nodes and fn_nodes[-1].source_text == '.':
#                 fn_nodes.pop()
#             param_list = [item.nodes for item in param_list.nodes]
#         case [*fn_nodes, Token(source_text='.'), Token(type=TokenType.Name) as name_tok]:
#             param_list = [[name_tok]]
#         case _:
#             param_list = split(nodes, TokenType.Comma)
#             fn_nodes = False
#     if fn_nodes:
#         try:
#             if len(fn_nodes) == 1:
#                 name = fn_nodes[0].source_text
#                 fn_val = Context.deref(name, False)
#             else:
#                 fn_val = expressionize(fn_nodes).evaluate()
#                 if fn_val.type is BuiltIns['str']:
#                     fn_val = Context.deref(fn_val.value)
#         except NoMatchingOptionError:
#             if dot_option:
#                 # raise NoMatchingOptionError(f"Line {Context.line}: "
#                 #                             f"dot option {' '.join((map(str, fn_nodes)))} not found.")
#                 # make new function in the root scope
#                 temp_env = Context.env
#                 Context.env = Context._env[0]
#             opt = read_option(fn_nodes, True)
#             if opt.is_null():
#                 fn_val = Function()
#                 opt.value = fn_val
#             else:
#                 fn_val = opt.value
#             if dot_option:
#                 Context.env = temp_env
#             # how many levels deep should this go?
#             # This will recurse infinitely, potentially creating many function
#         # if fn_val.type != BasicType.Function:
#         #     raise RuntimeErr(f"Line {Context.line}: "
#         #                      f"Cannot add option to {fn_val.type.value} {' '.join((map(str, fn_nodes)))}")
#         fn = fn_val
#         definite_env = True
#     else:
#         fn = Context.env
#         definite_env = not is_value
#     params = map(make_param if not is_value else make_value_param, param_list)
#     if dot_option:
#         patt = Pattern(Parameter(TableMatcher(Context.env)), *params)
#     else:
#         patt = Pattern(*params)
#     # try:
#     #     # option = fn.select(patt, walk_prototype_chain=False, ascend_env=not definite_env)
#     #     option = fn.select_by_pattern(patt, walk_prototype_chain=False, ascend_env=not definite_env)
#     #     """critical design decision here: I want to have walk_prototype_chain=False so I don't assign variables from the prototype..."""
#     # except NoMatchingOptionError:
#     #     option = fn.add_option(patt)
#     option = fn.trait.select_by_pattern(patt, ascend_env=True) or fn.trait.add_option(patt)
#     option.dot_option = dot_option
#     return option

def read_option2(nodes: list[Node]) -> tuple[Function, Pattern, bool]:
    nodes = nodes[:]
    dot_option = nodes[0].source_text == '.'
    if dot_option:
        if len(nodes) > 2 and not isinstance(nodes[1], ListNode):
            nodes.pop(0)
        if not isinstance(nodes[-1], ListNode):
            nodes.append(Token('.'))
            nodes.append(ListNode([], ListType.Params))
    # if len(nodes) > 1 and nodes[-2].source_text != '.':
    #     option_node = List([Statement(nodes)])
    #     nodes = []
    # else:
    option_node = nodes.pop()
    # nodes.pop() if len(nodes) else None
    if len(nodes):
        penultimate = nodes.pop()
        if penultimate.source_text != '.':
            # read the whole lhs as a single parameter
            option_node = ListNode([Statement([*nodes, penultimate, option_node])], ListType.Params)
            nodes = []
    match option_node:
        case ListNode(nodes=param_list):
            param_list = [item.nodes for item in param_list]
        case Token(type=TokenType.Name) as name_tok:
            param_list = [[name_tok]]
        case _:
            # raise SyntaxErr(f"Line {Context.line}: Cannot read option {option_node}")
            param_list = [[option_node]]
    # if len(nodes):
    #     dot = nodes.pop()
    #     if dot.source_text != '.':
    #         raise SyntaxErr(f"Line {Context.line}: Couldn't read option following {dot}; expected '.' or '['")
    arg_list = name = None
    match nodes:
        # case [*context_nodes, Token(source_text='.'), List(nodes=arg_list)]:
        #     pass
        case [*context_nodes, Token(source_text='.'), Token(type=TokenType.Name, source_text=name)]:
            pass
        case [Token(type=TokenType.Name, source_text=name)]:
            context_nodes = False
        case _:
            context_nodes = nodes
    if context_nodes:
        context_fn = expressionize(context_nodes).evaluate()
        ascend = False
    else:
        context_fn = Context.env
        ascend = True

    if name:  # and not dot_option:
        try:
            fn = context_fn.deref(name, ascend)
        except NoMatchingOptionError:
            fn = Function(name=name)
            if dot_option:
                # Context.root.trait.add_option(name, fn)
                raise NotImplementedError
            else:
                context_fn.trait.add_option(name, fn)
    else:
        fn = context_fn


    # if arg_list:
    #     args = [expressionize(arg).evaluate() for arg in arg_list]
    #     try:
    #         opt, bindings = context_fn.select_and_bind(args, ascend)
    #     except NoMatchingOptionError:
    #         opt = bindings = None
    #     if opt:
    #         fn = opt.resolve(args, context_fn, bindings)
    #     else:
    #         fn = context_fn.trait.add_option(patt, resolution).value
    # elif name:
    #     try:
    #         fn = context_fn.deref(name, ascend)
    #     except NoMatchingOptionError:
    #         fn = context_fn.trait.add_option(name, Function(name=name)).value
    # else:
    #     fn = context_fn

    params = map(make_param, param_list)
    if dot_option:
        # patt = Pattern(Parameter(TableMatcher(Context.env)), *params)
        raise NotImplementedError
    else:
        patt = Pattern(*params)
    return fn, patt, dot_option


def read_option(nodes: list[Node]) -> tuple[Function, Pattern, bool]:
    nodes = nodes[:]
    dot_option = nodes[0].source_text == '.'
    if dot_option:
        if len(nodes) > 2 and not isinstance(nodes[1], ListNode):
            # why don't we unconditionally pop the dot?
            nodes.pop(0)
        if not isinstance(nodes[-1], ListNode):
            # append implicit empty-param-set after dot-option
            nodes.append(Token('.'))
            nodes.append(ListNode([], ListType.Params))

    option_node = nodes.pop()
    if len(nodes):
        penultimate = nodes.pop()
        if penultimate.source_text != '.':
            # read the whole lhs as a single parameter
            option_node = ListNode([Statement([*nodes, penultimate, option_node])], ListType.Params)
            nodes = []

    param_list: list[list[Node]]
    match option_node:
        case ListNode(nodes=param_list_nodes):
            param_list = [item.nodes for item in param_list_nodes]
        case Token(type=TokenType.Name) as name_tok:
            # not sure if I want to allow this anymore
            param_list = [[name_tok]]
        case _:
            # raise SyntaxErr(f"Line {Context.line}: Cannot read option {option_node}")
            param_list = [[option_node]]

    # at this point, we should have three options: nodes is empty, nodes is one name, or nodes is an expression
    # which should evaluate to the function where we assign the option
    match nodes:
        case []:
            context_fn: Function = Context.env.fn
        case [Token(type=TokenType.Name, source_text=name)]:
            context_fn = Context.deref(name, None)
            if context_fn is None:
                context_fn = Function(name=name)
                if dot_option:
                    Context.root.names[name] = context_fn
                else:
                    Context.env.names[name] = context_fn
        case _:
            context_fn = expressionize(nodes).evaluate()

    params = map(make_param, param_list)
    if dot_option:
        match Context.env.fn:
            case Trait() | Table() as t:
                matcher = TableMatcher(t)
            case Function() as f:
                matcher = ValueMatcher(f)
            case _:
                raise RuntimeErr(f"Line {Context.line}: "
                         f"Dot options can only be declared in a scope which is a function, table, or trait.")
        patt = Pattern(Parameter(matcher, name='self'), *params)
    else:
        patt = Pattern(*params)
    return context_fn, patt, dot_option


if __name__ == "__main__":
    n = read_number("40000.555555555555555", 6)
    while True:
        strinput = input("Input number: ")
        n = read_number(strinput, 6)
        # n = 999999999/100000000
        print("decimal: " + str(n))
        print("senary: " + write_number(n, 6, 25))
        # print("via bc:", base(str(n), 10, 6, 15, True, True))

