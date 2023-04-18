from Abstract_Syntax_Tree import Tokenizer, AST
from BuiltIns import *
from StaticOperators import *

mode = 'shell'

if mode == 'shell':
    root = Function()
    BuiltIns['pili'] = root
    for key, builtin in BuiltIns.items():
        if not builtin.name:
            builtin.name = key
        root.add_option(ListPatt(Parameter(key)), builtin)
    Context.root = root
    while True:
        line = input()
        tokenizer = Tokenizer(line)
        ast = AST(tokenizer)
        block = FuncBlock(ast.block)
        fn = block.make_function({})
        Context.push(Context.line, fn, Option(Any))
        expr = block.exprs[0]
        output = expr.evaluate()
        # output = block.execute()
        print(BuiltIns['string'].call([output]))



if mode == 'test':
    script_path = "test_script.pili"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    ast = AST(tokenizer)
    root = Function(ListPatt(Parameter('main')), FuncBlock(ast.block))
    BuiltIns['pili'] = root
    for key, builtin in BuiltIns.items():
        if not builtin.name:
            builtin.name = key
        root.add_option(ListPatt(Parameter(key)), builtin)
    Context.root = root
    output = root.deref('main')
    print(output)
