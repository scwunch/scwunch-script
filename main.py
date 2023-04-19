#!/usr/bin/env python3
import sys
from Abstract_Syntax_Tree import Tokenizer, AST
from BuiltIns import *
from StaticOperators import *


def convert(name: str) -> Function:
    o = object()
    py_fn = getattr(__builtins__, name, o)
    if py_fn is o:
        raise SyntaxErr(f"Name '{name}' not found.")
    root.add_option(ListPatt(Parameter(name)), lambda *args: Value(py_fn((arg.value for arg in args))))

if len(sys.argv) == 1 and sys.executable == '/usr/bin/python3':  # test if running in console; pycharm executable is python3.10
    mode = 'shell'
elif len(sys.argv) == 2:
    mode = 'script'
    script_path = sys.argv[1]
else:
    mode = 'test'
    script_path = "test_script.pili"
    print('(test mode) running script', script_path)

root = Function(ListPatt(Parameter('main')), lambda x: NotImplemented)
BuiltIns['pili'] = root
for key, builtin in BuiltIns.items():
    if not builtin.name:
        builtin.name = key
    root.add_option(ListPatt(Parameter(key)), builtin)
Context.root = root
Context.push(0, root, Option(Any))

def execute_code(code: str) -> Function:
    block = FuncBlock(AST(Tokenizer(code)).block)
    if len(block.exprs) == 1:
        return block.exprs[0].evaluate()
    Context.env.assign_option(ListPatt(Parameter('main')), block)
    return Context.env.deref('main')
    # fn = block.make_function({}, root)
    # Context.push(Context.line, fn, Option(Any))


if mode == 'shell':
    while True:
        code = ''
        next_line = input('> ')
        if next_line.endswith(' '):
            while next_line:
                code += next_line + '\n'
                next_line = input('  ')
        else:
            code = next_line
        try:
            output = execute_code(code)
            if output != Value(None):
                if output.instanceof(BuiltIns['str']):
                    print(output.value)
                else:
                    output_string = BuiltIns['string'].call([output]).value
                    if output_string != 'root.main':
                        print(output_string)
        except Exception as e:
            print("Exception: ", e, '\n***')


if mode in ('test', 'script'):
    # script_path = "test_script.pili"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    ast = AST(tokenizer)
    root.add_option(ListPatt(Parameter('main')), FuncBlock(ast.block))
    # BuiltIns['pili'] = root
    # for key, builtin in BuiltIns.items():
    #     if not builtin.name:
    #         builtin.name = key
    #     root.add_option(ListPatt(Parameter(key)), builtin)
    # Context.root = root
    output = root.deref('main')
    print(output)
