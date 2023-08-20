#!/usr/bin/env python3
import sys
import timeit
from Abstract_Syntax_Tree import Tokenizer, AST
from BuiltIns import *
import operators
# import StaticOperators

# def factory(repeats: int):
# 	def p(msg):
# 		for i in range(repeats):
# 			print(f"{i}: {msg}")
# 	return p
#
# print3 = factory(3)
# print3("hello")
# exit()

if len(sys.argv) == 1 and sys.executable == '/usr/bin/python3':  # test if running in console; pycharm executable is python3.10
    mode = 'shell'
elif len(sys.argv) == 2:
    mode = 'script'
    script_path = sys.argv[1]
else:
    mode = 'test'
    script_path = "test_script.pili"
    script_path = 'syntax_demo.pili'
    # script_path = "Dates.pili"
    # script_path = 'fibonacci.pili'
    # script_path = 'test.pili'
    script_path = 'Tables.pili'
    print('(test mode) running script', script_path, '...')

# pili = Function({Pattern(): lambda: NotImplemented}, name='pili')
# BuiltIns['pili'] = pili
# for key, builtin in BuiltIns.items():
#     if not getattr(builtin, 'name', False):
#         builtin.name = key
#     pili.names[key] = builtin
#     # pili.add_option(Pattern(Parameter(Matcher(key))), builtin)
# Context.root = pili
# Context.push(0, pili, Option(Any))

top_closure = TopNamespace(BuiltIns)  # Closure(Syntax.Block([]), bindings=BuiltIns)
Context.root = top_closure
Context.push(0, top_closure)

def execute_code(code: str) -> Function:
    block = CodeBlock(AST(Tokenizer(code+"\n")).block)
    block.scope = top_closure
    return block.execute(())
    # if len(block.exprs) == 1:
    #     return block.exprs[0].evaluate()
    # Context.env.assign_option(Pattern(Parameter('main')), block)
    # return Context.deref('main')
    # fn = block.make_function({}, root)
    # Context.push(Context.line, fn, Option(Any))

Context.exec = execute_code

if mode == 'shell':
    Context.push(0, pili)
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
            if output != py_value(None):
                if output.instanceof(BuiltIns['str']):
                    print(output.value)
                else:
                    output_string = BuiltIns['string'].call([output]).value
                    if output_string != 'root.main':
                        print(output_string)
        except Exception as e:
            print("Exception: ", e, '\n***')


def execute_script(path):
    with open(path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    # print(tokenizer)
    # print('**********************************')
    ast = AST(tokenizer)
    # print(ast)
    # print('**********************************')
    # pili.assign_option(Pattern(), CodeBlock(ast.block))
    main_block = CodeBlock(ast.block)
    # BuiltIns['pili'] = root
    # for key, builtin in BuiltIns.items():
    #     if not builtin.name:
    #         builtin.name = key
    #     root.add_option(ListPatt(Parameter(key)), builtin)
    # Context.root = root
    output = main_block.execute(())
    # output = pili.deref('main')
    print('Script finished with output: ', output)


if mode in ('test', 'script'):
    repeats = 1
    t = timeit.timeit(lambda: execute_script(script_path), number=repeats) / repeats
    print('\n\n***************************************\n')
    print(t)
    # execute_script(script_path)