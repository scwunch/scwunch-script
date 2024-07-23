#!/usr/bin/env python3
import importlib
print(f"importing modules...")
import sys
import timeit
from Env import *
from BuiltIns import *
from Abstract_Syntax_Tree import Tokenizer, AST, mathological, Block
import operator_syntax

print(f"starting module {__name__} ...")


if len(sys.argv) == 1 and sys.executable == '/usr/bin/python3':  # test if running in console; pycharm executable is python3.10
    mode = 'shell'
elif len(sys.argv) == 2:
    mode = 'script'
    script_path = sys.argv[1]
else:
    mode = 'test'
    script_path = "test_script.pili"
    # script_path = "syntax_test.pili"
    # script_path = 'syntax_demo.pili'
    # script_path = "Dates.pili"
    # script_path = 'fibonacci.pili'
    # script_path = 'test.pili'
    # script_path = 'Tables.pili'
    # script_path = 'pili_interpreter.pili'
    print('(test mode) running script', script_path, '...')

# pili = Function({ParamSet(): lambda: NotImplemented}, name='pili')
# BuiltIns['pili'] = pili
# for key, builtin in BuiltIns.items():
#     if not getattr(builtin, 'name', False):
#         builtin.name = key
#     pili.names[key] = builtin
#     # pili.add_option(ParamSet(Parameter(Matcher(key))), builtin)
# Context.root = pili
# Context.push(0, pili, Option(Any))

for sym, op in Op.items():
    if sym in BuiltIns:
        pass
    BuiltIns[sym] = op.fn

for k, v in BuiltIns.items():
    if hasattr(v, 'name') and v.name is None:
        v.name = k
builtin_namespace = GlobalFrame(BuiltIns)  # Closure(Syntax.Block([]), bindings=BuiltIns)
Context.root = builtin_namespace
Context.push(-1, builtin_namespace)

def execute_code(code: str) -> Function:
    block = Closure(AST(Tokenizer(code + "\n")).block)
    block.scope |= builtin_namespace
    return block.execute(Args())


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
                    output_string = BuiltIns['string'].call(output).value
                    if output_string != 'root.main':
                        print(output_string)
        except Exception as e:
            print("Exception: ", e, '\n***')


def execute_script(path):
    with open(path) as f:
        script_string = f.read()

    Context.source_code = script_string
    tokenizer = Tokenizer(script_string)
    print(tokenizer)
    ast = AST(tokenizer)
    block = ast.block
    # print(ast)
    # print('**********************************')
    # pili.assign_option(ParamSet(), CodeBlock(ast.block))
    # main_block = CodeBlock(ast.block)
    # BuiltIns['pili'] = root
    # for key, builtin in BuiltIns.items():
    #     if not builtin.name:
    #         builtin.name = key
    #     root.add_option(ListPatt(Parameter(key)), builtin)
    # Context.root = root
    try:
        output = Closure(block).execute(Args())
    except PiliException as e:
        raise e
    except Exception as e:
        e.add_note(Context.get_trace() + f"\n> Line {Context.line}: python exception")
        raise e
    # output = pili.deref('main')
    print('Script finished with output: ', output)


# def test(*args):
#     return args
#
# # print(test(args=(1,2,3)))
#
# prog = [
#     Inst().match(TraitMatcher(IntTrait), 'a'),
#     Inst().save('b'),
#     Inst().match(TraitMatcher(StrTrait)),
#     Inst().split(-1, 1),
#     Inst().save('b')
# ]
#
# pattern = ParamSet(Parameter(TraitMatcher(IntTrait), 'a'),
#                       Parameter(TraitMatcher(StrTrait), 'b', '+'),
#                       named_params={'c': Parameter(TraitMatcher(NumTrait), 'c', default=py_value(0)),
#                                     'd': Parameter(TraitMatcher(NumTrait), 'd')})
# # pattern.parameters = (Parameter(TraitMatcher(IntTrait), 'a'),
# #                       Parameter(TraitMatcher(StrTrait), 'b', '+'))
# # pattern.vm = prog
#
# args = Args(py_value(1), py_value('two'),
#             named_arguments={'d': py_value(Fraction(1, 2)), 'c': py_value(55)})
# args = Args(py_value("one"), py_value("two"), py_value("three"),
#             named_arguments={'a': py_value(-1), 'd': py_value(Fraction(2, 3)), 'b': py_value('string')})
#
# print(pattern.match_zip(args))
# exit()


if mode in ('test', 'script'):
    repeats = 1
    t = timeit.timeit(lambda: execute_script(script_path), number=repeats) / repeats
    print('\n\n***************************************\n')
    print(t)
    # execute_script(script_path)
