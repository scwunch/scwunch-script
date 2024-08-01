#!/usr/bin/env python3
import sys
import timeit
from pili import run, pili_shell
# import state
# from utils import PiliException
# from lexer import Tokenizer
# from abstract_syntax_tree import AST
# from init_builtins import *

print('Start main.py')

# def init_state():
#     for sym, op in Op.items():
#         if sym in BuiltIns:
#             pass
#         BuiltIns[sym] = op.fn
#     for k, v in BuiltIns.items():
#         if hasattr(v, 'name') and v.name is None:
#             v.name = k
#     builtin_namespace = GlobalFrame(BuiltIns)  # Closure(Syntax.Block([]), bindings=BuiltIns)
#     state.root = builtin_namespace
#     state.push(-1, builtin_namespace)
#
# def pili(code: str):
#     """ parse and execute an expression (or several)
#         :returns evaluation of (last) expression
#     """
#     return run(script=code, closure=False)
#
# def run(*, path: str = None, script: str = None, closure=True):
#     if path and script or path is script is None:
#         raise ValueError("Specify either file path or script, but not both.")
#     if path:
#         with open(path) as f:
#             script = f.read()
#     orig = state.source_path, state.source_code
#     state.source_path = path
#     state.source_code = script
#     block = AST(Tokenizer(script)).block
#     if closure:
#         block = Closure(block)
#     try:
#         return block.execute()
#     except PiliException as e:
#         return e
#     except Exception as e:
#         e.add_note(state.get_trace() + f"\n> Line {state.line}: python exception")
#         return e
#     finally:
#         state.source_path, state.source_code = orig
#
# def pili_shell():
#     state.push(0, Frame(state.env))
#     while True:
#         code = ''
#         next_line = input('> ')
#         if next_line.endswith(' '):
#             while next_line:
#                 code += next_line + '\n'
#                 next_line = input('  ')
#         else:
#             code = next_line
#         try:
#             output = pili(code.strip())
#             if output != BuiltIns['blank']:
#                 print(output)
#         except Exception as e:
#             print("Exception: ", e, '\n***')
#             raise e


if __name__ == '__main__':
    # print("initialize global namespace...")
    # init_state()
    # print('Loading stdlib.pili')
    # e = run(path='lib.pili', closure=False)
    # if isinstance(e, Exception):
    #     raise e

    if len(sys.argv) == 1 and sys.executable == '/usr/bin/python3':  # test if running in console; pycharm executable is python3.10
        mode = 'shell'
        pili_shell()
    else:
        if len(sys.argv) == 2:
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
            # script_path = 'advent_2.pili'
            print('(test mode) running script', script_path, '...')

        output = None
        def get_output():
            global output
            output = run(path=script_path)
        repeats = 1
        t = timeit.timeit(get_output, number=repeats) / repeats
        print(f'{script_path} finished with output: ', output)
        print('\n\n***************************************\n')
        print(t*1000, 'ms')

        if isinstance(output, Exception):
            raise output
