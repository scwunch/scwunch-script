#!/usr/bin/env python3
import sys
import timeit
from pili import run, pili_shell

print('Start main.py')


if __name__ == '__main__':
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
            # script_path = 'advent.pili'
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
