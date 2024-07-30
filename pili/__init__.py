from pili import state
from pili.state import Op, BuiltIns
from pili.lexer import Tokenizer
from pili.abstract_syntax_tree import AST
from pili.runtime import GlobalFrame, Closure, Frame
from pili.utils import PiliException
from pili.builtins import base, operators

print(f'loading {__name__}.py')

__all__ = ['pili', 'pili_shell', 'run',
           'builtins',
           'runtime', 'lexer', 'abstract_syntax_tree', 'utils', 'interpreter', 'syntax',
           'state']

for sym, op in Op.items():
    if sym in BuiltIns:
        pass
    BuiltIns[sym] = op.fn
for k, v in BuiltIns.items():
    if hasattr(v, 'name') and v.name is None:
        v.name = k
builtin_namespace = GlobalFrame(BuiltIns)  # Closure(Syntax.Block([]), bindings=BuiltIns)
state.root = builtin_namespace
# BuiltIns['root'] = builtin_namespace
state.push(builtin_namespace)

def pili(code: str):
    """ parse and execute an expression (or several)
        :returns evaluation of (last) expression
    """
    return run(script=code, closure=False)

def run(*, path: str = None, script: str = None, closure=True):
    if path and script or path is script is None:
        raise ValueError("Specify either file path or script, but not both.")
    if path:
        with open(path) as f:
            script = f.read()
    orig = state.source_path, state.source_code
    state.source_path = path
    state.source_code = script
    block = AST(Tokenizer(script)).block
    if closure:
        block = Closure(block)
    try:
        return block.execute()
    except PiliException as e:
        return e
    except Exception as e:
        e.add_note(state.get_trace() + f"\n> Line {state.line}: python exception")
        return e
    finally:
        state.source_path, state.source_code = orig

def pili_shell():
    state.push(Frame(state.env))
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
            output = pili(code.strip())
            if output != BuiltIns['blank']:
                print(output)
        except Exception as e:
            print("Exception: ", e, '\n***')
            raise e


run(path='pili/builtins/standard.pili', closure=False)
from .builtins import standard

if __name__ == '__main__':
    run(path='pili/test.pili')
