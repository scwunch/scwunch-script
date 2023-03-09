import DataStructures
from DataStructures import Function
from Expressions import Context
from Abstract_Syntax_Tree import Tokenizer, AST
from BuiltIns import *
import StaticOperators

# Function.execute = execute

# context: Function
# context = Function()
if __name__ == "__main__":
    script_path = "test_script.ss"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    ast = AST(tokenizer)
    builtins_env = Function(env=None)
    for key, builtin in BuiltIns.items():
        builtins_env.add_option(ListPatt(Parameter(key)), Value(builtin))
    root = Function(ListPatt(), ast.block, env=builtins_env)
    Context.root = root
    Context.env = root
    output = root.call([])
    print(output)