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
    script_path = "test_script.pili"
    with open(script_path) as f:
        script_string = f.read()

    tokenizer = Tokenizer(script_string)
    ast = AST(tokenizer)
    root = Function(ListPatt(Parameter('main')), FuncBlock(ast.block))
    for key, builtin in BuiltIns.items():
        root.add_option(ListPatt(Parameter(key)), Value(builtin))
    Context.root = root
    Context.env = root
    output = root.deref('main')
    print(output)