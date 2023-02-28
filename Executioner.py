from baseconvert import base
import math
from DataStructures import *
from Abstract_Syntax_Tree import Tokenizer, AST
from DataStructures import *


class Executor:
    root: Function
    context: Function
    output: Value

    def __init__(self, script: str):
        tokenizer = Tokenizer(script)
        ast = AST(tokenizer)
        self.root = Function(block=ast.block)
        for key, builtin in BuiltIns.items():
            self.root.add_option(Pattern(Parameter(key)), builtin)
        self.context = self.root

        def execute(fn: Function) -> Value:
            for statement in fn.block.statements:
                expr = Expression(statement.nodes)
                result = expr.evaluate(context=fn)
                if isinstance(result, Action):
                    match result.action:
                        case 'return':
                            return Value(result.value)
                        case 'assign':
                            fn.assign_option(result.pattern, Value(result.value))

        Function.exec = execute

    def execute_root(self):
        # self.spot.chroot()
        print(repr(self.spot.root[0].block.statements))
        # return self.root.execute()


if __name__ == '__main__':
    script_path = "test_script.ss"
    with open(script_path) as f:
        script_string = f.read()
    Boris = Executor(script_string)
    Boris.spot.chroot()
    Boris.execute_root()