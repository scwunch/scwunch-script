import Env
from Env import BuiltIns, Op
from tables import *

print(f"loading module: {__name__} ...")

def default_op_fn(*args):
    raise Env.OperatorErr(f"Line {Context.line}: Operator has no function.")


default_op_fn = Function({Parameter(AnyMatcher(), None, "*"): default_op_fn})

class Operator:
    fn: Function = default_op_fn
    def __init__(self,
                 text,
                 fn=None,
                 prefix=None, postfix=None, binop=None, ternary=None,
                 associativity='left',
                 chainable=False):
        Op[text] = self
        self.text = text
        # self.precedence = precedence
        if fn:
            if not fn.name:
                fn.name = text
            BuiltIns[text] = fn
            self.fn = fn
        self.associativity = associativity  # 'right' if 'right' in flags else 'left'
        self.prefix = prefix  # 'prefix' in flags
        self.postfix = postfix  # 'postfix' in flags
        self.binop = binop  # 'binop' in flags
        # self.ternary = ternary
        self.chainable = chainable

        assert self.binop or self.prefix or self.postfix

    def eval_args(self, *terms) -> Args:
        # terms: list[Node]
        return Args(*(t.evaluate() for t in terms))
        raise NotImplementedError('Operator.prepare_args not implemented')

    def __repr__(self):
        return self.text


Operator(';', binop=1)
Operator(':', binop=2, associativity='right')
Operator('=', binop=2, associativity='right')

for op in ('+', '-', '*', '/', '//', '**', '%', '&', '|'):
    Operator(op+'=', binop=2, associativity='right')

Operator('??=', binop=2, associativity='right')
Operator('=>', binop=2)
Operator(',', binop=2, postfix=2, chainable=True)
# Operator('if', binop=3, ternary='else')
Operator('??', binop=4)
Operator('or', binop=5)
Operator('and', binop=6, chainable=True)
Operator('not', prefix=7)
Operator('in', binop=8)
Operator('==', binop=9)
Operator('!=', binop=9)
Operator('~', binop=9, chainable=False)
Operator('!~', binop=9, chainable=False)
Operator('is', binop=9, chainable=False)
Operator('is not', binop=9, chainable=False)
Operator('|', binop=10, chainable=True)
Operator('<', binop=11, chainable=True)
Operator('>', binop=11, chainable=True)
Operator('<=', binop=11, chainable=True)
Operator('>=', binop=11, chainable=True)
Operator('+', binop=12, prefix=14, postfix=3)
Operator('-', binop=12, chainable=False, prefix=14)
Operator('*', binop=13, postfix=3)
Operator('/', binop=13, chainable=False)
Operator('//', binop=13, chainable=False)
Operator('%', binop=13, chainable=False)
Operator('**', binop=14, chainable=False, associativity='right')
Operator('^', binop=14, chainable=False, associativity='right')
Operator('?', postfix=15)
Operator('has', binop=15, prefix=15)
Operator('&', binop=15)
Operator('@', binop=3, prefix=16)
Operator('.', binop=16, prefix=16)
Operator('.?', binop=16, prefix=16)
Operator('..', binop=16, prefix=16)

