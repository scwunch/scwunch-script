#!/usr/bin/env python3
import sys
import timeit
from Abstract_Syntax_Tree import Tokenizer, AST
from BuiltIns import *
import operators

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
    # script_path = 'Tables.pili'
    # script_path = 'pili_interpreter.pili'
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

for k, v in BuiltIns.items():
    if hasattr(v, 'name') and v.name is None:
        v.name = k
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
                    output_string = BuiltIns['string'].call(output).value
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


if __name__ == "__main__" and False:
    print("Test echo-generator...")

    ord = {
        0: 'zeroth',
        1: 'first',
        2: 'second',
        3: 'third',
        11: '11th',
        12: '12th',
        13: '13th'
    }
    def ordinal(n: int):
        if n in ord:
            return ord[n]
        match n % 10:
            case 1:
                return f"{n}st"
            case 2:
                return f"{n}nd"
            case 3:
                return f"{n}rd"
            case _:
                return f"{n}th"

    def permutation_generator(items):
        # base case: if the list is empty or has one item, yield it as is
        if len(items) <= 1:
            yield items
        else:
            # loop through each item in the list
            for i in range(len(items)):
                # swap the item with the first item
                items[0], items[i] = items[i], items[0]
                # yield the permutations of the remaining items
                for perm in permutation_generator(items[1:]):
                    # prepend the first item to each permutation
                    yield [items[0]] + perm

    perm_gen = permutation_generator([1, 2, 3])
    for perm in perm_gen:
        print(perm)


    def permutation_closure(items: list):
        idx = 0
        rest_of_perms = None  # permutation_closure(items[1:])
        def next_perm():
            nonlocal idx
            nonlocal rest_of_perms
            if idx == len(items):
                return 'done'
            if len(items) <= 1:
                idx += 1
                return items
            items[0], items[idx] = items[idx], items[0]
            if rest_of_perms is None:
                rest_of_perms = permutation_closure(items[1:])
            next = rest_of_perms()
            if next == 'done':
                idx += 1
                rest_of_perms = None
                return next_perm()
            return [items[0], *next]

        return next_perm

    print('\npermutation closure')
    next_perm = permutation_closure([1, 2, 3])
    for i in range(17):
        print(next_perm())

    exit()





    def echo_generator():
        call_count = 0
        while call_count < 20:
            call_count += 1
            print(f"this is the {ordinal(call_count)} output of this generator.")
            yield call_count
            print(f"... and the second half of the {ordinal(call_count)} output of this generator.")
            yield call_count + 0.5


    def echo_closure():
        call_count = 0
        flip_flop = 'flop'
        def inner():
            nonlocal call_count
            nonlocal flip_flop
            if flip_flop == 'flop':
                call_count += 1
                print(f"This is the {ordinal(call_count)} time you've called this closure.")
                flip_flop = 'flip'
                return call_count
            else:
                call_count += 0
                print(f"... and the flip-side of the {ordinal(call_count)} call.")
                flip_flop = 'flop'
                return call_count

        return inner

    gen = echo_generator()
    for i in range(25):
        next(gen)

    ecc = echo_closure()
    for i in range(25):
        ecc()



    exit()


    import random
    import string

    NUM_NAMES = 100
    REPEATS = 500
    # Define all possible characters
    characters = string.ascii_letters + string.digits


    def generate_random_string(length):
        # Create random string
        random_string = ''.join(random.choice(characters) for _ in range(length))
        # Return the random string
        return random_string


    name_gen = (random.randint(-1283471263847, 234209452938745) for x in range(NUM_NAMES))
    name_dict = {}
    i = 0
    for n in name_gen:
        name_dict[n] = i
    name_tuple = tuple(name_dict.keys())
    name_list = list(name_tuple)

    name_to_find = random.randint(-1283471263847, 234209452938745)

    print()


    def find(needle: str, haystack: dict | tuple | list):
        return needle in haystack


    t_dict = round(REPEATS / timeit.timeit(lambda: name_to_find in name_dict, number=REPEATS) / 1000)
    t_tuple = round(REPEATS / timeit.timeit(lambda: name_to_find in name_tuple, number=REPEATS) / 1000)
    t_list = round(REPEATS / timeit.timeit(lambda: name_to_find in name_list, number=REPEATS) / 1000)

    times = t_dict, t_tuple, t_list
    ids = (x for x in ('t_dict  ', 't_tuple ', 't_list  '))


    def bar(n):
        return f"{ids.__next__()}: {str(n)} {'█' * round(n / 300)}"


    print("\n".join(list(map(bar, times))))




    print("*********************************")
    NAME_LENGTH = 1
    NUM_NAMES = 2
    REPEATS = 500
    # Define all possible characters
    characters = string.ascii_letters + string.digits
    def generate_random_string(length):
        # Create random string
        random_string = ''.join(random.choice(characters) for _ in range(length))
        # Return the random string
        return random_string

    name_gen = (generate_random_string(NAME_LENGTH) for x in range(NUM_NAMES))
    name_dict = {}
    i = 0
    for n in name_gen:
        name_dict[n] = i
    name_tuple = tuple(name_dict.keys())
    name_list = list(name_tuple)

    name_to_find = generate_random_string(NAME_LENGTH)

    print()

    def find(needle: str, haystack: dict | tuple | list):
        return needle in haystack
    t_dict = round(REPEATS / timeit.timeit(lambda: name_to_find in name_dict, number=REPEATS) / 1000)
    t_tuple = round(REPEATS / timeit.timeit(lambda: name_to_find in name_tuple, number=REPEATS) / 1000)
    t_list = round(REPEATS / timeit.timeit(lambda: name_to_find in name_list, number=REPEATS) / 1000)

    times = t_dict, t_tuple, t_list
    ids = (x for x in ('t_dict  ', 't_tuple ', 't_list  '))
    def bar(n):
        return ids.__next__() + ": " + "█"*round(n/300)

    print("\n".join(list(map(bar, times))))

    exit()


if mode in ('test', 'script'):
    repeats = 1
    t = timeit.timeit(lambda: execute_script(script_path), number=repeats) / repeats
    print('\n\n***************************************\n')
    print(t)
    # execute_script(script_path)