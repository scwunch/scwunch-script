#2023/Mar/5 #programming

# Pili
Pili is the successor to [[22D28 ryanscript 2.1 everything is still a function — refined|ryanscript and scwunch-script]].

## Introduction and Overview
The core data structure in Pili is the "Option".  Functions have one or more *options* that are chosen by passing arguments that match a certain pattern, and yield a value.  The name "pili" is a Filipino word meaning "choose".  

The goal of this language is not to be a "good" language (I'm not capable of that).  Rather, it is an experiment in exploring the viability of an interpreted language revolving around this idea of options and pattern matching, rather than objects, keys, and functions.

## Primary Data Structures
### Values
Values are what they sound like, boolean, string, number etc.  Values have embedded types associated with them.  The value types are: 
- `bool` boolean
- `int` integer
- `float` float
- `str` string
- `none` none
- `type` type
- `fn` function

There are also two more "hidden types" used by the interpreter, but not readily accessible to the user: *pattern*, and *name*.  These value types are used by built-in operators like ` =` and `.`.

### Patterns
Pattern matching is a core feature of Pili.  It is how options are selected and therefore how functions are called.  Options (which act as variables) do not have names, they have *patterns*.  Though the simplest pattern consists of one named parameter matching any value.

A pattern is like a regular expression for a list of argument values.  A pattern consists of zero or more parameters.  A parameter matches a value based on type or value and an optional embedded function or an optional name.

A Parameter is or has at least one of the following: 
- a name, ie, a string without quotes (eg `myVar`) 
	- *Note: this only applies when the string is not a reserved keyword and does not already refer to a type*
- a value (eg, `5` or `"five"` or `true`)
- a type or types (eg, `str` or `int|float`)
- a restricted type (eg, `int < 4`) 

> [!code] Examples
> `my_var` only matches the name `my_var` or the string value `"my_var"`
> 
> `int` matches any integer value
> 
> `0` matches the value 0
> 
> `str my_var` matches any string value (and binds the name `my_var` if it is passed to a function)
> 
> `int|bool|float` matches any number or boolean value
> 
> `int>0` matches positive integers (this is a shorthand for `int[is_pos]` where `is_pos` is a boolean function)
> 
> ``str[`\w+`]`` matches whole-word strings (the argument is a regex)

### Function and Options
Functions are like objects or dictionaries — they contain arbitrary labelled data (options).  But the way that data is accessed by "calling" the function with arguments.  Functions exist within a certain scope (namespace) and may also be connected to a prototype, or prototype chain similar to javascript.

Options act like variables, properties, or methods.  Or an option may be thought of as a variant or overload of a Function.  An option consists of a Pattern and yields one of:
- a Value
- a block of code (which in turn will return a Value, which could be a Function)
- a native (python) function, which also returns a Value

## Variables and Function Assignment
Options are created and assigned with three operators:
- ` =` for assigning a *value* (right-hand-side is evaluated before assignment)
- `:` for assigning a block of code (right-hand-side is saved for evaluation later)
- `:=` for assigning an alias

```python
greeting = "Hello " + "world"
# immediately calculates the right hand side and assigns the VALUE to the option "greeting"
# if the name "greeting" is not already an option of the root function, it is added automatically

greeting[1] = "hello world"
# immediately calculates the right hand side and assigns the VALUE to the option [1] in the function "greeting".  
# if greeting is not an option of root, it will be added.  If it is an option, but is not a Function, an error occurs.

greeting[str who]: "hello " + who
# does not evaluate the right-hand side; only when the [str who] option is selected will it return a value

greeting[str who]: 
	return "hello " + who
# equivalent to above

greeting = 
	str who: "hello " + who
# function options can be defined within the scope of the function, or outside of it

```

### Pointers and Aliases
As a side of effect of how functions and variables work, a pointer can be created with the syntax:
```python
myVar = 5
myPointer: myVar
not_a_pointer = myVar

myPointer == myVar and not_a_pointer == myVar
> True

myVar = 10

myPointer == myVar
> True
not_a_pointer = myVar
> False
myPointer
> 10
not_a_pointer
> 5
```

Aliases are defined with the walrus operator `:=`
```python
myVar = 5
myPointer: myVar
myAlias := myVar

myVar += 1
myAlias
> 6
myAlias += 1
myVar
> 7
myPointer
> 7
myPointer += 1
> ERROR
```
This is analogous to hard-linking in the file system, whereas pointers are more like soft (symbolic) links.

## Function Options
Any one function may have numerous versions to be called on different types of data, different numbers of arguments, or in different circumstances.  In some languages, this is called "function overloading".  In ryanscript, these different version of the function are called **options** and they are defined dynamically, inside and outside a functions definition, while the function is running or not.

### Defining Options
There are two main ways to define function options.  The first is by using using match statements from within a function, and the second is by defining a function with a parameter set from outside the function definition. 

For example, the following two code blocks are functionally equivalent.
```python
myFunction =
	0: "zero"
	"string_key": 5
	string_key: 5       # equivalent to above
	int num: 3 + num    
	int<0: "negative integers"
```

```python
myFunction[0]: "zero"
myFunction["string_key"]: 5
myFunction.string_key: 5
myFunction[int num]: 3 + num
myFunction[int<0]: "negative integers"
```

There is a subtle difference between these two methods however.  When assigning values via ` =`, right-hand-side expressions are evaluated in whatever context the line of code exists in.  That is:
```python
myFunction = 
	some_option = 5
	another_option = some_option + 10

myFunction.another_option
> 15

my_Function.some_option = 5
myFunction.another_option = some_option + 10
> ERROR: 'some_option' option not found
```

However, this is possible when defining via the `:` operator, since those blocks of code are *always* executed within the context/namespace of the function on which they are defined, regardless of the scope in which that definition was created or executed.

```python
myFunction.another_option: some_option + 10
myFunction.another_option
> 15
```

### Calling Options
Function can be called by passing a list of arguments (Values).  If the expression is in the scope of the function you wish to call, the arguments may be passed by enclosing the value(s) in square brackets.  So, for example, to call an option that was defined as `5: "five"` in the same scope, the simple expression `[5]` will work, as well `[2+3]` or any equivalent expression.  

An option whose pattern consists of a single parameter with a name or string value is a **named option**.  Named options can be called the same way (`foo["bar"]` or `["bar"]`) or they can drop the quotes and brackets and sit naked, as long as the string doesn't conflict with other syntactic elements in Pili, such as whitespace, operator characters, or reserved keywords.  Therefore the following are equivalent:
```python
["bar"] == bar
foo["bar"] == foo.bar
# but these are NOT equivalent:
["and"] != and                  # interpreted as Operator
["hello world"] != hello world  # two names with no operator
["tic-tac-toe"] != tic-tac-toe  # read as tic MINUS tac MINUS toe
```

#### Reverse Dot Calls
By default, the `.` operator selects and calls a named option.  However, if the value is not a function, or no matching named option exists, it may also call a named option of the current scope with the first function or value as an *argument*.  This is only true for *named options* using the `.` syntax; the `[""]` syntax still works normally.  The following are equivalent:
```python
my_name = "Ryan"
len[my_name]
> 4
my_name.len
> 4
my_name["len"]
> ERROR: my_name is not a function
```

If the named function requires more than one argument, they may be passed as if the function was an option of the first argument.
```python
match[my_name, `\w+`]
> True
my_name.match[`\w+`]
> True
```

A Function with a named option that is the same as a named option on the present scope will always take precedence.
```python
Dog = 
	name = "Rover"
Dog.len
> 1
Dog = 
	name = "Rover"
	len = "13cm"
len[Dog]
> 2
Dog.len
> "13cm"
```

### Virtual Options or Dot Options
But what if you *did* want to change the default behaviour of `len[Dog]`?

Virtual options, also called "dot options" may be defined using matching statements whose key begins with a `.` character.  This statement technically does not define a option on the function.  It actually defines an option on the function represented by the key following the dot.

This strategy can be used to add more specific options to built-in functions.
```python
container = 
	data = {}
	name = "I'm a container"
	.len: len[data]
len[container]  # or container.len
> 0

Duration[int hr, int min]:
	is_afternoon: hr >= 12
	is_morning: hr < 12
	.["+"][other]: 
		new_hr = hr + other.hr
		new_min = min + other.min
		if new_min >= 60
			new_min -= 60
			new_hr += 1
		return Duration[new_hr % 24, new_min]
dur = Duration[1, 59] + Duration[0, 15]
print str[dur.hr] + ":" str[dur.min]
> "2:14"
```

For example (albeit a useless example):
```python
is_logical = 
	bool = True
	int[{v: v == 0 or v == 1}] = True
	float[{v: v == 0. or v == 1.}] = True
	str[`true|false`] = True
	any = False

is_logical['true']
> True
is_logical['yes']
> False

is_logical.len
> 5

my_log_func =
	data = True
	.is_logical = True

is_logical.len
> 6
my_log_func.len
> 1
is_logical[my_log_func]
> True


my_proto:
	.string: ...
	.['+'][int|float arg]: ...
	name: ...
	any name: ...
	name[pattern]: ...
	any: ...
	.string.len: ...

case ['.', fn, patt*]
case 
```

#### Dot Calls
Dot Options (or any function actually) may be called with alternative syntax.
`myCat.numLegs` is equivalent to `numLegs(myCat)`
The parentheses are unneeded only if `numLegs` takes a single argument.  If `numLegs` required more than one argument, the following would be the case: 
`myCat.numLegs(other, args)` is equivalent to `numLegs(myCat, other, args)`
If you are in the scope of the first argument, in this case `myCat`, then `myCat` may be omitted.  A dot option without a specified first argument defaults to the current scope.  Eg, `.numLegs` or `.numLegs(other, args)`.

> [!caution] For Consideration
> Again, as above, I may change this syntax in the future to another symbol.  Or I may not.

## Function Parameters
When defining an option, whether through a matching statement or classic style, each key in the match phrase (before the colon, or within square brackets) is also considered a parameter of the function.  

> [!Danger] **OH — I JUST HAD AN IDEA:** 
> What if each key actually has two parts, the matching part and the (quoteless string) name.  
> 
> If the match part is omitted, it defaults to the string value of the name.
> If the name part is omitted, then the key simply won't have a name.
> If only one is included and it is ambiguous which one it is (eg, `int`)... then I guess it should throw an error?

So anyway, each function parameter is also, of course, an option.  But these options are defined immediately **regardless of whether the option is defined with `:` or `:=`**.  For example: 
```ryanscript
coolFunc[string input: 345, number n: 43]:
	return input[n]
coolFunc["input"]
> 345
coolFunc[45]
> 43
coolFunc["n"]
> 43
coolFunc["abcdefg", 3]
> "c"
```

Ok, that's super weird, and not sure when that might be useful... but sure okay.

I guess I could avoid this weirdness by only defining the parameters options at the time the function is called (which could be immediately with the walrus operator `:=`)

## Other Things
- Classes, types, prototypes, inheritance
- type tree and option tree
- matching implementation
	- how to do determine the class of a function?  Track ancestry of function?
- option flags (eg `const`, `alias`, type hints, something else?)
- idea: side-effects as values


> [!important] Important
> When an option is defined via matching statement, and the interpreter reaches that point in the code, it adds the option to the *currently running copy* (not the original function template).  So that means that the option can now be called within that context and, if the function is saved and assigned a key, then the option can be called from that key.  But if the original function is run again, it will not have that option.
> 
> Why is this important?  Example: 
> ```
> myStrFunction[string input]:
> 	index: 0
> 	lines: input.split('\n')
> 	return lines[index]
> print[myStrFunction["index"]]
> ```
> This program will print the string "index" — it will not return the value 0.
> 
> However, the following program will, since the function is executed immediately:
> ```
> myStrFunction =
> 	index: 0
> print[myStrFunction["index"]]
> ```




***

## Implementation
### Python Classes
- Syntax
	- Node: Node
		- Token: Syntax
		- Statement: Token, Line
			- ... other statement types
		- Block: Statement
	- Line: Token, ==Tokenizer==
- Builder: Line
	- Tokenizer: 
	- AST: Tokenizer
- Functions.py
	- Value: type, value
		- Function: Option
	- Type
	- Option
	- Statement
		- Expression => value
			- mathological
			- conditional
			- loop 
		- Matching Statement => option: match_pattern, set_operator, expression=>value
		- Command => effect
			- return
			- break
			- continue
- Main.py
	- Context: call stack
***
I just realized now that one reason I was struggling to know when functions should be cloned or not is that I failed to recognize there is a hugely significant conceptual difference between these two.  It is the difference between *function template* and *called-and-running function*, equivalent to the difference between *class* and *instance*.  Or abstract and concrete.

Next challenge: what to name these things?
- class => instance
- abstract => concrete
- potential => actual
- function => object
- prototype => function
- pre-execution function => executed function
- Option => Function
- type => example
- template => document
- description => sample
- species => individual
-   Category => Item
-   Genre => Work
-   Brand => Product
-   Model => Instance
-   Style => Element
-   Variety => Specimen
-   Form => Case
-   Kind => Representative
-   Flavor => Sample
-   Variant => Version
-   Mode => Configuration
-   Family => Member
-   Group => Individual
-   Class => Case study
-   Make => Model number

Actually, there are *three* concepts here: one extra one in the middle for transitioning, the "currently running function" or "the process of instantiation".  That one requires a few more properties

What properties do they need?
- abstract class:
	- just a block of code... and a method for instantiation / concretization / cloning / actualization / objectification
	- maybe paramater slots for args?
- running function:
	- args
	- environment (including prototype)
	- return_value (default to self)
- instance function
	- just options

```python
class Function:
	prototype = Function                   1,2,3
    args: list[Value]                        2
    options: list[Option]                    2,3
    named_options: dict[str, Option]       
    block: Block                           1,2
    env: Function                          1,2,3
	exec: any                                2
    return_value: Value                      2
    is_null: bool                            2
    init: any                              1,2
    def __init__(self, opt_pattern: Pattern = None,
                     opt_value: Function = None,
                     options: dict[Pattern, Function] = None,
                     block: Block = None,
                     prototype: Function = None,
                     env: Function = Context.env,
                     value: Value = None,
                     is_null=False): ...

    def add_option            2,3
    def assign_option         2,3
    def index_of              2,3
    def select                2,3
    def call                1
    def deref                 2,3
    def execute             1
    def init                1
    def clone               1
```

What if we go with the "Option -> Function" pair, expanding the Option class with a few properties from the Function class?  Functions are like nouns, options are like verbs.

```
class Option:
    pattern: Pattern
    function: Function
    value: Value
    def __init__(self, params: Pattern | Parameter | str, fn_or_val: Function | Value = None): ..
    def is_null(self) -> bool: ...
    def not_null(self) -> bool: ...
```


#### Statement Types
- Expression (mathological)
	- statements
	- operators
- Conditional
	- antecedent
	- consequent
- For..in Loop
- While Loop
- Option (declaration/assignment)
	- match part
	- expression

Block(Node)
- Statement(Node)
	- Token(Node)
	- Statement(Node)
	- List(Node)
	- Block(Node)

#### Value Types: the types that go into and out of functions and statements
none = 'none'  
Boolean = 'bool'  
Integer = 'int'  
Float = 'float'  
String = 'str'  
Function = 'fn'  
List = 'list'
Option?
Block? — just made it a function

#### No more buffer types
Buffer Types: the structures that exist between values
- Parameter
- Option
- Block

### Assignment
- Types of assignment operators
	- ` =` — set value
	- `:` — set function
	- `:=` — set alias
- Left-hand-side assignee possibilities:
	- *name* token (eg `my_var = 5`)
		- `name:my_var = 5`
	- *type and name pattern*  (eg, `str text = "hello world"`)
		- `str &name text = "hello world"`
	- *function with parameters* (eg `foo[str arg1, arg 2]: ...`)
		- `foo &pattern [str arg1, arg2] : ...`
	- *option expression* (eg `foo[0] = "bar"`)
		- `foo &key [0] = "bar"`
- so I guess, whenever the assignment operator is `:`  then any `[list]` to the immediate left of the operator should be interpreted as param pattern
- otherwise, such a `[list]` should be interpreted as an argument list (like normal) but should *not* be executed.  Rather, anothe operator (similar to `&name`) should be inserted before the list, to transform the function and list into an option, whose function/value can be set
- interpreting left side of `:=` or` =`  op:
	- entire set of nodes interpreted as an `arg_list` (if not `List` already)
	- for each `arg` in list:
		- last_node is one of:
			- 
- what about the lhs of `?` and `??` ops?



```
greet[str whom]: "hello " + whom

greet[str[`^world.*`]] = "hello world!"

print greet["world"]
```

### Operators
- Logical: 
	- and, or, not
- Comparison:
	- ` ==, <, >, <=, >=, ~`
- Mathematical:
	- `+-/*%`
- Assignment: 
	- `:, =, :=, +=, *=, /=, etc`
- Function Ops:
	- `.` select operator / pseudo-method caller
	- `[]` select and call operator
- Unary Operators:
	- `?` option existence checker
	- `#` function class generator
	- `@` reference generator (for mutation)
- Other opeators:
	- `??` nullish coalescing
	- `.?` nullish option selection
	- `&name` parameter generator function (takes a type or pattern and a name)

#### Expression Tree Builder
imagine a string of numbers and letters: 
eg "a2b5c1d9e"

Now arrange this into a tree structure like:
- 1
	- a2b5c
	- d9e
->>>>
- 1
	- 2
		- a
		- b5c
	- 9
		- d
		- e

#### Dot Operator
Hierarchy of dot call:
- `function.option[args]`


```

```

***
### Testing
- make some more exception classes
- implement a print_trace method in Context object.
- code to test:
	- assignment
	- each individual operator
	- each builtin function
	- making and running custom functions
	- prototype inheritance
	- option selection and hierarchy

### Pattern Matching
I need to re-implement the whole pattern generation thing.  I think I may actually go back to using that hidden `&name` operator again.  It gives more flexibility.

Example patterns:
```python
foo = ...
foo.bar = ...
foo[0] = ...
foo[0, ''] = ...
foo: ...
int foo: ...
int foo, str bar: ...
int < 0: ...
str[4]: ...
int|str foo: ...
str[`\d+`] snum | int inum: ...
bool|"true"|"false" foo: ...
$List foo, bar?: ...
[a, b], int c: ...
$Coord[x==0]
```

What are the elements in here that might be ambiguous?
- names
	- put in a virtual static operator before every name that precedes a `:` or ` =`.
- literals (like `"string", 0, True`)
	- encase in brackets maybe?
- other Function names referring to interfaces or prototypes
	- maybe another prefix operator like `#` or `@`.

- **Pattern** ::= Element | (list of Parameters)
	- **Parameter** ::= Negation (frozenset of Elements) Name, Quantifier
		- **Negation** ::= `!` negates the match, eg `!str` matches any value *except* string values
		- **Element** ::= (value|type|class) with Guard
			- **Guard** ::= an expression or function that should return a boolean
				- in the case of builtin types, the `.[` operator returns an element with a variety of different Guards
				- in the case of Class: the `.[` operator takes an expression and returns an element with a guard such that evaluating that expression
		- **Quantifier** ::= `+` (at least one) or `*` (any number) or `?` (0 or 1)

**Class Guard**
- arguments: a class object, `.[` operator, expression
- when matching an arg to the element:
```python
if arg.prototype ~ element.prototype:
	Context.env = arg
	evaluate(expression)
else:
	return "no match"
```


- Pattern
	- name
	- guard
- Union(Pattern)
	- frozenset(Pattern)
- Value(Pattern)
	- value
- BasicType(Pattern)
	- basic_type
- Class(Pattern)
	- prototype
- List(Pattern)
	- parameters: tuple(Parameter)
- Parameter:
	- inverse
	- Pattern
	- quantifier



```python
# pattern for my Option Class
PattP = ...
ValP = 
OptP = $Option[pattern ~ PattP and (value~Value ?? block~Block ?? fn~lambda)]

OptP = $Option[
			   pattern ~ PattP
			   or(
				   value ~ Value
				   block ~ Block
				   fn ~ lambda
				)
]

# simple interface
Option[@Pattern pattern, @Value value]:
	pass

# explicit interface
Option = 
	$Pattern pattern
	$Value value
	[$Pattern patt, $Value val]:
		pattern = patt
		value = val
```


The `@` prefix operator takes a Function and returns a Pattern that matches
```
my_dog = 
	int num_legs = 4
	str name = "Rover"

@my_dog 
# is equivalent to:
[num_leg ~ int and name ~ str]

Dog[int num_legs, str name]:
	bark: do something

@Dog

```