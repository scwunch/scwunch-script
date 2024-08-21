---
created: 2023-03-05T00:00:00
---
#2023/Mar/5 #programming

Pili is the successor to [[22D28 ryanscript 2.1 everything is still a function — refined|ryanscript and scwunch-script]].

## Introduction and Overview
The core distinctive of Pili is the **map** data structure which combines the concepts of functions, hashmaps, and namespaces.  Values are passed to maps (as arguments), and Pili uses a combination of hashing and a pattern-matching algorithm to "select" the matching option, and return it's value or call the associated function.  The name "pili" is a Filipino word meaning "choose".  

Pili attempts to simultaneously value ergonomics, readability, conciseness, and expressivity. The goal of the language is for it to be a joy to read and write.  Pili opts to use newlines and tabs to demarcate statements and code blocks, rather than semicolons and curly braces.  Extraneous words and symbols are generally avoided.  The major elements of the type system are all denoted by English words (class, slot, formula, etc).


## Basic Types
Types are also called classes or traits.  Every value in Pili is a record of exactly one class.  Each class "inherits" from 0 or more traits.  By convention, class names are always capitalized, and trait names are usually capitalized (except all the builtin traits are lowercase).  All other names (singletons, variables, functions, etc) all follow snake_casing convention.  

### Basic Types
Here are some of the most basic built in classes: 
- `Blank` class has only one value: `blank`.  This value is falsy and treated specially in some situations, eg, the `.?` and `??` operators.
- `Boolean` has only two records:  `true` and `false`
- `Integer` integer number: arbitrary magnitude signed integer
- `Rational` rational number: exact representation any number that can be defined with integer numerator and denominator.  
- `Float` floating point number
- `String` string of text

#### Numeric types
The builtin number types are usually referred to by their traits: `bool`, `int`, `ratio`, `float`.   ("ratio" is short for "rational").  The `num` trait is inherited by all numeric classes, including `Boolean`.   However `float` is implemented only by the `Float` class.

The numeric classes implement the following traits:
- `Float:     num, float`
- `Fraction:  num, ratio`
- `Integer:   num, ratio, int`
- `Boolean:   num, ratio, int, bool`

Pili intelligently converts between these numeric types depending on the operation applied to them.  The `Boolean` values `true` and `false` used in a mathematical expression will be treated like 1 and 0 respectively.  If two numbers with trait `ratio` are divided, a `Fraction` will be the result — unless the numerator is divisible by the denominator (eg, 9/2 divided by 3/2 => 2), then an `Integer` will be the result.  `float` operations always result in `Float` values.

#### The Blank
There is a special value called `blank` which is the only value of type `Blank`.  It represents "there is no useful value here".  It is equivalent to what other languages call "null", "undefined", "None", etc.  There are a few operators that treat this value in a special way, namely: `??`, `.?`, and `?=`.

#### Strings
The next basic type is the `String (str)`.  It represent text.  Strings can be created with string literals.  Here are a few examples:
```
"this is a string literal"
'this is "literally" a string'
"strings with 'single' or \"double\" quotes can contain escaped characters such as \n(newline) and\t(tab) and \\(backslash)."
"Expressions can also be interpolated within strings using \{curly braces}:\n2 + 2 = {2 + 2}"
```

To render strings without all the escaping and formatting use backticks.
```
`any escape sequences (such as \n) or {expressions} will be rendered literally`
``You can even put in `backticks` in strings delimited by double backticks (or more)``
```

#### Collections and other types
Pili also provides lists, tuples, sets, and frozensets — all with functionality borrowed shamelessly from python.  There is no dictionary type, as maps provide dictionary-like functionality.

Indices start at 1 in Pili.  This applies to tuples and lists and other similar or derived types.  0 is not a valid index and will result in an error.  However, negative indices are usually allowed where element of list `ls` at index `-n` corresponds to the nth last element, ie 
`ls[-n] == ls[1 + len[ls] - n]`

The other types (functions, patterns, classes, traits, etc) are discussed in later sections.

***
See [[#Classes and Traits]] for more information on the type system in Pili.


## Maps and Functions
Maps are extremely flexible constructs in Pili.  Maps can act as namespaces since they have their own scope containing named variables.  Maps also act as functions, mapping dictionary-like containers holding key-value pairs using value options.  And of course, functions can run blocks of code by calling options — each of which has a certain signature that is dynamically chosen at runtime.

### Declaring Functions
The primary way to declare a function is with the `function` keyword, followed by the name of the function, followed by a block of code where all variables and options are defined.
```python
function greeting
	# variable
	default_who = "world"  
	# value option
	[1]: "one"
	# code options
	[]:
		return "hello " + default_who
	[str who]:
		return "hello " + who

print greeting.default_who
print greeting[1]
print greeting[]
print greeting["to you"]
```
```
world
one
hello world
hello to you
```

If you want to declare a function that has only one option, and no properties, there is a shorthand syntax:
```python
function foo
	[int x, int y]:
		return x + y
# is equivalent to
foo[int x, int y]:
	return x + y
# and also equivalent to
foo[int x, int y]: x + y
```
That last syntax, although the shortest, is only possible with one-line code blocks, and is not recommended except for the simplest of functions.

### Function Properties
In the above example, a variable called `default_who` was defined.  After being declared, this variable is accessible in any lower scope.  It can also be accessed from other scopes (wherever `greeting` is available) via the dot operator — this is the first of many uses of the dot operator.

The values of such properties can also be mutated wherever visible, and such changes will last the lifetime of the function.  New properties cannot be added outside of the function's block definition.
```python
greeting.default_who = "Me!"
print greeting[]
greeting.another_property = "some value"
```
```
hello Me!
ERROR: no proper "another_property" found in greeting
```

### Value Options
Value options consist of a key and a value (or possibly a code block, but not usually).  The key is usually one single immuclass value, but it could also be:
- no values
- multiple values
- muclass value(s)  (not recommended)

These key-value pairs are stored in a hash class and are therefore accessible in O(1) time regardless of the number of value options.  (The same is not true of code options.)

The syntax for defining key-value options is as follows:
```python
function foo
	[1]: "one"
	2: "two"
	[3, 4]: "three and four"
	5, 6: "five and six"
	[7]:
		return "seven"
	[]:
		return "no key" 
```

As you can see, the square brackets are optional — they have no effect on the program.  A key may consist of multiple comma-separated values, or no values at all.  The last two options are not key-value pairs, but they are still added to the hash class — they just run a function when called.

All of these value options can be retrieved and reassigned from other scopes using the same syntax as function calling.  New options can also be added this way.  The only difference is that value options must now use the ` =`  operator... basically just because programmers are used to this syntax.  
```python
foo[1] = "another one"
foo[5, 6] = "changed"
foo[5, 6, 7] = "new 5-7"
foo[7] = "SEVEN"
foo[]:
	return "still no key"
print foo[1], foo[2], foo[5,6], foo[5,6,7], foo[7], foo[]
```
```
"another one", "two", "changed", "new 5-7", "SEVEN", "still no key"
```

### Code Options
Code options consist of a pattern of parameters and a code block.  Patterns are ridiculously flexible, and therefore have their own section ([[#Patterns]]), but here are a few basic examples here:

Let's take our example from above:
```python
greeting[str who]: 
	return "hello " + who
```

If greeting has not been previously declared, `greeting` is now a function with one option.  Otherwise, this option is added or reassigned to the function `greeting`.  The pattern of the option has one parameter which is of type `str` (a trait) and binding `who`.  The resolution of this option is a code-block which will be executed each time `greeting` is called, and will return a string value.

Each parameter generally consists of a matcher pattern (usually a trait), followed by a name to bind, and optionally a quantifier or a default value.

```python
add[num x, num y]:
	return x + y

sum[num args+]:
	result = 0
	for n in args
		result += n
	return result

sequence[int start, int stop?, int step = 1]:
	if stop == blank
		stop = start
		start = 1
	res = []
	if step > 0
		condition[]: start < stop
	else
		condition[]: start > stop
	while condition[]
		res.push[start]
		start += step
	return res

# these functions are called like this:
add[1, 21]
sum[1, 2, 4, 5, 6, 7, 8, 9]
sequence[5]
sequence[1, 5]
sequence[5, 1, -1]
```

Quantifiers like `+`, `?` and pattern matching details can be found in [[#Patterns]]

#### Name-only Parameters and Named Arguments
When calling options, any of the parameters can be explicitly set by name at the call site.  But some parameters can *only* be set by name.  Option patterns actually have two sections separated with a semicolon; the first is for ordered parameters, and the second is for parameters that must be called by name.
```python
foo[str first_arg, int second_arg = 0; bool bar]
# bar must be explicitly set when calling foo
foo["first arg", bar=10 > 5]
foo["first arg", 10, false]  # will not match because bar is not a positional parameter
foo[bar=true, "first arg"]  # legal but not recommended
```

When calling a function, named arguments can go anywhere in the list and the remaining arguments will be matched in order.  However, it's usually clearest to list the named arguments in the same order as the parameters.  Using named arguments often is recommended because it leads to clearer code, slightly speeds up pattern-matching, and can help to disambiguate between similar options.
```python
foo["first", 10, bar=true]  # good
foo[first_arg="first", 10, bar=true]  # better
foo[10, bar=true, first_arg="first"]  # works, but not as clear
```

Another example:
```python
foo[param1=arg1, param2=arg2, param3=arg3]
```

This call is identical to the one above.  Except the order of the named arguments makes no difference.  Named arguments can also be interspersed with positional arguments, but the positional arguments will only be indexed by their position relative to one another, as if the named arguments were not present.  Therefore the following call is also equivalent:
```python
foo[param3=arg3, param1=arg1, arg2]
#   |            |            ^ first positional argument, takes first 
#   |            |              available position (2)
#   |            ^ fills position 1
#   ^ fills position 3
```

However, for clarity, it's generally best to maintain the position of each argument even when passing by name.

#### Nameless Parameters
Pattern parameters do not require names, however.  The function could be modified like so:
```python
greeting[str]: 
	return "hello"
```
*Note: I keep changing my mind about whether this is allowed or not... best not to try it.*

Of course, in this case, the actual argument that matched the `str` parameter is not bound to any name and therefore inaccessible to the body of the function, so functions parameters almost always should have names.

The usual exception to this rule is in value patterns (matched by only one specific sequence of values, possibly only one or zero values).  For example:
```python
greeting['John']:
	return "Hi John, I missed you!"
```

In this case, when `greeting` is called on the string `"John"` it will execute the associated code-block.  Whereas any other string will fall back to the `greeting[str who]` option we defined.  Usually, when there are no named parameters, it's simpler to just define a value-option.  The below option is equivalent to the above:
```python
greeting['John'] = "Hi John, I missed you!"
```

#### Call Flags
Pili also allows the usage of call flags.  
```python
greet[str name, bool capitalize = false]:
	if capitalize
		print title_case[name]
	else
		print name

greet['john doe']
# => prints "john doe"
greet['john doe', true]  # or: greet['john doe', capitalize=true]
# => prints 'John Doe'
```

The preceding code is fine, but it can be made easier to read and write using call flags.  
```python
greet[str name, !capitalize]:
	if capitalize
		print title_case[name]
	else
		print name

greet['john doe']
# => prints "john doe"
greet['john doe', !capitalize]
# => prints 'John Doe'
```
The `!` bang is just syntactic sugar:
- at the option definition, `!param_name` <=> `bool param_name = false`
- at the call site, `!param_name` <=> `param_name=true`


> [!NOTE] For future consideration
> Bools can be useful for tweaking function behaviour, but enums are arguably better in most cases.  Perhaps this `!syntax` could be expanded to include enums as well somehow.

#### Universal Function Call Syntax
The dot operator pulls double duty in Pili.  It's first role is to access fields of records (see [[#classes and Traits]]).  For example, `person.height` might return a number value, provided the `person` is a record that has a field called `height`.  If no field is found with that name, then Pili will instead try to treat the expression as a dot-call: that is, a somewhat inverted function call.  Pili tries calling the function `height` on the record `person`.  If no function `height` exists in the scope of the call site either, an error is raised.

Example:
```python
my_name = "Ryan"
len[my_name]
> 4
my_name.len
> 4
```

If the function is called with only one argument, the brackets are optional.  Ie, `my_name.len` is the same as `my_name.len[]`.  The one exception to this is in dot-options defined in function blocks — in this case leaving off the `[]` returns the function value, whereas including them calls it.

If the named function requires more than one argument, more arguments may be passed as if the function was an option of the first argument.
```python
match[my_name, `\w+`]
> True
my_name.match[`\w+`]
> True
```

If a function or record happens to have a field with the same name as another function in scope, then the field name will take precedence.  For example, assuming a `dog` record comes from a class called `Dog` and has a slot called `type` and value `"Chihahua"`:
```python
dog.type
> "Chihahua"
type[dog]
> Dog
```

In cases like this, though the program executes perfectly fine, it's considered better to use a different name for the slot (eg "breed" in this case would make sense), to avoid confusion.

However, in some instances, changing or extending built-in previously defined functions may be exactly what you want to do.  For example, you may want to implement the addition operator for a custom class, or modify the behaviour of `str[foo]`.  This is the purpose of "dot options."  See [[#Dot Options]]

## Variables Declaration and Scope
Variables, and options are created and assigned with three operators:
- ` =` for assigning a *value* (right-hand-side is evaluated before assignment)
- `:` for assigning a block of code (right-hand-side is saved for evaluation later)

```python
greeting = "Hello " + "world"
# immediately calculates the right hand side and assigns the VALUE to the option "greeting"
# if the name "greeting" is not already an option of the root function, it is added automatically

greeting[1] = "hello world"
# immediately calculates the right hand side and assigns the VALUE to the option [1] in the function "greeting".  
# if greeting doesn't exist, it will be initialized as a function
# if greeting exists as a record whose class does not define setting values, an error occurs.

greeting[str who]: 
	return "hello " + who
# creates a greeting function with one option whose pattern is [str who] and whose block is the single indented expression.
```

### Unconventional Handling of Scope
Variable declaration is not required in Pili; values are simply assigned to names without the need for keywords like "let", "var", "mut", etc.  Variables initialized in a higher scope are accessible in a deeper scope, but if that variable name is assigned, it will not be overwritten — rather, a "shadow" variable will be created in its stead.  That new variable is now accessible in the current (or deeper) scope, and the original variable remains untouched in the higher scope. 

Variables from higher scopes are still accessible in deeper scopes before the name is assigned.  
```python
x = "global x"
closure[]:
	print x  # a reference to "global x"
	x = "new value"  # creates a local shadow of x; does not overwrite the global x
	return x

print closure[]
print x
```
```
global x
new value
global x
```

The default scoping behaviour can be overridden in Pili by declaring variables using the `var` keyword.  In these cases, the variables are accessible *and muclass* in deeper scopes.  The default shadowing behaviour can be re-enabled with an explicit `local` keyword.
```python
var y = "global muclass y"
closure[]:
    print y  # prints global y
    y = "mutated y"  # mutates global y
    print y
    local y = "not global y"  # shadow of y; global y unaffected
    print y

closure[]
print y
```
```
globally muclass y
mutated y
not global y
mutated y
```


#### What about function options and record properties?
Any assignments where the left-hand side uses dot notation or brackets follows the conventions in other programming languages.  The scope of such identifiers is bound to the object itself, not the scope of the function.
```python
foo[1] = "one"
bob = Person["Bob"]

closure[]:
	# accessing foo from above scope
	foo[1] = "new one"  # mutation
	foo[3] = bob  # new dictionary entry
	foo[3].name = "Robert"  # mutates object stored in foo[3]

print bob.name
closure[]
if bob == foo[3]
	print bob.name
else
	print 'not the same'
```
```
Bob
Robert
```

#### Potential for Confusion or Unexpected Behaviour
If you are coming from a js paradigm where all variables are declared, you're basically good to go.

If you're coming from a python paradigm, and you like to use closures, it could be a little bit jarring to have to declare variables in the higher scope, rather than using the keywords 'nonlocal' or 'global' *in* the closure scope.

## Classes and Traits
Types in Pili are called classes.  Every value in Pili is a record of exactly one class.  classes are defined by a set of fields, options, and dot-options that together make up the interface for records in that class.

Here is a simple example of a class with some slot fields, record initialization, and the behaviour of such records.

```python
class Person
	slot name str
	slot height int
	slot friend Person?

ryan = Person[name='Ryan', height=183]

ryan.name
> "Ryan"
ryan.height += 1000
> 1183

fred = Person["Fredrick", 175, friend=ryan]

fred.friend == ryan
> true

ryan.friend
> blank
```

### Fields
Each field in a class has a name and a type, and may also have an associated function, or default value.  The slot is only one type of field.  There are two more: `formula`, and `setter`.

#### Slot
Each slot in a class corresponds to a piece of data in each record of the class.  Every slot of every record must be filled with a value.  Sometimes, however, that value may be the special value `blank`.  Any slot that is not filled with data upon initialization of the record (either by an initialization function or by a default value) will automatically be filled with the value `blank`.  If the type of the slot does not not allow `blank`, an error will be raised.  

In general, if the slot is ever filled with a value that doesn't agree with the slot type, an error is raised.

The question mark at the end of a slot type (like in `slot friend Person?`) marks a slot as optional.  Technically, it is a shorthand for the union of the type with the type `blank` and adds a default value of `blank`.  Ie,
```python
slot friend Person?
# is equivalent to:
slot friend Person | blank = blank
```

#### Getter
In other languages, a formula might be called a "getter", but Pili uses the term "formula" to evoke the feeling of a spreadsheet.  Formulas behave somewhat like slots, but there is no underlying data stored — instead, the data is generated by a function every time the formula field is accessed.

Here is an example of a class with a formula:
```python
class Product
	slot price_per_unit float
	slot quantity int
	formula total_price float :
		return self.price_per_unit * self.quantity

prod1 = Product[25.0, 2]

prod1.total_price
> 50.0

prod1.total_price = 74.0
> ERROR: no setter found for field 'total_price'
```

#### Setter
The setter is the complement of the formula.  Setters also have functions rather than stored data, but whereas a formula field is accessed by using `record.field` in an expression, the setter function is called anytime that data would be assigned to the field.  As such, setters fields usually only make sense when they have the same name as a formula field.  However, it is not strictly necessary and a setter may be defined for a slot, or simply on its own.

Example:
```python
class MyList
	slot _items list = []
	formula last any :
		if self._items
			return self._items[-1]
		return blank
	setter last[any item]:
		if self._items
			self._items[-1] = item
		else
			raise error["no items yet"]

ls = MyList[[1,2,3]]
ls.last
> 3

ls.last = 5
ls.last
> 5
```

#### Hidden/Private Fields
Pili does not officially have the concept of public/private methods or visible/hidden attributes.  However, these can be easily emulated.  Since a class can also act as a closure (see [[#class Options and Record Options|class Options]] below), local variables can be defined in a class scope just like in a function scope, and then those variables are accessible by all fields, options, and dot-options of the class.

In the example above we defined a slot `_items`, prepending an underscore to indicate privacy.  This is usually the best way to do it.  However, if need be, we could rewrite the `MyList` class above like so:
```python
class MyList
	items = {}   # {} is a function literal
	
	opt [int index]:
		return self.items[index]

	[seq ls]:    # the main constructor converts any seq to list
		rec = MyList.new
		items[rec] = list[ls]
		return rec

	[]: MyList[[]]  # this is a shortcut constructor for an empty list

	.push[any element]:
		self.items.push[element]

    .str:
        return str[self.items]


myls = MyList[]
myls.push[3]
print myls.str
> '[3]'
```

Now, for all code within the scope of `MyList`, `self.items` (ie, `items[self]`) should return the private list of items.  

### Record Initialization
Pili has one constructor function for all classes; it is called `new` and takes a class as its first argument, and any number of positional arguments after which will fill the slots of that record, in that order.  Because of this flexibility, the `new` function is prone to error (when, eg, slot order changes), so Pili automatically generates an option for each individual class.  This option is like a default constructor function — it has exactly one parameter for each slot: the name and type of each parameter matches the name and type of each slot, and the parameter is marked as optional of the slot provides a default value.

So, if we were to manually write this option in Pili for the Person class, it would look like this:
```python
class Person
	slot name str
	slot height int
	slot friend Person?
	
	[str name, int height, Person|blank friend = blank]:
		return Person.new[name, height, friend]

```

Hence, the following are all valid calls:
```python
Person['Fred', 145]
Person[name='Fred', height=145]
Person["Fredrick", 175, ryan]
# etc, etc
```

Of course, we can make our own constructor by defining options directly on the class that return a call either to the `new` function or to the automatically generated option.

### class Options and Record Options
classes are also functions, so options can be defined on classes in the same way that they are defined on function.

classes can also give records their own options.  This allows you to then treat records like functions.  This might be useful for custom container types, shortcuts for common methods, or function-like records.

```python
class Container
	
```


### Dot Options are like Methods
see also [[2024-Jul-4#How do methods work? Pili]] for brainstorming on this

#2024/Jul/13 I am realizing that there is still a need to improve this system.  Particularly, I believe we need to reserve the dot options ***only*** for adding options to existing functions.  Why?
- because any function scoped to the trait/class will (generally) only be accessible when called from an instance of that trait/class — and therefore already has access to the `self` record
	- the rare exception might be when you want to save a function from an instance to call later — but that's not a core feature and I can deal with that when it comes to it.
- and how can/should the `self` be inserted when defining a function with the `function` command?  It's not a dot-option, and it would be too much boilerplate to add `self` to each first parameter.

#### The following is very out of date
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

### The `self` keyword
The `self` keyword, by default, will resolve a reference to the "caller" of the option, that is, either the function holding the currently called option, or the record used to select an option of a trait/class.

```python
function foo
	[int n]:
		self  # this refers to foo

class Foo
	opt [int n]:
		self  # this refers to the *instance* of Foo (not class Foo)
```

However the `self` keyword can (and should) be overwritten in some circumstances.  Recall that dot-options are syntactic sugar for functions whose first parameter is a "self"  parameter.  In this case, `self` will resolve to the first argument passed.  In this case, there is no practical difference, but in other cases there might be.  

For example, you might want to change the pattern of the `self` parameter to expand or contract the matching.

```python
function bar
	...
	
class Foo
	slot prop
	slot func fn
	
	method[]:
		self.prop
		
	.bar[]:
		self.prop

	bar[Foo self, arg]:
		self.prop
	bar[arg; Foo self?]
		
	opt [int i]:
		self.prop

foo = Foo['value', x => x**2]
foo.method[]
foo.bar[]
foo.bar[arg]
foo[1]
foo.fn[5]

Foo.method[foo]
Foo.bar[foo]
Foo.bar[foo, arg]

```

##### Case 1: `method[]:`
the 


### Traits
Traits are like partial classes.  They may contain fields, options, and dot-options just like classes, but they cannot create or contain records (at least not directly) and the definition of such fields, options, and dot-options is allowed to be incomplete.  

classes are created by combining traits together...
```python
```

If a trait is found to be incomplete or incompatible with a given class, an error is given.  This usually happens when multiple traits define incompatible types for fields or options.

### class and Trait Conventions
By convention, the order of the elements of traits and class should be the following:
- local variables
- slots
- formulas
- setters
- class/trait options (eg, constructors)  %% to be honest, I actually have no idea where this should go in the ordering... %%
- record options
- dot options

But this ordering is by no means enforced, and it the user prefers another ordering for clarity or other reasons, Pili will not throw any errors.

#### Captialization
classes are in CamelCase, traits are snake_case.


## Patterns
Pattern matching is a core feature of Pili.  It is how options are selected and therefore how functions are called.  Functions, as stated before, are essentially maps where the key-value pairs are called options.  

#### Pattern Types
- **Value Pattern**.  
	- The simplest kind of pattern simply matches one particular value and nothing else.
- **class Pattern**
	- 
- **Type Pattern**.  
	- The most used type of pattern though are types.  Type patterns match any value which is descended from (has as its prototype) the given type.  
- **Union**.  
	- Union patterns combine multiple patterns together.  The sub-patterns can be any class of pattern.
- **List Pattern**.  
	- This is a fundamentally important class of pattern because this is usually how the the signature of a function option is defined.

####  List Patterns
A list pattern is like a regular expression for a list of argument values.  A list pattern consists of zero or more **parameters**.  A parameter is a pattern optionally associated with a *name* and/or a *quantifier*.

A Parameter is or has at least one of the following: 
- a name, ie, a string without quotes (eg `myVar`) 
	- *Note: this only applies when the string is not a reserved keyword and does not already refer to a type*
- a value (eg, `5` or `"five"` or `true`)
- a type or types (eg, `str` or `int|float`)
- a guard clause or sub-pattern (see [[#Advanced Patterns]])
- a quantifier (see [[#Advanced Patterns]])

> [!code] Examples of Parameters
> `my_var` only matches the name `my_var` or the string value `"my_var"`
> 
> `int` matches any integer value
> 
> `int index` matches integer values and binds the name "index" to the matching integer
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

### More thoughts on patterns
- sequence patterns are definitely different from dictionary/record-like patterns
	- ordered rather than named
- but both have bindings... ie, each element (parameter?) is:
	- index | field_name
	- matcher
	- \[quantifier?] -- only for sequenced parameters, of course
		- except, I guess, even field params could be labeled as optional
		- + and \* are fundamentally sequential in nature, disallowed for named parameters
	- binding 
- syntax for a sequential parameter: `str|num binding+ = "defalt_value"`
- syntax for a named parameter: `field_name: str|num binding = "default_value"`

#### Sub-patterns
sub-patterns can take three different shapes:
- **guard expression** (an expression that is evaluated for a truthy value, and each `.name` in treated as `foo.name` where `foo` is the value that matched the super-pattern.
- sequential pattern (for lists and tuples)
- named parameter pattern (for basically every other record)

Additionally, there are a number of special cases I may want to accommodate:
- string regex matching
	- shortcut for: ``str(.match[`\w`])``
- number ranges
	- shortcut for: `num($ in range[0,1])`
- list types
	- short for: `list($ is [num+])`

And then generic types as well.  But that's a whole other ballgame.. and probably should be done with \<angle brackets>.


#### Function Types
By the way, function types should be represents like this:
`[pattern] -> pattern`
eg
`[str, num] -> str`

I guess that's a matcher.

#### .................................. more pattern thoughts

Umm, so I guess what I currently have as class "Pattern" is actually just one of the cases of sub-patterns I listed above.  Specifically, its an Args sub-pattern... so maybe I actually need to go back to the structure I had before where list-pattern was actually a sub-class of pattern.  So my pattern class will be just for inheritance, and the pattern types will be:
- class
- trait
- value
- function
- ~~field~~ (no more field, that's just a sub-pattern)
- any

and then two or three types of sub-patterns:
- sequential pattern
- named fields pattern
- expression guard

All three can be combined.

And these pattern types are composed like so:
- Parameter
	- pattern
	- sub-pattern
	- binding
	- quantifier
	- default

Or, no Parameter, just:
- Pattern
	- sub-pattern
	- binding
	- quantifier ... or "required" na lang
	- default

and SubPattern is:
```
class SubPattern:
	pass

class ContainerPattern(SubPattern):
	parameters: tuple[Pattern]
	quantifiers: tuple[str]
	fields: dict[str, Pattern]

class ExpressionPattern(SubPattern):
	expression: Expression
```

Hold up, I actually have more classes of sub-patterns... 
- sequence
- record
- args (tests for both positional and named parameters)
- function (tests for options)

and, I guess sub-patterns aren't really fundamentally different from other patterns.

#### new proposed structure for pattern objects
```python
class Pattern:
	matchers: tuple[Matcher, ...]  # acts as an intersection of matchers
	sub_pattern: SubPattern | None
	guard: ExprPattern | None
	binding: str | None
	required: bool
	default: Record | None

class Union(Pattern):
	matchers: dict[str, Matcher]
	# a union doesn't need: required or default
	# does not sub-pattern, guard, and binding

# intersection is not needed, each pattern is potentially an intersection
```


```python
class Matcher:
	pass

class Pattern:
	matchers: tuple[Matcher, ...]  # acts as an intersection of matchers
	sub_pattern: SubPattern | None
	guard: ExprPattern | None
	binding: str | None

class Parameter:
	pattern: Pattern
	quantifier: str  # for named field params, only "" and "?" are allowed
	default: Record | None

class ProductMatcher(Matcher):
	# parameters: for matching iterables
	parameters: tuple[Parameter, ...]
	# fields: for matching slots and formulas by name 
	# OR named parameters of an Args record
	fields: dict[str, Parameter]

class Union(Pattern):
	# each matcher comes with a binding
	matchers: dict[str, Matcher]





foo[int|float x*, Bar(baz: str k), !flag]: ...


```

- ContainerPattern:
	- parameters:
		- int|float x*
		- Bar
	- fields:
		- flag


## Operators
### The Colon (`:`) Operator
The colon operator has several uses:
#### Function Declaration and Option Assignment
The colon used *outside* a function block always assigns blocks to options
```python
foo[int a, num b]: 
	return a + b
# defines a function foo with option assigned
# equivalent to
foo[int a, num b]: a + b
```

The colon used *inside* a function block or function literal assigns options to the function in scope.  One-line expressions with a colon will assign a **value option** whereas a colon followed by an indented block will assign a **block option**
```python
function foo
	# assign a code block to the params
	[int a, num b]:
		return a + b
	# assign the value three to the key [1, 2]
	[1, 2]: 3
	# square brackets optional
	1, 2: 3
```

Ambiguous cases must still be disambiguated:
```python
function foo
	a = 1
	b = 2
	[]: a + b
	# this assigns the value 3 to option []
	[]: 
		return a + b
	# this calculates a + b and returns the sum each time it's called

foo.b = 30
print foo[]
```
```
3
31
```


> [!error] Problem
> These rules disallow a parameterless function literal.  That is, the expression `{[]: a + b}` will return a function with **no** code options, and **one key-value option**.  If it stays like this, Pili programmers will have to come up with silly workarounds like `{[any _?]: a + b}`.

##### Potential Solution
Keep the function literal syntax as is and introduce another operator (eg ` => ` or ` -> ` or `::` 
) to create anonymous functions.  Or lambda syntax like Python.

✅ Just implemented that now 🙂 


### Categories of Operators
- Logical: 
	- and, or, not
	- can short-circuit
- Comparison:
	- ` ==, !=, <, >, <=, >=`
	- each of these can be chained with themselves, but not with others
	- `a == b == c` checks in order if each is equal to the last.  Can short-circuit.
	- `a != b != c` checks to make sure *each* operands is unequal to *all* other operands.  Also short-circuits.
	- `a > b > c` checks if a, b, c are in strictly ascending order.
	- mixing `>` and `>=` is currently unsupported.
- Mathematical:
	- `+, -, /, *, %, ^, **`
	- `+` is used for numbers, string concatentation, and sequence concatenation (like lists and tuples)
		- also has built-in chaining, so `ls1 + ls2 + ... + lsn` still runs efficiently for seq concatenation.
	- `/` produces rational numbers if both operands are rational.  Float otherwise.
	- `^` and `**` are identical
- Assignment: 
	- `:, =, +=, *=, /=, etc`
	- the ` =` operator reads any pattern on the left-hand side, interpreting bare identifiers as 'any', just like parameters
	- `&&=` and `||=` are equivalent to `foo = bar and baz` and `foo = bar or baz` respectively.
- Pattern operators
	- `&, |, @, ?, is, is not, ~`
	- all pattern operators will automatically convert their operands to patterns (if they aren't already)
	- `&` generates an intersection pattern — value must match both operands
	- `~` acts the same as `&` but negates the second operand.  
		- eg `seq~str` will match any **non-string sequence** value, 
		- eg `num~0` matches any **nonzero number**
		- Can also be used as a prefix operator, in which case it's equivalent to the first operand being `any`.  Eg, `~blank` matches any possible values **except `blank`**.
	- `|` generates a union pattern — value can match either of the operands
	- these operators do not short-circuit
	- `@` is a prefix operator that generates a **value pattern**, ie, a pattern that only matches one single value.  This is useful in contexts where the syntax would otherwise yield a trait matcher or a wildcard matcher.  Eg, `value is str` checks if `value` has a `str` trait, whereas `value is @str` checks to see if value is in fact the trait `str` itself.  
	- `is` takes a value on the left, and a pattern on the right — returns `true` if match, `false` otherwise.  Reverse is the case for the `is not` operator.
	- `?` as a postfix make an pattern that is union of the original pattern with the `blank` value, and also adds a default value of `blank`
		- so therefore `str text = 4` raises an error, but `str? text = 4` assigns `blank` to `text`.
- Function Ops:
	- `.` select property / call method or function
	- `[]` select and call option
	- ` => ` creates a function with parameters on left, and an expression on the right
- "safe" operators
	- `.?` is the "safe-get" operator — it acts the same as a dot except that it returns `blank` instead of raising an error in the case that a property/function is not found.
	- `?` is the "safe-call" operator — it goes between a function and a set of arguments.  It returns `blank` if the option is not found in the function, otherwise calls normally.
	- `??` nullish coalescing: evaluates and returns the first argument if it is not `blank`.  If it is `blank` or it is a name that doesn't exist, evaluate and return the second operand.
		- similar to the `or` operator in that it short circuits, but only for `blank` values, and will not raise an error for undefined names
	- `??=` has the same behaviour, but also works for assigning properties: `foo.bar ??= "one"`
- Other operators:
	- `in`: checks if a value is a key of a function, or a member of a list
	- `has`: checks if a record or function has a property (takes a string value as it's right operand) or has a matching option (takes a list of arguments as it's right operand).
		- when used as a prefix operator, checks to see if a name (string) exists in the current scope
	- `to` creates an inclusive range object.  Can also be combined with `by` to set the "step" property of the range.
		- eg `for i in 1 to ducks.len` iterates through all indices of `ducks`
		- eg `(1, 2, 3, 4, 5, 6)[-1 to 2 by -2]` yields `(6, 4, 2)`
	- `..` the 'swizzle operator' or 'map-dot' operator
		- left-hand-side is an iterable, right-hand-side is a name, method call, or function value.
		- for `my_sequence`, returns a list where the element at `i` is:
			- `my_sequence[i].prop` for the expression `my_sequence..prop`
			- `my_sequence[i].foo[args]` for the expression `my_sequence..foo[args]`
			- `my_sequence[i]^2` where `foo` is the function in `my_sequence..(n => n^2)`
	- `..?` safe-swizzle works the same way, but will default to `blank` for list members missing the given property.  This may be combined with the safe-call operator as well
		- eg,  `people..?paycheck?[100]` will add $100 to the bank account of each person with a bank account, yielding `blank` for those without.
- pseudo operators
	- some symbols that syntactically resemble operators but are evaluated as different kinds of expressions
	- `?`, `+`, and `*` are regex-like pattern quantifiers when used right after a parameter binding
	- `!` as a prefix in a set of parameters creates a flag parameter.  Sets a flag in arguments.
	- `.` as a prefix creates a "dot method"
	- `*` prefix in a list-like context spreads any iterators
		- eg, `nums = [1, 2, 3];  (0, *nums)`  yields `(0, 1, 2, 3)`

### Assignment
There are several uses for the ` = ` operator.  The simplest and most common is assigning names to values.
- eg, `foo = 5`
- where `foo` is any name token

The second use for assigning values to values of fields.
- eg `foo.bar = 5`
- where `foo` is any expression that evaluates to a record, and `bar` is the name of a field in said record.

The third use is assigning value-options
- eg `foo[bar] = 5`
- where `foo` is any expression that evaluates to a function, and `bar` is any expression

The fourth use is destructuring/pattern-matching assignment.  The left side is any valid pattern-matching expression, and the right side is any expression
- eg `Person(first_name: str name) = fred`
- if `fred` is a record like `Person[first_name='Fred']`, then this expression will assign "Fred" to `name`.

#### Should I support assignments like:
```
foo.len = 5
```
#### ?
#### Nullish Assignment
- nullish assignment uses the same syntax as regular assignment, except for the last one, `<pattern> = <expression>`
- It works differently for all three variations of syntax:
- `<name> ??= <value>`
	- assigns value to name if name is *undefined* **or** is defined but evaluates to `blank` (the singleton).
- `<expression>.<name> ??= <value>`
	- assigns value to the field named "name" of the record which is the evaluation of expression **iff** the `<expression>.<name>` evaluates to `blank`.
- `<expression>[<args>] ??= <value>`
	- assigns value to the option located in expression at args **iff** the option does not already exist.
- in summary:
	- bare name => checks for existence *and* blankness
	- assigning to option => checks *only* if exists
	- assigning to field => checks *only* if blank

## Advanced Patterns
In addition to basic types, unions, and prototypes, a pattern may also have further specification.  This comes in the form of a **guard** or **sub-pattern**.

### Quantifiers and List Pattern Matching
The three parameter quantifiers are:
- `?` — optional
- `+` — multiple (at least one)
- `*` — optional multiple (zero or more)

#### Examples
```python
[int*, str]
~ ['hi']
~ [0, 2, -3, 'hi']
!~ [2]
!~ []

[int*, float?, str+]
~ ["hi"]
~ [1, 2, "hi"]
~ [-1, 0.23, "hi", "there"]
~! [0.01, 1.1, "s"]

[int, float*, str*, float*]
~ [1, 2.3, "str", 0.01]
~ [1, 2.3, 4.5]  # the 4.5 is counted in the first float list

[int, float*, str*, float+]
~ [1, 2.3, "str", 0.01]
~ [1, 2.3, 4.5]  # the 4.5 is counted in the SECOND float list

[num*, ratio, int, str]
~ [1, 1, "end"]  # but a simple greedy algorithm won't catch this, because the two 1s will be consumed and then "end" doesn't match ratio
# solution: track minimum number of arguments still required

[num*, ratio?, int+, str]
# ratio will never get matched
~ [1, 1, "end"] # matches with same logic as above
# if a parameter has already been "satisfied" (ie, at least 0,0,1 matches for *,?,+) and 

[int*, num*, str?, int, str]
~ [1, 1, "str"]
~ [0.4, 2/3, ]

[int*, str, int, ratio*, str]
~ [1, "h", 5, 1/3, "j"]

[num*, str?, int+, str*]
~ [.1, .1, 1, "s"]
# problem: greedy algorithm will consume all three nums with the first param because it will measure only one min length.  And it won't match the last int
# backtrack?  
```

chatgpt prompt:: I want to write a pattern-matching algorithm for the programming language I'm writing. The pattern-matching is similar to regular expressions, but instead of matching strings, the pattern will match against lists of values. A pattern consists of zero or more parameters. A parameter consists of a type (which either matches a given value or not) and also a quantifier. There are three quantifiers: +, *, and ?. They are analagous to those characters in regex matching. That is to say, if a parameter has a + quantifier, then it can match one or more values in a row. If * then it can match zero or more values in a row, and if ? is the quantifier then it can match zero or one value. If there is no quantifier, then it must match one value.


### Pattern Guards
`int > 0` is an expression that yields a pattern of type `int`, so it will match integer values, but it will only match integer values greater than 0

### Sub-Patterns
Prototype patterns may also contain sub-patterns such that a function value only matches the pattern when it's prototype matches AND the specified property matches the sub-pattern.

Sub-patterns are usually defined as comma-separated expressions.  Value `foo` matches pattern `patt` with expression `expr` only when all the expressions evaluate to truthy values in the context of `foo`.

```
pos_point = @Point[x > 0, y > 0]

print Point[0, 5] ~ pos_point
print Point[-1, -1] ~ pos_point
print Point[1, 4] ~ pos_point
```
this program will print false, false, true

### Function Class Patterns
maybe some way to pattern-match the keys of a function... meta-patterns
```
pos_point = Pattern{
	x : numeric
	y : numeric
}
```

Either patterns that match patterns... OR check for the existence of an option matched by a given value or values.


## Ideas and Issues
### IDEA: Enums, Atoms, Symbols
- **observation:** maps have both hashed options, and named variables.  If map `foo` has a named variable `bar` as well as a hashed option at `"bar"`, then `foo.bar` and `foo["bar"]` are very similar conceptually.
- **idea:** make another string-like type (enum, atom, symbol, or something similar) and put these values in the hashmap
- suppose I use the syntax `#bar` to make a symbol
- **implications:** 
	- now these two expressions are equivalent: `foo.bar` <=> `foo[#bar]`
	- likewise, these two are also equivalent: `bar = 5` <=> `#bar: 5` (within a map block)
	- using a name in an expression would then check for symbols in the scopes ascending like normal
#### how does this help with enums?
- now we can make an enum function that can be used like this:
- `TokenType = enum[#name, #number, #string]`
- equivalent to:
```
map TokenType
	name = #TokenType/name
	number = #TokenType/number
	string = #TokenType/string
```

or maybe
```
class TokenType (enum)
	var name
	var number
	var string
TokenType.name = TokenType[]
TokenType.number = TokenType[]
TokenType.string = TokenType[]
```

or maybe
```
map TokenType
	name = #name
	number = #number
	string = #string
	.@:
		#name | #number | #string   
	# meaning TokenType will be patternized as this union pattern
```

This one is nice, because it allows me to define functions like this:
```python
map parse
	[Token(text: text, type: TokenType type), Scope context]:
		if type == #name
			...
		...

print_options = enum[#pretty, #info, #normal]
```

Or, even better, just make the UnionMatcher an iterable object.
```
TokenType = (#name | #number | #string)
print_options = ( 
	  #pretty
	| #info
	| #normal )

map print
	[msg, print_options popts=#normal]:
		<some code>
		match popts
			#normal:
				<do normal print>
			#pretty:
				<print nicely>
			#info:
				<print informatively>
		<some more code>
	
	[msg, #info]:
		<print informatively>

	[msg, bool pretty?]:
		if pretty
			<print nicely>
		else
			<print normally>

	[msg, pretty=false]
	
```

#### Does this complement or clash with the idea of leaning further into UFCS?
- it clashes.  if `foo.bar` is equivalent to `foo[#bar]`, then how can it also be equivalent to `bar[foo]`?
- So, it's a good thing that [[#Idea Converge on Dot Options]] was **rejected**
- Actually, it gives the developer a fairly ergonomic option to specify field-access without risking an accidental function-call.


### Destructuring Assignment via Patterns
IDEA: make the equals sign simply run the pattern-matching algorithm as if calling a function  
  - that will also bind names — and allow very complex destructuring assignment!  
What about assigning values to names of properties and keys?  
- eg, `foo.bar = value` and `foo[key] = value`  
- `foo[bar.prop]: ...  `
- `foo[5]`
- special case it?  It's not like you're gonna see that in a parameter pattern anyway
	- ... except you might want to use it as a value pattern like I do all the time with enums.
	- `[Node(type: NodeType.foo), ...]: ...`
	- similarly, there are times that you want to use a bare name as a value-matcher, eg `function foo... ; len[foo]: ...` 
	- what about having both of these default as `bind(any, name)` and only interpret them as patterns with the `@` operator?  Or the other way around?
		- `[Node(type: @NodeType.foo), ...]: ...`
		- `len[@foo]: ...`
		- this might be the only/best way to make patterns match literal traits and classes
		- prolly good to have the builtin singletons as exceptions, not needing the `@`: `blank, true, false, inf`
		- wait lang... I'm sensing another potential path here: what about pattern expressions with other (non-pattern-specific) operators like `-foo` or `foo+bar`?
			- what if we go back to only treating bare names as `bind(any, name)` and just evaluate other expressions... no that is what I'm doing, the debate is just 
- 0r could actually integrate that behaviour into pattern matching.  
	- standalone dotted names will bind to those locations (not local scope)  
	- function calls same thing... `foo[key]` will bind to that location  

btw, if I start making more widespread use of patterns like this, I might have to add in a method  
to Node to evaluate specifically to patterns.  Node.patternize or Node.eval_as_pattern

**Issue:** in the case of assigning to function keys like `foo[key]`, the binding is not a name.  Patterns right now can only bind to names.  And speaking of which, those names don't have scopes associated with them... maybe I need another class `BindTarget` or just `Target`.

Node.eval_patt
```python
class Node:
	def eval_patt(self):
		return patternize(self.evaluate())

class Token(Node):
	def eval_patt(self):
		if self.type == TokenType.Name:
			return Bind(any, self.source_text)
		return super().eval_patt()

class OpExpr(Node):
	def eval_patt(self):
		match self.op:
			case '.':
				pass
			case 'call':
				# assign to a function key like foo['key']
				fn, args = lhs.evaluate(), rhs.evaluate()
				return IntersectionMatcher(
						ValueMatcher(fn),
						OptionMatcher(args))  # oh, the binding is not a name... how do I handle this?
			case _:
				return super().eval_patt()

class ListNode(ArrayNode):
	def eval_patt(self):
		match self.list_type:
			case Tuple:
				# multiple assignment
			case Function:
				return FunctionMatcher(...)
				# destructured assignment

class VarExpr(Declaration):
	def eval_patt(self):
		
```

```python
```

### Implicit Dot Calls, Implicit Dot assignment
In Pili, `foo.bar` is an overloaded expression:
1. if bar is a field of foo's class, it will evaluate the field
2. if bar is a function, (either in foo's class or traits, or containing scope) it will call that function using foo as the first argument
	- ie, `foo.bar` in this case is equivalent to `bar[foo]`

Currently, in setting the value using that expression as a left-hand-side of the eq operator, eg `foo.bar = 5`, it will only set the field.  It will not attempt to set it as if it was `bar[foo] = 5`.

Of course, for consistency, we should allow `foo.bar = 5` to assign 5 to the location `foo` in `bar`.  But there are two problems:
1. It feels weird: flexible calling/getting seems stranger than flexible setting/defining
2. nullish assignment get even weirder:
	- eg `foo.bar ??= 5` 
		- if bar is a field of foo, set it to 5 iff it's not blank
		- if bar is a function.... then do we try calling it to see if it evaluates to blank first?
		- I guess not.  That would be too unexpected.  So we must draw the blank-checking line somewhere.  Where?
			- we could check for blank only if it's in op_dict
				- that would mean the only inconsistent case is when bar is found in op_list
				- that's not too bad, but there are other options
				- for example: we could *never* check for blank only check for existence
					- so then `foo ??= 5` => assigns 5 only if foo not exists
					- `foo.bar ??= 5` where bar is a field of foo loses semantic value, because this would imply setting bar only if bar doesn't exist, but assignment can take place only if the property bar does exist.
					- `bar[foo] ??= 5` => assigns 5 only if foo is not an option of foo
				- compromising solution: check for blank only on names and properties, but in functions only check for existence.
					- `{"key": num foo} ??= {"key": 5}`
					- assigns 5 to foo
					- `{"key": num foo} ??= {"key": "string value"}`
					- ... maybe suppresses the error?
					- nah, if you want that functionality, best to use a match statement instead

### Loops and Generators
In python there are `for ... in` loops, `while` loops, generator functions, generator expressions and iterators.  While loops are the simplest — they just loop until they break.  `for ... in` loops require an iterable object — `iter` (a generator function) is called on the iterable, which produces a generator iterator.  The iterator spits out values for the body of the loop to act on.  Custom generator functions and generator expressions can also be used.  Usually generators are one line long, but they can be arbitrarily complex.

Can these tightly related functionalities be repackaged in a different way that is at least as ergonomic and expressive?

#### So what does it take to make a loop?
1. initialization code
2. code to produce a value
3. code to do something with that value
4. code to send a value back into the loop (Python's `send` function)
5. code to break/stop the loop/generator

Of course its prudent to combine a few of these at a time into one code-block (or even one expression).  Here are two examples of how python does it:

```python
ls = [1, 2, 3, 4]    
# 1., 2., 5. initialization and yielding code created with an implicit iter() call.  Also defines loop end.
for i in ls:         
	if i == 0:       # 3. block of code
		break        # 5. optionally break code within block
	print i
# in this case, there is no (4) send value back into loop
```

```python
def accumulate():
    tally = 0               # 1. init
    while 1:                # 5. break condition
        next = yield        # 2, 4: send and receive value
        if next is None:    # 5. another break condition
            return tally
        tally += next       # 2. more code to produce value


for i in range(5):          # 
	print(acc.send(i))      # 3. do something with value

# another way to write this? without def accumulate?
__tally = 0
for i in range(5):
	yield = i
	# -----enter------
	next = yield
	if next is None:
		out = __tally
		break
	__tally += next
	# while 1
	yield = None
	# ----exit-------
	print(yield)


```

So in general, in python, the steps are located as follows:
1. **init**: body of generator function and/or callsite of generator function
2. **yield value**: body of generator function
3. **do something** with value: body of loop
4. **send:** send-expression in loop (or wherever) and yield expression in body of generator function
5. **break:** return statement in generator, or break statement in loop

```python
loop i = 0  
    print i  
    yield i  
    i++  
int i => break if i

for i=0; i<len(ls); i++ {
	do stuff
}

for i=0; i<len(ls); yield ls[i++]
	do stuff

for i in (i=0; while i<len[ls] {yield ls[i]; i+=1})
loop
	i ??= 0
	if i >= len[ls]
		break
	send ls[i]
	i++
	receive 
		[str signal]:
			print signal
		[0]:
			break
		[1]:
			continue
		[int signal]:
			raise Error['Some other int signal']
	continue
then
	receive str element
	if element in kwargs:  
	    if param.pattern.match_score[kwargs[name]]
	        bindings[name] = kwargs[name]
	        kwargs.del[name]  
	    else:  
	        send 'no dice'
	elif element == 'default'
	    bindings[name] = param.default  
	else
	    send 0
	send 1
```

```python
def accumulate():
    tally = 0
    while 1:
        next = yield
        if next is None:
            return tally
        tally += next

def gather_tallies(tallies):
    while 1:
        tally = yield from accumulate()
        tallies.append(tally)

tallies = []
acc = gather_tallies(tallies)
next(acc)  # Ensure the accumulator is ready to accept values
for i in range(4):
    acc.send(i)

acc.send(None)  # Finish the first tally

for i in range(5):
    acc.send(i)

acc.send(None)  # Finish the second tally
tallies
```

### Syntax for Assigning Options and Variables
- A given assignment operation has many dimensions in Pili:
	- context: function scope, trait/class scope, other (eg `foo.name` or `foo[something]`)
	- type of LHS: name, key, pattern
	- type of RHS: value or code block
- simply multiply these dimensions together and you get 3×3×2 = 18 different possibilities.
	- can this be trimmed down?  What combinations don't make sense?
		- `_, pattern, value`: might make sense in some edge cases, but could be ruled out for simplicity if convenient
		- `_, name|key, code block`: also potentially useful (just for performance, likely) but could be rules out as well if need be
		- combine the above two and suddenly the type of LHS determines the type of RHS
- syntactically, I have quite a few choices to make to distribute these 
	- of course the context is determined for me (presence or absence of expression before name|key|pattern, or the function|class|trait keyword above current block)
	- RHS: block or expression — this is also already more or less set in stone.
		- however, I could allow reading expression as block in certain cases (eg, when pattern is detected)
	- this just leaves LHS.
		- distinguishing a name is easy.  `Token(type=TokenType.Name)`
		- but how to disambiguate between key and pattern?
			- option 1 (currently in partial use): treat all as patterns and detect which ones are hashable
			- option 2: key iff RHS is expression, pattern iff RHS is block
			- option 3: key iff operator is ` =`, pattern iff operator is `:`
			- option 4: key iff no brackets, pattern if brackets
				- only works within function scope, how do you do it for other scopes?
					- another operator, like `@`... eg `foo@1`... (not great for tuples)
					- other langs: `[brackets]` for key, and `(parens)` for pattern
			- option 5: option 4 in function scope, option 3 outside function scope

Option 5
```python
function foo
	name = value
	name = 
		block
	key: value
	key:
		block
	[key] = value
	[key] = 
		block
	[pattern]: value
	[pattern]:
		block
	# no more one-line blocks

foo.name = value
foo.name = 
	block
foo[key] = value
foo[key] =
	block
foo[pattern]: value
foo[pattern]: 
	block
foo.bar: value/block
```

Option 3: key iff operator is ` =`, pattern iff operator is `:`
```python
function foo
	name = value
	name = 
		block
	[key] = value
	[pattern]: value
	[pattern]:
		block
	# (no key-block, or one-line blocks)

foo.name = value
foo.name = 
	block
foo[key] = value
foo[key] =
	block
foo[pattern]: value
foo[pattern]: 
	block
```

Option 2: key iff RHS is expression, pattern iff RHS is block
```python
function foo
	name = value
	name = 
		block
	key: value
	[key]: value
	[key] = value
	[pattern]:
		block
	# (no pattern-value, or key-block, or one-line blocks)

foo.name = value
foo.name = 
	block
foo[key] = value
foo[key] =
	block
foo[pattern]: value
foo[pattern]: 
	block
```

Option 6:
```python
  function foo
1	  name = value
2	  name1, name2 = multiple, values
3	  [str name1, str name2] = "multiple", "values"
4	  name = 
		  block  # acts like a formula: foo.name executes block
5	  "key": 'value'
6*	  ["key"]: 'value'  
7	  ["key"]:
		  block
8	  [str key]:
		  block

1 foo.name = value
2 foo.name1, foo.name2 = multiple, values
3 [str foo.name1, str foo.name2] = "multiple", "values"
4 foo.name = 
	  block # acts like a formula
5 foo["key"] = 'value' | foo["key"]: 'value'
6*foo["key"] = 'value' | foo["key"]: 'value'
7 foo["key"]:
	  block
8 foo[str key]:
	  block
	
```

`*` I might have to make 6 into a special case

#### Additional Ambiguity: `foo[] : value`
- is `foo[]` a name and parameters to be defined?
- or is `foo[]` a function call that returns a key to be used as an option?
- I think it has to be the latter, otherwise I don't know if there's anything reasonable to do other than admitting the syntax is inconsistent.  Because `(foo[]): value` is definitely a key assignment
	- oh wait, that gives me an idea...
	- docs could state that `[key]: value` is the canonical way to define a key, but allow the `[`brackets`]` to be dropped ... except in this exact case
- so, alternatively, I need to change the syntax for function definition.
	- `foo = [] => ...`  <- too verbose, messy symbols
	- `fn foo[]: ...`  <- pretty clean and clear
	- `function foo[]:`  <- more verbose, but has the benefit of matching the function keyword
	- plus side of the last two: helps to visually distinguish from expressions like `my_list[1] = 'one'`
	- problem: what then do we do when we just simply want to assign *extra* options to an existing function?
		- potentially I could use the extra case from `6*` above for adding a function
		- ... or go back to options requiring square brackets by default
- there's also **another issue** I just realized: if the key is automatically detected as hashable args vs paramset, then what about name Tokens outside of brackets?  Is `foo: ...` intepreted as `[any foo]: ...` or `<value of foo>: ...` or even `foo[]: ...`?                                             

Currently:
```
one = 1
function foo
	1: "one"           # option assignment
	[1]: "one"         # with or without brackets
	any n: "one"       # automatically detects if key is
	[any n]: "one"     # hashable value(s) or paramset
	[any n]:           # value vs block is determined by
		return "one"   # one-liner vs indented block
```

How does Python do it?
- `1: "one"` for assigning keys within dict, `foo[1] = "one"` for assigning keys outside dict
- `def foo(args): ...` for defining function option
- no function option

#### Brainstorming
A few strategies I have available to me:
- automatically detect hashable keys
- What if I made `@ <expression> = ...` for key definition?
	- notice the inconsistency in python where `{1: 'one'}` is kinda equivalent to `foo[1] = 'one'` even though they use different operators.
- also making me think, what if I resurrect an old idea: no variables in function scope, only string options?  I mean, make those two concepts equivalent.
	- so then `name = value` in `foo` is equivalent to `foo["name"] = value`
	- what about non-string options?  You can literally assign to any value:
		- `"var_" + "name" = 1`
		- `1 = 'one'`
		- `foo[bar] = 'one'`  # assigns value 'one' to the key whose value is the output of `foo[bar]`
		- `foo.bar = 'one'`  # likewise, doesn't assign to `bar` in `foo`... makes a key out of the value `foo.bar`
	- yeah, this is super dumb, now that I think about.  `foo.bar = ` and `foo[bar] = ` are really useful and ergonomic ways to assign names and values in `foo`, I don't want to get rid of them so easily.  I would have to replace it with something like `foo::bar = ...` and `foo::[bar] = ...`
- ok, what about using the `:=` symbol?
	- so then `1 := 'one'` and `foo := 'one'` both assign keys to the values `1` and `<value of foo>`
- wait, I forgot about another tool I have in my toolbox that could really help with this disambiguation: **one-liner vs block distinction.**
	- I could make it so that this syntactic distinction applies to ***both*** evaluated-now value vs evaluated-later block ***and*** hash-key vs paramset

```
function foo
	1: "one"           # key assignment
	[1]: "one"         # with or without brackets
	any n: "one"       # *still* key assignment, but assigns to 
						 the value of the pattern `any n`
	[any n]: "one"     # same as above
	[any n]:           # now this is finally interpreted as pattern
		return "one"   # of parameters option

	bar[baz]: "one"    # so this assigns the value "one" to the key
						 whose value is the result of bar[baz]
	bar[baz]:          # and this assigns to option(patt=baz)
		return "one"   # of function bar
	any n:             # this might raise a syntax error
		return "one"
	bar:               # because this is ambiguous — is bar the param
		return "one"   # or is bar the function?
	.bar:
		return "one"   # this is not ambiguous
	
```

If you do want to assign an evaluate-later block to a simple key value, you have to make it a function.  And if you want to make a pattern return a value, just `return` it.

Alternatively, make the distinction using `[`brackets`]` ... except that still makes `bar[baz]: ...` ambiguous: is it a key with value `bar[baz]` or is it an option of `bar` being defined?


### Prototypes

> [!warning] Stub
> This section is not developed yet, just some ideas floating around.


So, eg, a string *value* (ie, what is returned from a string literal expression) is now a Function with prototype=`str` and python property `value` equal to the python string value.  And I could put a few more options on the prototype as well, if need be.

I could also drop the word "prototype" completely and just say "type".

***
So now that i've started implementation of this, I'm realizing a syntax parsing issue.  Before, `int` specifically referred to a *type*, and prototypes had to be designated with `@`, and other names became values or param-names, depending on context.

So now that there is no difference between "Value" and "Function" and also types all became prototypes... 

What do names refer to when reading parameters?
- name
- prototype pattern
- value pattern

```python
foo[name]: ... # name
foo[0]: ... # value or prototype 
foo["value"]: ... # value or prototype
foo[int]: ... # prototype only 
foo[Date d]: ...
foo[Date]: ...
foo[today]: ...
```

One possible: **name last**
- execute the phrase, but if there is a name left over, interpret it as a name 
- if phrase is name only: 
	- name iff unused
	- union(Value|Prototype) by default 

Explicit via `as` operator
- name only if following `as`
- `int as n`, `int|Date as d`, `any as arg` 

Other complicated logic?
- no

Okay, option 2 is too verbose.  Here's a refinement of option 1, for more consistency and predictability:
- AST will add `as` keyword iff last token is a name **and** either no binary operator precedes name.
- resulting value is patternized into value|prototype 
- so a name on it's own is invalid, it must be `any name`.  

### Idea: Converge on Dot Options

> [!summary] Status: Rejected
> `foo.bar` should **NOT** be equivalent to `bar[foo]` in general.


[[#Virtual Options or Dot Options]] already form a core part of the functionality of Pili.  What if we turn towards more fully relying on dot options to replace names and even function properties/methods?

> [!Currently] 
> - `name` is equivalent to `["name"]` in a given scope
> - `scope.name` is equivalent to `scope["name"]`
> - `foo.bar` is equivalent to `bar[foo]`
> - `foo.bar[args]` is equivalent to `bar[foo, args]`

#### Proposal 1
- only one scope for all names
- `name = "hello"` defines a name in the namespace.  `name` is not an option of any function.
- `name` is not equivalent to `["name"]` as the latter calls an option with the string argument `"name"`
- `foo.bar` is still equivalent to `bar[foo]`, but both `foo` and `bar` are global names
	- the dot-call only works if `bar` resolves to a function that has an option whose pattern matches `[foo]` — otherwise an OptionError occurs.
- `foo.bar[args]` is also still equivalent to `bar[foo, args]`
- `foo.bar.baz` is likewise equivalent to `baz[bar[foo]]`
- `len` selects the global name `len` which can then be used in any expression like `len[foo]`
- `.len` (without any leading name) is equivalent to calling `len` on the current context.

##### Problems
- [x] Will there be many options for any given name in the namespace?
	- probably not actually... likely just one or a few prototypes for each name.  Just like there is not usually a substantial amount of overlap in property names of different objects.  
	- And for each prototype, probably also just one or a few options.
	- I guess one name is likely to have many options: `i`.
- [x] Will I ever have the need to refer to properties by their constructed string name, rather than literal names?
	- [x] well, in regular programming, no, so why should it be any different here?
- [ ] in order to call name on the given context, a `.` prefix is required.  Will this lead to excessive dotting?
	- [ ] cause every time you want to use a non-function variable in a piece of code, you have to prefix it with a dot!
	- [ ] So what if we reverse the syntax, or just get rid of the need to prefix with a dot

By default, `len` on it's own will be called on the current context as if it was `scope.len`.  So now we have two possibilities: 
- since `len` on it's own is essentially `scope.len` then either:
1. `len[foo]` => `scope.len[foo]` => `len[scope, foo]` (ie, the context is *always* the first argument)
2. `len[foo]` loses the scope when given an argument

Number 2 is more consistent, I think.  Otherwise `foo.len` is actually `scope.foo[len]` which is weird. 

so then the next problem is, how do we get the actual `len` function as an argument, if using the word `len` actually calls the function?  Well, perhaps the `len[root]` option could return the `len` function itself.

That works for now... we'll see if it holds up

##### Defining Dot Options
- `name = "Hello"` => defines a function `name` which, when called with an argument matching the prototype pattern of the current context, returns the value `"Hello"` 
	- it is functionally equivalent to `name: "Hello"`
- `foo[str bar]: "Hello "+bar`
	- defines a function `foo` with an option with pattern `[scope, str bar]`

Alright, I think I'm realizing this is not actually any different to the current state of affairs.  It's just moving all the names to one place, and moving the context to the first argument.

#### Updated Proposal
- bare names have regular scoping
- but dotted names are looked up in the same scope, and interpreted as reverse function calls
- ie, `foo.bar` is always equivalent to `bar[foo]` in a given scope, and does NOT look up a variable `bar` in `foo`.
- `foo.bar[args]` is also still equivalent to `bar[foo, args]`
- `foo.bar.baz` is likewise equivalent to `baz[bar[foo]]`, as is `foo.bar[].baz`
- `.len` (without any leading name) is equivalent to `len[scope]` where `scope` is the currently running function instance

##### Advantages
- it's different, unique, fun to play with
- simplifies dot-calls, less ambiguity
- appropriately differentiates between properties and local variables

##### Potential Issues
- more lookups?  **No**.  Actually it should be exactly the same, assuming we're able to use dictionaries for all properties
- of course, it's **counter-intuitive** for `foo.bar` to be a member of `bar` rather than a member of `foo`.
- Where can I get a collection of all "keys" of `foo`? 
	- for user-defined keys, I can just add them to a set as they are defined
	- for built-in keys, the only way to do it is to iterate over every builtin and check `key has foo`
- What about when you have a variable name in the global scope (or some upper scope), and an unrelated property of the same name?
	- eg, `i = 0; foo.i = 'eye'` ; now `i` is an integer with an option `foo`, isn't that weird?
		- and then when `i += 1`, does `foo.i` stop working?
		- no, because `i` will only be reassigned in the current scope, not in the special "function scope"
	- or worse: `greeting[str name]: ...; person.greeting = 'Ryan'`;  now we have a function `greeting` with a function option `str name` *and* a `person` key yielding `Ryan`.  Isn't that even stranger?
	- **maybe that's okay**, actually, since I don't plan on making any pili functions to reveal all the options of a function (at least not any intended to be used regularly) and those functions (`i` and `greeting`) can still be used as expected in every context I can think of
	- (I could also separate the dictionaries for properties and names, but then that breaks the equivalency of `foo.bar <=> bar[foo]` and therefore kinda defeats the purpose of doing it this way)
- MRO issues: 
	- suppose you have an object `foo` that inherited from one or more types with a custom `len` attribute.
	- Then calling `foo.len` or `len[foo]`, 

```
A = 
	[]:
		none
	.len = 1
B = 
	[]: 
		none
	.len = 2

C = 
	inherit A, B

inst = C[]
print inst.len
>>> ????


len = 
	[A obj] = 1
	[B obj] = 2
	[list obj]:
		...
	...



```

#### Proposal 3
Single scope for all variables.  

`bar = 5` means... what?
- it means, when in a map block called `foo`, the same thing as `bar[foo] = 5`
- what if `bar` is already a record in a higher scope?  Like `bar = 1` in the global scope?
- that's fine.  That just means `bar[root] == 1 and bar[foo] == 5`
- so, while in the scope called `foo` then any reference to `bar` will be evaluated as `bar[foo]`
- what if I literally type the expression `bar[foo]` ... then how does `bar` get evaluated?
	- well, if we follow the above rule, we get a circular reference to evaluate: `bar => bar[foo] => bar[bar[foo]] ...`
	- so, I guess we would have to make an exception for function call syntax
	- so `bar` => `bar[foo]`
	- but `bar[foo]` just stays the same
	- that's super confusing
	- what about this:
		- `bar` => `global[#bar, foo]`
		- `bar[foo]` => `global[#bar, foo][foo]`
		- so then I guess names are stored in the location that's the hash of the tuple of the symbol and scope.
		- nah, this eliminates the equivalence I was seeking in the first place

Hmm

If there's only one scope for variables, then does that mean I must sacrifice closures and local variables?  Basically, the answer is yes.  So can I do this like [[#Updated Proposal]] and have regular scoping for bare names, but reverse map scoping for other names?

```
bar = 4
map foo
	bar = 1

print bar
print foo.bar
print bar[foo]
```

In this case, I could live with `bar` being  a sneaky double-agent kinda map, where it carries the value 4 when mentioned, but also maps values like foo.

But what about this example?
```
map foo
	bar = 1

map bar
	foo: 5

print bar
print foo.bar
print bar[foo]
```

The only way to do this would be to embed a "secondary mapping" within each record.

> [!summary] Conclusion
> OKAY, I'm finally ready to completely reject these proposals.  `foo.bar` really should **not** be equivalent to `bar[foo]`.  

### Discussion: Re-merge Maps, Classes, and Traits?
- observation: classes and traits are both maps
	- 
### Reimagining Types & Patterns
#### Types vs Patterns
- a type should be viewed as the possible space for a given value (including composite values, ie, product types and sum types)
- a pattern is a construct that matches a sequence of zero or more values, based on type and other things
- so, a "union" pattern should actually be a type
- but patterns have several things that types do not have:
	- an ordered sequence of elements
	- name-binding
	- quantifiers
	- fn guard / expr guard

#### class Type System
All values are organized into classes.  classes consist of fields and records.

In this system, the dot operator will no longer play triple duty.  Before, the three uses of dot were:
1. calling an option (ie, `foo.bar` <=> `foo["bar"]`)
2. ^this construct also stood in place for property access
3. calling a function (ie, `foo.bar` <=> `bar[foo]` and `foo.bar[5]` <=> `bar[foo, 5]`)

Now, the dot operator will lose (2) the ability to call options (which was kinda excessive) and instead play the double duty of:
1. accessing a slot/formula (no other syntax available for this job)
2. calling a function (same equivalency as (3) above, except the square brackets will be required)
	- ie `foo.bar[]` is equivalent to `bar[foo]` (as long as no `bar` field exists on foo) but NOT equivalent to `foo.bar` which ONLY access the `bar` field, and results in an error if none exists

##### Fields
- a class field has the following properties:
	- name
	- type
	- default: 
		- default value  OR 
		- function/formula/calculated field
- pseudo fields AKA dot-options

###### Types and Dimensions of Fields and Dot Function
- Slot
	- with default value (`blank` if missing)
	- with default formula
- Getter
	- the API looks like a slot
	- essentially it is a dot-function with no extra arguments
- Action
	- a dot-function that does something, returns nothing
- Dot-Function
	- any arbitrary function that takes the record as it's first argument

So, the last three could all be combined into one category, and not necessarily conflated with "fields".

Ok, but there is still some advantage in a special category for "formula".
- calling a formula is the same syntax as retrieving a property (clean interface) 
	- this could/should be differentiated from dot-functions
	- eg `my_list.len` vs `my_list.len[]` or `my_list.last_item` vs `my_list.copy[]`
- resembles "calculated field" or "formula" in database analogy
- resembles a *getter*
	- could also allow implement of an optional *setter*

Reasons against a special "formula" category:
- added complexity to the "Field" class
- ... should a class keep track of the dot-functions defined on it?  Should all of them be called psuedo-fields?
	- I can't really think of a reason to do that, given my current implementation of dot-functions
	- maybe just to get a list of methods for some reason?  For copying a whole class?

However, maybe a formula should not be a sub-class of field.  Or maybe it's fine.  I don't know.

##### class Anatomy
What things go in a class definition?
- slots
- formulas
- dot-options
- hericlass options
- class options 
- constructors

These are categorized like so:
- directly hericlass:
	- slots
	- formulas
- hericlass through dot-option patterns:
	- dot-options
	- hericlass options (under `.call`)
- for class only:
	- class options
	- ... incl. constructors

##### Records
- each record has one value for each field, even if that value is `None`.

##### Built-in Fields
classes have a few fields built-in with default values or formulas.  The most important one is "key".  The 'key' field, if left undefined, defaults to an `int` field with a formula that increments every time a record is added to the class.

##### Built-in classes
- string class, boolean class, int, float, ratio
	- key-only class: hash of python value
- None class
	- consists of only one value: None, key=0
- List class
	- no hashing of lists in python...
	- key: regular default incrementing
	- no other fields... or maybe some fields like `len` 
	- oh, hold on a second, I guess lists should actually be implemented as classes themselves (but not tuples or sets?)
- class class
	- class of all classes
	- key: class name?  No, allow anonymous classes/lists
- Pattern class
- Function class
	- fields:
		- key: function name?
		- options:
			- signature: pattern
			- code-block
		- closure
	- ... or maybe a "call" pseudo field?
- Option class:
	- fields:
		- signature: pattern
		- block: code-block

##### Slices
A slice object is an object that shares the same fields as its parent class (and maybe extended fields?) but only a subset of the records.   The subset can be defined in three different ways, and therefore there are three different types of slices.
- pattern slice: all records in a class that match a given pattern
- filter-function slice: all records in a class that return a truthy value given a function
- manual slice: a slice that contains no records by default, but that can be added to manually
	- this is subtly different to just forming a list of records from one class
	- this one is the most like subclass

Slices also have additional properties.  In particular, a slice has a `parent: class` property, and an `extended_fields: Field*` property.

###### Slice Syntax
```pili
class Bird
	slot species <Species> 
	formula call <blank>:
		print self.species.melody

Penguin = Bird.slice[<species='penguin'>]
## OR
slice Penguin
	formula habitat <str>: "Antarctica"
	formula waddle <blank>:
		1 < 2 & 3 > 2
```

#### Types
- so a type is one of:
	- a class (product type)
	- a union (sum type)
- or maybe make all types sum types where some of them are len=1
- well, I guess *patterns* can be like that.  But a *type* is just gonna be a class.

#### Patterns
pattern ::= parameter | parameter "," pattern
parameter ::= matcher (name | "") quantifier
matcher ::= (class | value | any) guard? fn?
intersection-matcher ::= slice+ guard? fn?
name ::= alpha +
quantifier ::= "" | "?" | "+" | "\*"

examples:
```
(list & callable)+
^^^^^^^^^^^^^^^^^
    matcher
```

So a monad is one of:
- class
- value
- union of parameters
- intersection of patterns

A parameter is a monad with:
- quantifier

```
int
int|str
list & callable | blank
(list & callable)+ | blank
 => [1,2], []
 => blank

## Is there a valid use case for union parameters?  YES
(list & callable) fnls+ | blank, num
(num, int) | (int, num)
>>> both of these could/should be made into separate options.  Respectively:
>>> 1. 
>>> 	1. (list & callable) fnls+, num
>>> 	2. blank, num
>>> 2. 

int | float | "other" | Record(type~int), str

int n | float f

int|float n

(int|str)&(int+
```

#### Inheritance?
Records inherit a few specific things from classes:
- slots
- formulas
- dot-options

And that's it.  Regular options are not inherited, nor any other properties, hidden or otherwise.

- probably won't do regular inheritance
- but might do "class duplicating"
	- be careful about dynamic modification of classes, because changes may not apply to classes that have already been duplicated
- and/or composition
	- like a class has a field with a pointer to a record to another "parent" class 
	- (probably not ideal in most cases)
- class templates
	- allows fields (and rows?) to be shared between classes
	- if rows: somehow need to make sure the keys don't overlap 
- related: filtered views


So we have three levels of abstraction.  
1. Ad-hoc functions (objects) can be defined
2. classes (templates for objects) can be defined
4. Traits (templates for classes) can be defined 
There should be clear, consistent, easy syntax for all three levels.  

- in a function, you can define slots, formulas, options, and dot-options.
- in a class, you should be able to define all of those things both for the class, and also for the template.  The constructor (unique to the class, usually, is an option of the class)
- in a trait, usually you just want to define things to be inherited by the instance, but I guess it could be fun to modify class behaviour too

- one idea: 
	- for the current level (ie `Context.env.fn`), define those fields and options directly
		- dot-option patterns will start with a value matcher
	- for the instance level (ie, the instance) define the fields with the keywords: `slot`, `formula`, and `opt`.  
		- dot-option patterns will start with a class matcher, value matchers for classes/traits must be explicitly specified
	- for the class level (ie, properties that traits give to *classes*), some other syntax will be required... like `metaslot` ... so the class itself will gain a new slot... but let's not worry about that one for now


#### filtered class/virtual class
- b;

#### Garbage Collection
- I may need a way to delete old records that no longer have names... otherwise memory could get eaten up pretty quick

#### Syntax
```python pili
class Dog:
    # slot field
    slot name:
        str
        ""

    # formula field
    formula bark[]:
        print "Helo, my name is {self.name}!"

    # dot-option
    .eat[Food food]:
        del food
        self.full = True

	slot property_name (type_expression) = default_value_expression
	OR
	slot property_name (type_expression) = 
		default_value_function_block

	# example
	slot start_date (Date) = 
		return Date.Today

	formula prop_name (type_expression):

	slot start_date as Date | blank

	slot start_date <Date | blank>: Date.Today
	formula start_date <Date | blank>:
		run + this + code
		return self.end_date - self.duration

	start_date <Date | blank> = { Date.Today }
	start_date <Date | blank>:
		run + this + code
		return self.end_date - self.duration
	start_date.setter = 
		[str value]:
			
```


### Issues, room for improvement
#### Scope and Closures
> [!NOTE] Update
> So it turns out pili has actually had closures all along, I just didn't realize it until I fixed the (potentially surprising) feature/bug of pili wanting to modify global variables by default.  When I changed it to just create shadow variables instead, all of a sudden it has closure behaviour.  But I'm too confused to test it thoroughly.  it just seems to magically work now.

- currently, a statement like `foo = 5` will default to the global `foo` if it exists.  Sometimes this may be desired behaviour, but most of the time, you should assume the programmer wants to create a shadow of foo.  
	- How do other langauges solve this problem?
		- Python solves this problem by defaulting to shadows unless you use the `global` keyword to specify otherwise
		- Javascript solves this problem by requiring variable declaration (var, let, or const)
	- potential solutions for Pili:
		- Two requirements:
			- The code must be readable, it should be very easy to tell the scope of a variable
			- the programmer should have ergonomic control of the scope of each variable
		- Solution 1: copy python
			- quick and easy to write pili code
			- cons: it seems strange that there is different behaviour for *setting* and *getting* variables
		- Solution 2: require declaration
			- gives more power/flexibility to programmer
			- opens the door for other option flags (eg `const`, `alias`, type hints, something else?)
		- Solution 3: always shadow (ie, only local variables)
			- simplest to read and understand
			- functions are more "pure"
			- requires passing all necessary variables as arguments (no modifying variables from outer scopes)
			- harder to make a function factory?
			- *but still allow variable search to ascend the prototype chain*

Compare and contrast: 
```python
def factory(repeats: int):
	def p(msg):
		for i in range(repeats):
			print(f"{i}: {msg}")
	return p

factory(3)("hello")
> 0: hello
> 1: hello
> 2: hello
```

```python pili
## Solution 3: Always Shadow
factory[int>=0 repeats]:
	p[any msg]:
		for i in range[reps]
			print["{i}: {msg}"]
	p.reps = repeats
	return p

## Solution 2: require declaration
factory[int>=0 repeats]:
	return
		local repeats = repeats
		[any msg]:
			for i in range[repeats]
				print["{i}: {msg}"]

factory[3]["hello"]
> 1: hello
> 2: hello
> 3: hello
```

#### `self` and `args` keywords
- it's kinda confusing... in some (most?) contexts, `self.prop` and `prop` are equivalent.  In other contexts, they both work for retrieval of values, but `prop = 4` will shadow prop.  Other times, `self` doesn't even refer to what you want it to refer to
####  Types
- Classes, types, prototypes, inheritance
- type tree and option tree
- **I have a sinking feeling that the prototype tree is going to break down somewhere**
	- why does the stack trace look suspiciously similar to the prototype ancestry line?
- Prototypes vs type-tags:
	- The current model is a prototype model, where all values are functions, and all types are also functions.  So `int` is a value, and also the prototype for `1`.
	- However, I feel like 99% of the time, you want to separate classes (types) from values.  
	- Maybe I should explore the idea again of "type tags" — little pieces of data that describe the capabilities of an object.  Some tags might be "numeric", "iterable", "princlass", "lengthable" ^type-tags
	- could simplify pattern-matching
	- One Major **advantage** of Type-Tags:
		- if types are indeed a separate kind of entity from other values, then it becomes much simpler to separate 'param list' from 'arg list'.  
	- **Disadvantages:**
		- but I guess it's still not trivial if you want to retain the ability to pass types as arguments in some cases
		- and you still have to deal with quantifiers either way
		- and how to form the MRO with just a flat list of tags?  ordered list of tags?
	- 
- Unify TYPES and PATTERNS
	- so `int` is a pattern, not a prototype
	- so then the listpatt of options can be: 
		- a value (dictionary-like) (what was "ValuePattern" before)
		- a type/pattern (what was "Prototype" before)
		- a union (like before)

Pattern
- value pattern
- prototype pattern
- union
- any
- listpatt

But I could make it more like this: a pattern expression consists of tags and operators:
- tags are: 
- operators:
	- comma (`,`): indicates sequence
	- quantifiers (`+`, `*`, `?`, etc): also indicate sequences, but in a specific sense
	- sub-pattern (`@`): adds guard expression to a pattern
	- union (`|`): alternative patterns
	- intersection (`&`): must match both patterns

##### Multiple Inheritance
> [!NOTE]
> I just had a lengthy conversation with Bing about multiple inheritance, the "method resolution order" (MRO) and the C3 linearization algorithm for computing the MRO.  C3 sounds pretty smart, but I think I actually prefer a depth-first search algorithm.

Example to illustrate the difference:
```python
class X:
    def foo(self):
        print("X.foo")

class Y:
    def foo(self):
        print("Y.foo")

class A(Y): pass

class B(X, Y): pass

class Z(A, B): pass

print(Z.__mro__)
## (Z, A, B, X, Y, object)

z = Z()
z.foo()
## python prints "X.foo"
## but DFS would print "Y.foo", even if B had a foo method

"""
 C3 => Z A B X Y object
DFS => Z A Y object B X
"""

```


##### Pili-izing Pattern Objects
So the thing is, of course values can be patterns and patterns can be values.  But if want to just "use values *as* patterns" then how do you handle guards, list-patterns, sub-patterns, and quantifiers?

Well, currently all those things are python constructs and pili is unable to inspect them.  But I guess I could make all of them pili constructs.
```python
Parameter = 
	[quantifier?]:
		quantifier ??= ""
	
	ValuePattern[any value]:
		value = value
		['~'][any val, ValuePattern patt]:
			return val == patt.value

	Prototype[any proto, fn guard?, expr sub_patt?]:
		prototype = proto
		guard = gaurd ?? None
		sub_patt = sub_patt ?? None
		['~'][any val, Prototype patt]:
			t = type(val)
			match = t and (t == patt.prototype or t ~ patt.prototype)
			if not match
				return false
			if guard and not guard[val]:
				return false
			if sub_patt and not eval[sub_patt]


	Union[Parameter patt+]:
		pass
	
```



#### Effects as Values
- Consider:
	- each statement produces an "effect", but that effect is actually a class of a pili value.  
	- then the function executor can decide what to do with that effect
		- break a loop, stop and return, print a string, etc
- advantages & disadvantages?
	- advantage: a list like `[1, 2, flag=True]` can be passed to a function as arguments without the last arg losing meaning before it gets there

#### Performance and Option Hashing
- I tried implementing option hashing, so that every function also doubles as a python-like dictionary 
	- ... but for some reason it didn't seem to improve the speed... bugs?

```python pili
foo = {}
foo.name = 5  # ValuePattern(name)
foo.name : 5  # ValuePattern(name)
foo['name'] = 5  # ValuePattern(name)
foo['name' name]: 5  # ValuePattern(name) name
foo[str name]:  # Prototype(str) name
foo[2, 3] = 23  # ValuePattern(2), ValuePattern(3)
foo[2|3] = 23
```



***

### Implementation
#### Abstract Syntax Tree: AST
So what tokens need to be grouped into nodes?  Well, some groups should be done in a recursive manner, and others should be done with a pair of stacks for terms and operators.  And other groups I'm not sure yet.

Recursively groupable
- parentheses
- brackets
- braces
- blocks
- either prefix or postfix operators 
	- but not both at once, because that requires comparing precedence
- commands (acting as prefix operators that have looser precedence than all postfix ops)

All of these should be split by commas, except for blocks which are split by newlines.

Stackly groupable
- operators (prefix, infix, and postfix operators)

Other:
- if-else expressions
	- maybe this could be integrated into mathological?  It would take some logic
- keyword special syntax expressions
	- eg, if, while, and for all require special syntax... as well as slot, formula, and setter
- certain expressions get 
```

```
##### Expression Parsing
So I have linear list of tokens.  Currently, how I handle that is in a few steps:
- tokenize
- groupings:
	- brackets
	- lines
	- blocks

I could have an intermediate representation on the second pass:
- `Statement: list[Node], stmt_type: empty, assignment, if-else, etc`

`a * - 4`
`*: postfix=14, binop=13`
`-: binop=12, prefix=13`

`if *.postfix and *.binop and -.binop and -.prefix: AMBIGUITY ERROR`
`if `

advantage of doing three passes (token -> node -> expression) instead of just two (token -> expression)
- the second pass does groupings, which is not strictly necessary, but it does help to break up the logic
- the type of statement, and therefore the AST format of the first part of the statement may be influenced by a later part in the statement
	- eg variable assignment means I must interpret the first half as a pattern
	- option assignment means I must interpret the square brackets as param-pattern.
	- if-else infix operator
- easier to debug with simpler steps

What are the advantages of doing two passes instead of three?
- efficiency
- elegance

##### Hierar


#### NEXT: 
- implement `inherit` command
- enhance "self" option:
	- when does it refer to the running function and when does it refer to the first argument?
		- dot_options: first arg
		- default: running function
		- non-dot-method: ???
#### Python Classes
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


##### Statement Types
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

##### Value Types: the types that go into and out of functions and statements
none = 'none'  
Boolean = 'bool'  
Integer = 'int'  
Float = 'float'  
String = 'str'  
Function = 'fn'  
List = 'list'
Option?
Block? — just made it a function

##### No more buffer types
Buffer Types: the structures that exist between values
- Parameter
- Option
- Block

#### Assignment
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

#### Operators
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

##### Expression Tree Builder
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

##### Dot Operator
The dot operator is like a super operator.  It gets properties of objects, calls methods, and calls virtual options.  The dot-op is implicitly inserted everywhere the syntax `<name><list>` occurs.  eg `foo[a, b]` => `foo.[a, b]`.

The dot-op eval_args function does the following: 

These are the options of the dot-operator Function:
- Function, Value -> calls the function with the value as its only arg (useful mostly just for derefing names)
- Function, List -> calls the function using the elements of the list as args
- Pattern, List -> makes a pattern with a specific guard
	- question: should this be an option of `dot` or just make options on those basic types?

Hierarchy of dot call:
- `function.option[args]`

Possibilities for . calls:
- expression . name . name
- expression . name
	- => deref => virt_op
- fn . list
	- => call it
- fn . name
	- => deref => virt_op
- expression . name . list
	- => deref => virt_op => call

```
foo.bar.string
foo.string.bar
```

***
#### Pattern Matching
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
## pattern for my Option Class
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

## simple interface
Option[@Pattern pattern, @Value value]:
	pass

## explicit interface
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
## is equivalent to:
[num_leg ~ int and name ~ str]

Dog[int num_legs, str name]:
	bark: do something

@Dog

```

##### Pattern Matching Algorithm
1. Generate the NFA state machine for the pattern
	- a collection of states
	- a state is/has:


- a simple state machine is one literal character:
	- (s0) --a--> ((s1))
- + plus quantifier
	- has two arrows: one pointing back to the start state, and another pointing to success
	- (s0) --[m]--> 

```python
class State:
	""" a state has EITHER a pattern and success, OR two branches """
	pattern: Pattern | None
	success: State | None
	branches: tuple[State, State] | None

	def __init__(self, patt, success, branches):
		pass

class NFA:
	states: set[State]
	def run(self, args: list[Function]):
		i = 0
		while True:
			next_states = set([])
			for state in self.states:
				if state.pattern.matches(args[i]):
					if state.success == SuccessState:
						return "Success"
					next_states.add(state.success)
					i += 1
				if state.branch:
					if state.branch == SuccessState:
						return "Success"
					next_states.add(state.branch)
			if len(next_states) == 0:
				return "Failure"
			self.states = next_states

class NFA:
	start: State
	states: set[State]
	
	def run(self, args: list[Function]):
		self.states = set([self.start])
		for i, arg in enumerate(args)
			current_states = union(self.states, self.states..branch)
			next_states = set([])
			for state in current_states:
				if state.pattern.matches(arg):
					next_states.add(state.success)
			self.states = next_states

	def __init__(self, *params):
		# self.start = State(0)
		if len(params) == 0:
			self.start = Success
		p = params[0]
		next = NFA(params[1:]).start
		this = State(p.pattern, next)
		match p.quantifier:
			case "":
				self.start = this
			case "?":
				self.start = State(branches=(this, next))
			case "*":
				self.start = State(branches=(this, next))
				this.success = self.start
			case "+":
				self.start = this
				this.success = State(branches=(this, next))

		self.start = this if p.required else State(branches=(this, next))
		this.success = 
		
				
```


###### Example
regex: a+b
string: aaab

```
State(0):
	pattern: a
	success: State(1)
	branch: None

State(1):
	pattern: a
	success: State(2)
	branch: self

State(2):
	pattern: b
	success: Success
	branch: None


Steps: 
1. active: {0}
	- i, arg : 0, a
	- a matches State(0).pattern => add State(1)
2. active: {1}
	- i, arg : 1, a
	- a matches pattern => add State(2), and keep State(1)
3. active: {1, 2}
	- i, arg : 2, a
	- State(1): a matches pattern => add State(2), keep State(1)
	- State(2): fail
4. active: {1, 2}
	- i, arg : 3, b
	- State(1): fail
	- State(2): b matches State(2).pattern => Success State
```

###### Compiling the NFA
pattern: a+b

1. init: create State(0) with first pattern and match quantifier:
	- `+` — 
	- `?`
	- `*`


```
NFA:
	def __init__(self, *params)
	self.start = State(0)
	if len(params) == 0:
		self.start = Success
	p = params[0]
	match p.quantifier:
		case "":
			self.start.pattern = p.pattern
			self.start.success = NFA(params[1:]).start
			self.start.branch = None
		case "?":
			self.start.pattern = p.pattern
			self.start.success = NFA(params[1:]).start
			self.start.branch = self.start.success.start
		case "*":
			self.start.pattern = p.pattern
			self.start.success = self.start
			self.start.branch = NFA(params[1:]).start
		case "+":
			self.start = NFA(params[1:]).start
			self.start.success = State()
			self.start.success.branch = self.start
			self.start.pattern = p.pattern
			self.start.success = NFA(params[1:]).start
			self.start.branch = self.start.success.start
			

```

##### Option Tree
I want to explore again the idea of creating an option TREE for each function, rather than a hashed dictionary of options and a list of options.  Because I feel like there should be a way to do something similar to hashing for *all* options, if not exactly a hash class.

But the thing is, it doesn't really work with quantifiers.  

But for individual parameters, imagine something like this: 
```
## eg for pattern `int` and a given value:
opt = Option[int, value]
option_tree: dict[class, ...] ??= {}
option_tree[int] ??= (None, {}): tuple[<option matching type int>, dict[Value, Option]]
option_tree[int] = opt

## eg for pattern `3`
opt = Option[3, value]
option_tree
option_tree[int]
option_tree[int][3]

```

I don't know... again, I guess it works fine for just types and values, but not for multi-parameter patterns...

#### New Function Class
```python
class Function:
	name: str  # mostly for debugging
	types: list[Function] # multiple inheritance  
	options: list[Option]  
	hashed_options: dict[tuple[Value,], Option]  
	args: list[Option]  
	block: Block  
	closure: Function
	return_value: Function  
	value: val_types  
	is_null: bool
```


How do I want to define classes, instances, and types?

```python pili
Person =
	label 'Person'
	name = ""
	age = 0
	[str name, int age]:
		label 'p{name, age}'
		None
	birthday[]:  # class method
		label 'birthday instance'
		debug self.age += 1
		return age
		# > Context.env: [] # currently running instance
		# > Context.env.env: birthday
  #       > Context.env.env.env: instance
	birthday.name = 'birthday'

p = Person['Ryan', 30]

print debug p.birthday[]

Employee = 
	inherit Person, OtherClass
	: int id
	[str name, int age]:
		pass

Employee[str name, int age]	:
	inherit Person, OtherClass
```

- all `[]:` options are constructors
- others are class properties or methods

I have a `self` problem.  When the expression `p.birthday[]` is called, I lose all access to the instance p.  `p` just gets me to the birthday method and then disappears.  So `self` only works with dot-options.


What are the potential function definition cases?
```python
.[args].[params] :
.name.[params] :
.name :  # should be equivalent to .name[]
name :
name.[params] :
expr.name :
expr.name.[params] :
~~expr.[args].[params] :~~
type_expr name :

after popping last node and dot => 
.[args]
.name

name
expr.name
expr.[args]

```


So, 
1. Pop first node if dot:
	1. If dot_option: Skip 3.1., (and eval step 3.2 in root env?  No)
	2. also skip step 2 if last node is not list 
2. pop the last node; it will give us the **option pattern**
3. pop the next node; it will give us the function name
	1. if is name:
		1. Evaluate the rest of the list to get the context
			1. If none: env = Context.env and definite=False
			2. Else: env = expr Val, and definite= True
		2. Find name in env
			1. if exists, use it
			2. if not exists, create it and use it
	2. if [args] or any other expression,
		1. Evaluate all nodes, including last one
		2. That is our fn, raise error if missing
4. Now we have fn and pattern and dot_option; Return those values as tuple

#### Functions & classes (and traits, maps, prototypes, etc)
- Function(Record, OptionMap)
- class(Function, FieldMap)
- Trait(Function)

```pili
trait record
	slot data list[record]: []
	.get[str name] -> record:
		...
	.set[str name, record value]:
		...

trait option_map
	slot option_list list[option]: []
	slot option_dict dict[tuple, option]: {}

trait field_map
	slot getters
	slot setters

trait trait
	slot fields list[field]
	slot options list[option]

trait function
	slot slot_dict
	slot formula_dict
	slot setter_dict

class Record @record

class Function @function @option_map

class class @function @field_map @option_map

class Trait @trait @function @option_map
```