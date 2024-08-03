---
created: 2024-07-29 22:10
up: "[[Pili]]"
---
#2024/Jul/29 [[Pili]]
***
Pili is a dynamically typed language that has a similar feel to Python, but with a few major differentiating features.

One of the first you will notice is that Pili uses square brackets for function calls rather than parentheses.  This is in part because the concepts of key-access and function calling are mushed together in Pili.  Functions act as maps, namespaces, and containers for multiple-dispatch (overloaded) functions.  Pili has a powerful pattern-matching subsystem and makes use of this for defining and selecting overloaded definitions dynamically.

### Functions
This is how you write a function in Pili:
```python
function multiprint  
    repeats = 3  
    
    [any msg]:  
       for i in range[repeats]  
          print "{i}: {msg}"  
       return blank
       
   'hello': 'world'
```

`repeats` is a variable initialized with value `3` in `multiprint`'s scope.  It is accessible without dot syntax within the lexical scope of the function, and it can also be accessed (and mutated) outside that lexical scope with a statement like `multiprint.repeats += 1`.

`[any msg]: ` is a set of parameters (just one in this case).  The type is `any` and the binding is `msg`.  After the colon is an indented block of code that will be evaluated with a call like `multiprint['this is my message']`.  Together, these two elements make up what is called and **option**.  It's kinda like a key-value pair of a hash map or dictionary, but the "key" in this case could be a value, or it could be any — potentially highly complex — pattern of parameters.  And the "value" doesn't need to be a static value, it could be (as is the case here) a dynamically computed return value.

`'hello': 'world'` is also an **option** — a simple key-value option in this case.  This option can be called just like the previous one, but it will return the preset value.

Let's look at how the following code would be evaluated:
```
multiprint.repeats
# 3
multiprint.repeats -= 1
# 2
multiprint['this is my message']
# 1: this is my message
# 2: this is my message
multiprint['hello']
# 'world'
multiprint['hello'] = "hello back"
multiprint['hello']
# 'hello back'
```

Notice that the the key `'hello'` should also match the pattern `any`, but since it is a hashable value, it takes priority and that option is selected rather than the functional one.  In fact, basically all patterns take priority over the `any` pattern, which is generally only selected if none of the other patterns match.  All the patterns are sorted in order of priority — not in order of declaration — based on specificity of the pattern with value patterns being the highest, and `any` patterns being the lowest.

Of course, most of the time functions will be relatively simple and have only one option.  In that case, there is a shortcut syntax:
```
multiprint[any msg]:
	for i in range[repeats]
		print "{i}: {msg}"  
    return blank
```

Note though that this does not overwrite a previous declaration of `multiprint`.  if `multiprint` is already a function, this syntax simply adds a new option or redefines an existing option.  But if `multiprint` does not exist, this syntax creates it.

### String and Number Literals
Strings can be defined with single or double quotes and in both cases escape characters are accepted as well as string interpolation: any valid pili expression within unescaped `{}` will be converted to a string and inserted into the string in place.  Valid expressions even included other strings with interpolation.  Multiple strings in a row will be automatically concatenated.
```
print ("This is a string with a list: {range[1 to 3].list} and\n\talso "
	   "{"a " + 'substring containing more{blank || 'values'}'}!")

# output:
This is a string with a list: [1, 2, 3] and
	also a substring containing more values!
```

Strings can also be defined with one or more backticks.  These are super-literal strings where no escape sequences are possible.  If you need a backtick in your string, just surround your string literal with at least one more backtick than you have backticks in a row.
```
print `this \n will \t print \d literally`
# output:
this \n will \t print \d literally

print ```even
`newlines`
are read literally```
# output:
even
`newlines`
are read literally
```

Number literals are read basically how you expect, except that Pili defaults to rational numbers rather than floats.  Floats can be forced with the float function `float` or with the `f` suffix.
```
0.1 + 0.1 + 0.1
# 3/10
0.1f + 0.1f + 0.1f
# 0.30000000000000004
```


### Function Call
#### Universal Function Call Syntax
*AKA "dot-calling*"
Pili supports UFCS, even without brackets.  That means that `len[my_list]` and `my_list.len` are equivalent calls.  The same goes for `replace["colour", "ou", 'o']` and `"colour".replace["ou", 'o']`.  For multiple calls in a row, this can save space, reduce bracket nesting, and potentially make code more readable.  Let's illustrate with an example of a function.  an Armstrong number is a number that equals the sum of its individual digits, each raised to the power of the number of digits. For example, 153 is an Armstrong number because 1^3 + 5^3 + 3^3 equals 153. 

A function that checks if a number is an armstrong number may be written in python code like this:
```python
def is_armstrong(num):
    num_str = str(num)
    n = len(num_str)
    total_sum = sum(int(digit) ** n for digit in num_str)
    return total_sum == num
```

```pili
is_armstrong[int num]:
	num_str = str[num]
	n = len[num_str]
	total_sum = sum[num_str..(dig => int[dig] ^ n)]
	return total_sum == num
```

If this function was needed just once as an argument to another function, both of these could be rewritten as one-liners — python using the lambda notation, and pili using the ` => ` function operator.
```python
lambda num: sum(int(dig) ** len(str(num)) for dig in str(num)) == num
```

```pili
int num => n.str..(dig => dig.int ^ num.str.len).sum == num
```

For another (admittedly contrived) example, suppose you have a function `get_records` that takes a Database object and generates a list of records of type `Person`.
```pili
cities = (db.get_records 
			.filter[rec => rec.id is not blank] 
			.sort[rec => rec.order] 
			..name
			..lookup_address
			..city)
```

How does this compare to a similar solution in python?
```python
cities = (lookup_address_from_name(rec.name).city  
          for rec in sorted(filter(lambda rec: rec.id is not None,  
		                            get_records(db)),  
			                key=lambda rec: rec.order)  
          )
```

#### Variadic Functions, Named Arguments and Flags
It's very easy in Pili to define functions with a variable number of arguments.  Of course each option has its own signature, but even individual options can have variadic parameters.  Any parameter can be postfixed with regex-like quantifiers, namely `?`, `+`, and `*`.

```python
function add
	[num nums*, num acc]:
		for n in nums
			acc += n
		return acc
	[str texts+, str delimiter = '']:
		return texts.join[delimiter]
	[list lists+]:
		new_list = []
		for ls in lists
			new_list.extend[ls]
		return new_list
	[tuple tpls*]:
		if not tpls
			return ()
		return tpls.list.add.tuple

add[1, 2, 3, 4]
# returns 10
add['hello', 'to', 'the', 'world']
# in this case, the + quantifier consumes all the strings and so delimiter defaults to ''
# returns "hellototheworld"
add['hello', 'to', 'the', 'world', delimiter=' ']
# in this case, delimiter is specified explicitly
# returns "hello to the world"
add[[1, 2, 3], ['a', 'b', 'c']]
# returns [1, 2, 3, 'a', 'b', 'c']
add[]
# selects the only option that matches zero arguments: [tuple tpls*]
# returns () 
add[1, '2']
# raises NoMatchingOptionErr
```

In a function signature, a semicolon divides the ordered parameters from the name-only parameters.  At the call site, any parameter can be named, and the named argument's position is ignored; the rest of the arguments are matched in the relative order they appear.
```python
foo[int a, str b+; num c = 0, num d]:  
	return a, b, c, d

foo[a=-1, "one", "two", "three", d=20/30]
# (-1, ("one", "two", "three"), 0, 2/3)
foo[1, "two", c=55, d=1/2]
# (1, ("two",), 55, 1/2)


fsf[float f1*, str s*, float f2*]:
	return f1.?[1],  s.?[1],  f2.?[1]

fsf[-1.3f, 's', -0.122f]
# (-1.3, "s", -0.122)
fsf['s', -0.122f]
# (blank, "s", -0.122)
fsf[0.2345f, 1.2f, 3.0f]
# (0.2345, blank, blank)
fsf['STRING']
# (blank, "STRING", blank)
fsf['h', 0.1f, 0.2f]
# (blank, "h", 0.1)
fsf[]
# (blank, blank, blank)

rsnri[ratio r1?, str s?, num n*, ratio r2*, int i]:
	return r1, s, n, r2, i

rsnri[13]
# (blank, blank, (), (), 13)
rsnri[10/30, 'string', -1/2, 4/3, 1]
# (1/3, "string", (), (-1/2, 4/3), 1)
rsnri[345/2, 55]
# (345/2, blank, (), (), 55)
rsnri['s', 345/2, 55]
# (blank, 's', (), (345/2,), 55)
rsnri[345/2, 4.1f, 55]
# # (345/2, blank, (), (), 55)
# raises NoMatchingOptionErr
```

**Flags** can be defined in function signatures in the section for named parameters with the `!` operator like this: `foo[str param1; !my_flag, !another_flag]`.  The call-site follows the same syntax, but without the semicolon.  `!` is not a real operator — it's just syntactic sugar.  The following two definitions and calls are equivalent:
```
# function definition
greet[str name; !excited]:
	if my_flag
		return "Hiya {name}!!!"
	else
		return "Hello {name}."

greet[str name; bool excited?]: ...

# call-site
greet['Anthony', !excited]
greet['Anthony', excited=true]
```

###  Tables and Traits
Classes in Pili are called tables for hysterical raisins[^1].  Traits are like table components that can be combined to make tables.  Each value in Pili is a record of exactly one table.  Each table implements zero or more traits.

Traits and tables are also functions, and therefore can have variables and options defined in their body.  Unlike functions, traits and tables also have **fields**: slot, formula (ie getter), and setter.  The syntax for fields is as follows:
```
slot <slot name> <slot type>[ = <default value>]
formula <name> <type>:
	<body of function to generate value
setter <name>[<single parameter>]:
	<code to run when value is set>
```

Fields can overlap and overwrite each other to a certain extent, but Pili will check for compatibility.  So if a table attempts to implement two traits that each define a field "foo", but the type signatures are not compatible, it will reject it and raise an error.
```
trait animal 
	slot is_alive bool = true 
	slot warm_blooded bool 
 
trait mammal 
	formula warm_blooded bool: 
		return true 
	slot num_legs 
 
table Dog (mammal, animal) 
	slot name str
	slot num_legs = 4 
 
rover = Dog['Rover'] 
print rover 
# output:
Dog['Rover', 4, true, true]
```

#### Methods: dot-options etc
Any variable defined in a trait/table body is accessible via dot syntax, just like with functions.  Additionally, these values are also accessible by instances of the table.  If the value accessed in this way is a function, it will be supplied the instance as it's first argument.  This is how Pili does class methods.

```
table Dog
	slot name str

	bark[Dog self]:
		return "Hi my name is {self.name}"
	# the same as
	.bark:
		return "Hi my name is {self.name}"

	
```



## Other things
this quick start guide is far from complete... for more info, take a look at the readme.md and the code examples.


***
[^1]: Originally I was going to implement a range of functionality related to filter views and such, but kinda lost motivation to do that.