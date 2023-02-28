greet = "hello world"
print 'greet = "hello world => ' + str[greet == 'hello world']
greeting[str src]: 'Hello '+src
debug
print greeting['srcsasdfads']
exit

# ***********************
my_var[int num]:
	prop1 = "hello"
	prop2 = "world"
	return num + 5
my_var[str text]: text + '!'
next_var =
	str text: text + '?!?!'

my_var[3] = "three"
print 345 ~ type["345"]
print "****************"
debug
Dog[str name, str breed]:
	bark: print "hello my name is " + name
spot = Dog['Spot', 'terrier']
debug
spot.bark
print spot.breed == "terrier" or spot.name == "Spot"
print spot.breed != "terrier" or spot.name == "Spot"
print spot.breed == "terrier" and spot.name != "Spot"
print spot.breed != "terrier" and spot.name == "Spot"
exit
# ***********************

coolFunc[str input: 345, int n: 43]:
    0: "zero"
    "string_key": 5
    string_key: 5   // equivalent to above
    int: 3 + 3      // key is not string "int", rather the type `int`
    int<0: "negative integers"
	return input[n]

myFunction[0]: "zero"
myFunction["string_key"]: 5
myFunction["string_key"]: 5
myFunction[int]: 3 + 3
myFunction[int<0]: "negative integers"

coolFunc["input"]

Token[string src] =
	string[r'\s*'] src: "SPACE"
	string src: "other string"

print Token[' ']   # returns the above executed function exactly

Token =
	string[r'\s*'] src: "SPACE"
	string src: "other string"
	string src, int n: src + " " + str[n]

print Token[' '] == "SPACE"


# So maybe what I would want is this:
Token[string src]:
	_Token =
		str[`\s*`] src: "SPACE"
		str src: "other string"
		any: None
	partial_result = _Token[src]
	if partial_result
		return partial_result
	else
		return "some default"

return 5
