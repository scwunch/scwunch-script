test =
	0: 'test init success'
print test[0]

greet = "hello world"
test[1] = greet == 'hello world'
print 'test 1 passed' if test[1] == true else 'test 1 failed'

greeting[str src]: 'Hello ' + src
test[2] = "Hello Veronica" == greeting['Veronica']
print 'test 2 passed' if test[2] == True else 'test 2 failed'

my_var[int num]:
	prop1 = "hello"
	prop2 = "world"
	return num + 5
my_var[str text]: text + '!'
test[3] = my_var[1] == 10
print 'test 3 passed' if test[3] else 'test 3 failed'

test[4] = my_var[2] == 11
print 'test 4 passed' if test[4] else 'test 4 failed'

next_var =
	str text: text + '?!?!'
test[5] = next_var['hahahahaha'] == 'hahahahaha?!?!'
print 'test 5 passed' if test[5] else 'test 5 failed'

my_var[3] = "three"
test[6] = my_var[3] == 'three'
print 'test 10 passed' if test[10] else 'test 10 failed'

test[7] = not 345 ~ type["345"] and 345 ~ type[3333]
print 'test 11 passed' if test[7] else 'test 11 failed'

Dog[str name, str breed]:
	bark: "hello my name is " + name
spot = Dog['Spot', 'terrier']
test[8] = spot.bark == "hello my name is Spot"
print 'test 12 passed' if test[8] else 'test 12 failed'

test[9] = (spot.breed == "terrier" or spot.name == "Spot") and (spot.breed != "terrier" or spot.name == "Spot") and not (spot.breed == "terrier" and spot.name != "Spot") and not (spot.breed != "terrier" and spot.name == "Spot")
print 'test 13 passed' if test[9] else 'test 13 failed'

spot.name = 'Rover'
test[14] = spot.name == 'Rover'
print 'test 14 passed' if test[14] else 'test 14 failed'

test[15] = spot.bark == "hello my name is Rover"
print 'test 15 passed' if test[15] else 'test 15 failed'

Range[int start, int end, int step]:
	index = start
	next:
		return 'done' if index >= end else (index = index + step)
r = Range[0, 3, 1]
# print r.index
# print r.next
# print r.next
test[20] = r.index == 0 and r.next == 1 and r.next == 2 and r.next == 3 and r.next == 'done'
print 'test 20 passed' if test[20] else 'test 20 failed'

debug
Dog.bark_breed: "Hi, I'm a " + breed
test[21] = spot.bark_breed == "Hi, I'm a terrier"
print 'test 21 passed' if test[21] else 'test 21 failed'

exit
# ***********************

Dog =
	[str name, str breed]:
		bark: "hello my name is " + name
	bark_breed: "hi I'm a " + breed
















coolFunc[str input: "345", int n: 43]:
    0: "zero"
    "string_key": 5
    string_key: 5   // equivalent to above
    int: 3 + 3      // key is not string "int", rather the type `int`
    int<0: "negative integers"
    return input + str[n]

myFunction[0]: "zero"
myFunction["string_key"]: 5
myFunction["string_key"]: 5
myFunction[int]: 3 + 3
myFunction[int<0]: "negative integers"

coolFunc["input"]

Token =
	str[`\s*`]: "SPACE"
	str src: "other string"
	str src, int n: src + " " + str[n]

print Token[' '] == "SPACE"
print Token['blah'] == "other string"
print Token["my age is", 55] == "my age is 55"

# So maybe what I would want is this:
Token[str src]:
	_Token =
		str[`\s*`] src: "SPACE"
		str src: "other string"
		any: None
	partial_result = _Token[src] ?? false
	if partial_result
		return partial_result
	else
		return "some default"


