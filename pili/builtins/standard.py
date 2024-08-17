from .base import *
from ..utils import IndexErr
import json
import yaml

print(f'loading {__name__}.py')

##############################################
# min/max
##############################################
def pili_max(iterable):
    iterable = iter(iterable)
    biggest = next(iterable, None)
    if biggest is None:
        return BuiltIns['blank']
    for item in iterable:
        if BuiltIns['>'].call(item, biggest).truthy:
            biggest = item
    return biggest
state.deref('max').op_list[0].resolution = pili_max


def pili_min(iterable):
    iterable = iter(iterable)
    least = next(iterable, None)
    if least is None:
        return BuiltIns['blank']
    for item in iterable:
        if BuiltIns['<'].call(item, least).truthy:
            least = item
    return least
state.deref('min').op_list[0].resolution = pili_min


def transmogrify(ref: Record, data: Record):
    ref.__class__ = data.__class__
    ref.__dict__ = data.__dict__
    return ref

state.deref('transmogrify').op_list[0].resolution = transmogrify


##############################################
# files
##############################################
def read_file(arg: Record, lines=BuiltIns['blank']):
    with open(arg.get('path').value, 'r') as f:
        if lines == BuiltIns['true']:
            return py_value(f.readlines())
        else:
            return py_value(f.read())
state.deref('read').op_list[0].resolution = read_file


##############################################
# regular expressions
##############################################
def regex_constructor(s, f=py_value('')):
    return RegEx(s.value, f.value)
state.deref('regex').op_list[0].resolution = regex_constructor
state.deref('RegEx').op_list = state.deref('regex').op_list[:]

def regex_flags(regex: RegEx | PyValue[str],
                a=BuiltIns['blank'], i=BuiltIns['blank'], m=BuiltIns['blank'],
                s=BuiltIns['blank'], x=BuiltIns['blank'], l=BuiltIns['blank']):
    flags = 0
    if a.truthy:
        flags |= re.RegexFlag.ASCII
    if i.truthy:
        flags |= re.RegexFlag.IGNORECASE
    if m.truthy:
        flags |= re.RegexFlag.MULTILINE
    if s.truthy:
        flags |= re.RegexFlag.DOTALL
    if x.truthy:
        flags |= re.RegexFlag.VERBOSE
    if l.truthy:
        flags |= re.RegexFlag.LOCALE
    if not flags and isinstance(regex, RegEx):
        for f in regex.flags:
            flags |= getattr(re.RegexFlag, f.upper())
    return flags

def regex_extract(regex: RegEx | PyValue[str], text: PyValue[str], *,
              a=BuiltIns['blank'], i=BuiltIns['blank'], m=BuiltIns['blank'],
              s=BuiltIns['blank'], x=BuiltIns['blank'], l=BuiltIns['blank']):
    flags = regex_flags(regex, a, i, m, s, x, l)
    return py_value(re.findall(regex.value, text.value, flags))
state.deref('regex').frame['extract'].op_list[0].resolution = regex_extract

def regex_split(text: PyValue[str], sep: RegEx, maxsplit: PyValue[int] = py_value(0), *,
                a=BuiltIns['blank'], i=BuiltIns['blank'], m=BuiltIns['blank'], s=BuiltIns['blank'], x=BuiltIns['blank'], l=BuiltIns['blank']):
    text = text.value
    flags = regex_flags(sep, a, i, m, s, x, l)
    return py_value([py_value(s) for s in re.compile(sep.value, flags).split(text, maxsplit.value)])
state.deref('split').op_list[0].resolution = regex_split
state.deref('split').op_list[1].resolution = lambda txt, sep: py_value([py_value(s) for s in txt.value.split(sep.value)])

def regex_match(regex: RegEx | PyValue[str], text: PyValue[str], start: PyValue[int] = py_value(1), end: PyValue[int] = py_value(-1), *,
              a=BuiltIns['blank'], i=BuiltIns['blank'], m=BuiltIns['blank'], s=BuiltIns['blank'], x=BuiltIns['blank'], l=BuiltIns['blank']):
    text = text.value
    flags = regex_flags(regex, a, i, m, s, x, l)
    idx = start.__index__() if start.value else 0
    if idx < 0:
        idx += len(text)
    end_idx = end.value or -1
    if end_idx < 0:
        end_idx += len(text)+1
    return PyValue(BuiltIns['Match'], re.compile(regex.value, flags).search(text, idx, end_idx))
state.deref('match').op_list[0].resolution = regex_match

MatchTable: Frame = state.deref('Match').frame
MatchTable['group'].op_list[0].resolution = lambda self, *args: py_value(re.Match.group(self.value, *(i.value for i in args)))
MatchTable['expand'].op_list[0].resolution = lambda self, template: py_value(re.Match.expand(self.value, template.value))
MatchTable['groups'].op_list[0].resolution = \
    lambda self, default=BuiltIns['blank']: py_value(re.Match.groups(self.value, default))
# MatchTable['span'].op_list[0].resolution = lambda self, group=py_value(0): py_value(re.Match.span(self, group.value))
MatchTable['start'].op_list[0].resolution = lambda self, group=py_value(0): py_value(re.Match.start(self.value, group.value)+1)
MatchTable['end'].op_list[0].resolution = lambda self, group=py_value(0): py_value(re.Match.end(self.value, group.value))


##############################################
# list functions
##############################################
def push(ls, item, index = BuiltIns['blank']):
    if index is BuiltIns['blank']:
        return ls.value.append(item) or ls
    return ls.value.insert(index.__index__(), item) or ls
state.deref('push').op_list[0].resolution = push
state.deref('pop').op_list[0].resolution = lambda ls, idx=-1: ls.value.pop(idx.__index__())

def extend_list(ls: PyValue[list], it: Record):
    ls.value.extend(it)
    return ls
state.deref('extend').op_list[0].resolution = extend_list

state.deref('join').op_list[0].resolution = \
    lambda ls, sep=py_value(''): py_value(sep.value.join(BuiltIns['str'].call(item).value for item in iter(ls)))
state.deref('join').op_list[1].resolution = \
    lambda sep, *args: py_value(sep.value.join(BuiltIns['str'].call(item).value for item in args))
def filter_function(seq: PyValue, fn: Function, *, mutate=BuiltIns['blank']):
    filtered = filter(lambda el: fn.call(el).truthy, seq)
    if mutate.truthy:
        seq.value[:] = filtered
        return seq
    return py_value(type(seq.value)(filtered))
state.deref('filter').op_list[0].resolution = filter_function
state.deref('filter').op_list[1].resolution = filter_function


##############################################
# JSON / YAML
##############################################
state.deref('parse_json').op_list[0].resolution = lambda text: py_value(json.loads(text.value))
state.deref('write_json').op_list[0].resolution = lambda fn: py_value(json.dumps(to_python(fn)))
state.deref('yaml').frame['load'].op_list[0].resolution = lambda text: py_value(yaml.safe_load(text.value))
state.deref('yaml').frame['dump'].op_list[0].resolution = lambda fn: py_value(yaml.dump(to_python(fn)))

