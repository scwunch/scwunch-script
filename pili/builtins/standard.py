from .base import *
from ..utils import IndexErr
import json

print(f'loading {__name__}.py')

def pili_max(iterable):
    iterable = iter(iterable)
    biggest = next(iterable, None)
    if biggest is None:
        return BuiltIns['blank']
    for item in iterable:
        if BuiltIns['>'].call(item, biggest).value:
            biggest = item
    return biggest
state.deref('max').op_list[0].resolution = pili_max


def pili_min(iterable):
    iterable = iter(iterable)
    least = next(iterable, None)
    if least is None:
        return BuiltIns['blank']
    for item in iterable:
        if BuiltIns['<'].call(item, least).value:
            least = item
    return least
state.deref('min').op_list[0].resolution = pili_min

def read_file(arg: Record, lines=BuiltIns['blank']):
    with open(arg.get('path').value, 'r') as f:
        if lines == BuiltIns['true']:
            return py_value(f.readlines())
        else:
            return py_value(f.read())
state.deref('read').op_list[0].resolution = read_file


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

def regex_match(regex: RegEx | PyValue[str], text: PyValue[str], start: PyValue[int] = py_value(1), end: PyValue[int] = py_value(-1), *,
              a=BuiltIns['blank'], i=BuiltIns['blank'], m=BuiltIns['blank'],
              s=BuiltIns['blank'], x=BuiltIns['blank'], l=BuiltIns['blank']):
    text = text.value
    flags = regex_flags(regex, a, i, m, s, x, l)
    idx = start.__index__() if start.value else 0
    if idx < 0:
        idx += len(text)
    # if not (0 <= idx < len(text)):
    #     raise IndexErr('fLine {state.line}: invalid index: {start}')
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






def extend_list(ls: PyValue[list], it: Record):
    ls.value.extend(it)
    return ls
state.deref('extend').op_list[0].resolution = extend_list

# BuiltIns['join'] = Function({ParamSet(SeqParam, StringParam):
#                              lambda ls, sep: py_value(sep.value.join(BuiltIns['str'].call(item).value
#                                                                      for item in iter(ls))),
#                              ParamSet(StringParam, Parameter(AnyMatcher(), quantifier="+")):
#                              lambda sep, items: py_value(sep.value.join(BuiltIns['str'].call(item).value
#                                                                         for item in iter(items))),
#                              })

state.deref('join').op_list[0].resolution = \
    lambda ls, sep=py_value(''): py_value(sep.value.join(BuiltIns['str'].call(item).value for item in iter(ls)))
state.deref('join').op_list[1].resolution = \
    lambda sep, *args: py_value(sep.value.join(BuiltIns['str'].call(item).value for item in args))


state.deref('parse_json').op_list[0].resolution = lambda text: py_value(json.loads(text.value))