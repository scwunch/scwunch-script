from .base import *

print(f'loading {__name__}.py')

def read_file(arg: Record, lines=BuiltIns['blank']):
    with open(arg.get('path').value, 'r') as f:
        if lines == BuiltIns['true']:
            return py_value(f.readlines())
        else:
            return py_value(f.read())
state.deref('read').op_list[0].resolution = read_file

def regex_extract(regex: RegEx | PyValue[str], text: PyValue[str], *,
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
    return py_value(re.findall(regex.value, text.value, flags))
state.deref('regex').frame['extract'].op_list[0].resolution = regex_extract


def regex_constructor(s, f=py_value('')):
    return RegEx(s.value, f.value)
state.deref('regex').op_list[0].resolution = regex_constructor
state.deref('RegEx').op_list = state.deref('regex').op_list[:]

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