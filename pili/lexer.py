# from stub import *
import re
from itertools import islice
from .syntax import Token, TokenType
from .utils import SyntaxErr
from .state import Op

print(f'loading {__name__}.py')


class Tokenizer:
    idx: int
    indent: int
    ln: int
    col: int
    char: str | None
    in_string: list[str]
    fn_lv: int
    tokens: list[Token]
    def __init__(self, script: str):
        self.script = script
        self.idx = -1
        self.ln, self.col = 0, 0
        self.char = '\n'
        self.next_line()
        self.in_string = []
        self.fn_lv = 0
        self.tokens = self.read_tokens()

    def __str__(self):
        head = f"Tokenizer State:\n\t" \
               f"head:({self.ln}, {self.col})\n\tcurrent_line:{self.ln}\n\tchar:{self.char}\n"
        lines = []
        start = 0
        indent = ''
        for i, tok in enumerate(self.tokens):
            if tok.type in (TokenType.NewLine, TokenType.BlockStart, TokenType.BlockEnd):
                ln = self.tokens[start].pos.ln
                lines.append(f"{ln}. {indent}{' '.join(t.text for t in self.tokens[start:i])}")
                start = i+1

            if tok.type in (TokenType.BlockStart, TokenType.BlockEnd):
                indent = tok.text
        return head + "\n".join(lines)

    def read_tokens(self):
        tokens: list[Token] = []
        text: str
        token_type = None
        while self.char:
            while self.char and self.char in ' \t':
                self.next_char()
            start = self.idx
            pos = (self.ln, self.col)
            if re.match(r'\d', self.char):
                text = self.read_number()
                token_type = TokenType.Number
            elif self.char == "`":
                text = self.read_string_literal()
                token_type = TokenType.StringLiteral
            elif self.char in ('"', "'"):
                text = self.char
                if len(self.in_string) > self.fn_lv:
                    token_type = TokenType.StringEnd
                    self.in_string.pop()
                else:
                    token_type = TokenType.StringStart
                    self.in_string.append(text)
            elif re.match(r'\w', self.char):
                text = self.read_word()
                if text == 'not' and tokens and tokens[-1].text == 'is':
                    tokens[-1].text = 'is not'
                    tokens[-1].pos.stop_index = self.idx
                    continue
                if text == 'in' and tokens and tokens[-1].text == 'not':
                    tokens[-1].text = 'not in'
                    tokens[-1].pos.stop_index = self.idx
                    continue
            elif re.match(r'[:.<>?/~!@$%^&*+=|-]', self.char):
                # Note: , (comma) and ; (semicolon) are excluded because of special treatment
                text = self.read_operator()
                if text == ':':
                    if len(tokens) > 2 and tokens[-1].type == TokenType.Operator and tokens[-2].text == '.':
                        # This allows operators to be used as function names like this: `.>:`
                        tokens[-1].type = TokenType.Name
                    if (''.join(islice(
                            (t.text for t in reversed(tokens) if not re.match(r'\s', t.text)),
                            3)) == "]*,"):
                        # if the last three non-whitespace tokens are `,*]` then
                        # special case syntax to designate arbitrary kwargs
                        tokens[-2].type = TokenType.Name
                token_type = TokenType.Operator
            elif self.char in "{}":
                text = self.char
                if self.in_string:
                    if self.char == '{' and len(self.in_string) - self.fn_lv == 1:
                        self.fn_lv += 1
                    elif self.char == '}':
                        self.fn_lv -= 1
            elif self.char in '[](),;':
                text = self.char
                if text == '[' and tokens and tokens[-1].text == '?':
                    tokens[-1].text = 'call?'
            elif self.char == '#' and self.peek(1, 5) == 'debug':
                self.next_char(6)
                continue
            elif self.char in "#\n":
                last_indent = self.indent
                self.next_line()
                if self.indent == last_indent:
                    text = '\n'
                else:
                    text = '\t' * self.indent
                    if self.indent > last_indent:
                        token_type = TokenType.BlockStart
                    else:
                        token_type = TokenType.BlockEnd
            elif self.char == '\\':
                while self.next_char() and self.char in ' \t':
                    pass
                if self.char != '\n':
                    raise SyntaxErr(f"Expected newline after backslash at {pos}")
                self.next_line()
                continue
            else:
                raise SyntaxErr(f'{pos} What kind of character is this? "{self.char}"')

            if self.idx == start:
                self.next_char()
            end = self.idx
            tokens.append(Token(text, token_type, pos, start, end))
            token_type = None
            if len(self.in_string) > self.fn_lv:
                pos = self.ln, self.col
                start = self.idx  # ie, the index after the last token, which was a quote or brace
                text = self.read_string(self.in_string[-1])
                tokens.append(Token(text, TokenType.StringPart, pos, start, self.idx))

        tokens.append(Token('', TokenType.BlockEnd, (self.ln, self.col)))
        tokens.append(Token('', TokenType.EOF, (self.ln, self.col)))
        return tokens

    def next_char(self, count=1):
        """ if count>1, assumes there is no newline character in that range """
        if self.char == '\n':
            self.ln += 1
            self.col = 0
        self.idx += count
        self.col += count
        self.char = self.script[self.idx:self.idx+1]
        return self.char

    def next_line(self):
        while self.char and self.char != '\n':
            self.next_char()
        # self.ln += 1
        self.indent = 0
        while self.next_char():
            if self.char == '\t':
                self.indent += 1
            elif self.char == '#' and self.peek(1, 5) == 'debug':
                break
            elif self.char in '#\n':
                return self.next_line()
            elif self.peek(0, 4) == '    ':
                self.next_char(3)
                self.indent += 1
            elif self.char == ' ':
                continue
            else:
                break
        # self.col = 1

    def read_number(self):
        start = self.idx
        has_radix = False
        while self.next_char():
            if re.match(r'\d|_', self.char):
                continue
            elif not has_radix and re.match(r'\.\d', self.peek(0, 2)):
                # this pattern disallows number literals from ending in a dot.
                # if I ever go back to interpreting dotted numbers as floats, then I need to re-allow this
                has_radix = True
                self.next_char()
            else:
                break
        if self.char in 'btqphsond':
            self.next_char()
        if self.char == 'f':
            self.next_char()
        return self.script[start:self.idx]

    def read_string(self, quote: str) -> str:
        # str_text = ""
        str_start = self.idx
        while self.char:
            if self.char == '\n':
                break
            if self.char in ("{", quote):
                return self.script[str_start:self.idx]
            if self.char == "\\":
                self.next_char()
            self.next_char()
        raise SyntaxErr(f"Unterminated string at {{ln={self.ln}, ch={self.col}}}: "
                            f"{self.script[str_start:self.idx]} ")
                            # f"{str_text}")

    def read_string_literal(self):
        backticks = 1
        while self.next_char() == "`":
            backticks += 1
        str_start = self.idx
        # str_text = "`" * backticks
        while self.char:
            if self.peek(0, backticks) == "`" * backticks:
                s = slice(str_start, self.idx)
                self.next_char(backticks)
                return self.script[s]
            # if str_text.endswith("`" * backticks):
            #     self.next_char()
            #     return str_text
            self.next_char()
        raise SyntaxErr(f"Unterminated string at {{ln={self.ln + 1}, ch={self.col + 1}}}: "
                            f"{self.script[str_start:self.idx]} ")
                            # f"{str_text}")

    def read_word(self):
        word = self.char
        while self.next_char() and re.match(r'\w', self.char):
            word += self.char
        return word

    def read_operator(self):
        # try to get the longest operator (up to three characters)
        chars = self.peek(0, 3)
        if chars in Op:
            self.next_char(3)
            return chars
        if chars[:2] in Op and chars[1:] != "==":
            # second condition is to disambiguate these two cases: +== and *==
            self.next_char(2)
            # op_text = chars[:2]
            # if op_text == 'is' and self.peek(1, 3)
            return chars[:2]
        self.next_char()
        return chars[0]

    def peek(self, offset=1, count=1):
        i = self.idx + offset
        return self.script[i:i + count]
