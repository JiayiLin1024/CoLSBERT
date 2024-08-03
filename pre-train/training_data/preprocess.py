from tree_sitter import Language, Parser

import re
from io import StringIO
import  tokenize

def remove_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remain comments:
            if token_type == tokenize.COMMENT:
                out += token_string
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
                       
def preprocess(js,tokenizer,lang,args):
    """obtain different types of pre-training dataset"""
    if args.input_type == "join_tokens":
        js["function_tokens_tokenize"] = tokenizer.tokenize(" ".join(js["code_tokens"]))
        js["docstring_split_tokenize"] = tokenizer.tokenize(" ".join(js['docstring_tokens']))
    elif args.input_type == "source_code":
        js["function_tokens_tokenize"] = tokenizer.tokenize(remove_docstrings(js['code'], lang))  
        js["docstring_split_tokenize"] = tokenizer.tokenize(" ".join(js['docstring_tokens']))
    
    return js
