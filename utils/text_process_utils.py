import re
import nltk
import inflection
from nltk.stem import PorterStemmer
import string
import pyparsing

# English stop words
stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
              'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
              'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
              'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
              'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'b', 'c', 'e', 'f', 'g', 'h', 'j', 'k', 'l',
              'n', 'p', 'q', 'u', 'v', 'w', 'x', 'z', 'us', 'always', 'already', 'would', 'however', 'perhaps', 'done',
              'cannot', 'can', 'sure', 'without', 'hi', 'could', 'doesn', 'must', 'able', 'much', 'everyone', 'anyone',
              'whatever', 'anyhow', 'yet', 'hence'}

# Java language keywords
java_keywords = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'false', 'final', 'finally', 'float',
                 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new',
                 'null', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super',
                 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'void', 'volatile',
                 'while'}

def remove_comments(file_type='Python'):
    if file_type=='Python':
        commentFilter = pyparsing.pythonStyleComment.suppress()
    else:
        commentFilter = pyparsing.javaStyleComment.suppress()
    return commentFilter.transformString

def remove_prefix(file_type='Python'):
    if file_type=='Python':
        def remove_prefix_python(content):
            lines = content.splitlines()
            in_comment_block = False
            start = 0
            for no,line in enumerate(lines):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                    if not in_comment_block:
                        in_comment_block = True
                        if stripped_line[3:].endswith('"""') or stripped_line[3:].endswith("'''"):
                            in_comment_block = False
                        continue
                    else:
                        in_comment_block = False
                        continue
                if in_comment_block:
                    if stripped_line.endswith('"""') or stripped_line.endswith("'''"):
                        in_comment_block = False
                    continue
                if stripped_line.startswith('#'):
                    continue
                if stripped_line:
                    start = no
                    break
            return '\n'.join(lines[start:])
        return remove_prefix_python
    else:
        def remove_import_java(content):
            lines = content.splitlines()
            package_no,import_no = -1,-1
            for no,line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith('package'):
                    package_no = no
                if stripped_line.startswith('import'):
                    import_no = no
            max_no = max(package_no,import_no)
            return '\n'.join(lines[(max_no+1):])
            
        def remove_prefix_java(content):
            lines = content.splitlines()
            in_comment_block = False
            start = 0
            for no,line in enumerate(lines):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith('/*'):
                    if not in_comment_block:
                        in_comment_block = True
                        if stripped_line[2:].endswith('*/'):
                            in_comment_block = False
                        continue
                    else:
                        in_comment_block = False
                        continue
                if in_comment_block:
                    if stripped_line.endswith('*/'):
                        in_comment_block = False
                    continue
                if stripped_line.startswith('//'):
                    continue
                if stripped_line:
                    start = no
                    break
            return '\n'.join(lines[start:])
        return remove_import_java


# --------------------------------------------------------------------------------------------------------
# Extracting stack traces from bug reports
def extract_stack_traces(content):
    
    # Simple pattern to retrieve stack traces
    pattern = re.compile(r' at (.*?)\((.*?)\)')
    # Signs of a true stack trace to check in the retrieved regex grouping
    signs = ['.java', 'Unknown Source', 'Native Method']
    st_candid = re.findall(pattern, content)
    # Filter actual stack traces from retrieved candidates
    st = [x for x in st_candid if any(s in x[1] for s in signs)]
    return st

# Extracing specific pos tags from bug reports' summary and description
def pos_tagging(content):
    # Tokenizing using word_tokeize for more accurate pos-tagging
    summ_tok = nltk.word_tokenize(content)
    sum_pos = nltk.pos_tag(summ_tok)
    pos_tagged_summary = [token for token, pos in sum_pos if 'NN' in pos or 'VB' in pos]
    return pos_tagged_summary
    
def tokenize(content):
    return nltk.wordpunct_tokenize(content)

def split_camelcase(tokens):

    # Copy tokens
    returning_tokens = tokens[:]

    for token in tokens:
        split_tokens = re.split(fr'[{string.punctuation}]+', token)
        # If token is split into some other tokens
        if len(split_tokens) > 1:
            returning_tokens.remove(token)
            # Camel case detection for new tokens
            for st in split_tokens:
                camel_split = inflection.underscore(st).split('_')
                if len(camel_split) > 1:
                    returning_tokens.append(st)
                    returning_tokens += camel_split
                else:
                    returning_tokens.append(st)
        else:
            camel_split = inflection.underscore(token).split('_')
            if len(camel_split) > 1:
                returning_tokens += camel_split

    return returning_tokens

# Removing punctuation, numbers and also lowercase conversion
def normalize(tokens):
    # Building a translate table for punctuation and number removal
    punctnum_table = str.maketrans(
        {c: None for c in string.punctuation + string.digits})
    token_punctnum_rem = [token.translate(punctnum_table)
                            for token in tokens]
    tokens = [token.lower() for token
                      in token_punctnum_rem if token]
    return tokens

stop_words = set(stop_words).union(set(java_keywords))
# Removing stop words from tokens
def remove_stopwords(tokens,stop_words):
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def stem(tokens):
    # Stemmer instance
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# --------------------------------------------------------------------------------------------------------
def process_pos_text(content):
    content = pos_tagging(content)
    content = split_camelcase(content)
    content = normalize(content)
    content = remove_stopwords(content,stop_words)
    content = stem(content)
    return ' '.join(content)
    
def process_text(content):
    content = tokenize(content)
    content = split_camelcase(content)
    content = normalize(content)
    content = remove_stopwords(content,stop_words)
    content = stem(content)
    return ' '.join(content)

def process_name(tokens):
    content = split_camelcase(tokens)
    content = normalize(content)
    content = remove_stopwords(content,stop_words)
    content = stem(content)
    return ' '.join(content)