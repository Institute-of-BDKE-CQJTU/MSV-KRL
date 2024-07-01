import re
from nltk import word_tokenize


def URI_parse(uri):
    """Parse a URI: remove the prefix, parse the name part (Camel cases are plit)"""
    if '#' not in uri:
        ind = uri[::-1].index('/')
        name = uri[-ind:]
    else:
        name = re.sub("http[a-zA-Z0-9:/._-]+#", "", uri)

    name = name.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
        replace('"', ' ').replace("'", ' ')
    words = []
    for item in name.split():
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
        for m in matches:
            word = m.group(0)
            words.append(word.lower())
#            if word.isalpha():
#                words.append(word.lower())
    return words


def pre_process_words(words):
    text = ' '.join([re.sub(r'https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in words])
    tokens = word_tokenize(text)
    # processed_tokens = [token.lower() for token in tokens if token.isalpha()]
    processed_tokens = [token.lower() for token in tokens]
    return processed_tokens


# Some entities have English labels
# Keep the name of built-in properties (those starting with http://www.w3.org)
# Some entities have no labels, then use the words in their URI name
def label_item(item, uri_labels_dict):
    if item in uri_labels_dict:
        return uri_labels_dict[item]
    elif item.startswith('http://www.w3.org'):
        return [item.split('#')[1].lower()]
    elif item.startswith('http://') or item.startswith('https://'):
        return URI_parse(uri=item)
    else:
        return [item.lower()]