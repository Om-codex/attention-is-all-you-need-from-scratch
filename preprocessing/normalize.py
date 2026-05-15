import unicodedata
import string

def lowercase(text: str) -> str:
    return text.lower()

def unicode_normalize(text: str) -> str:
    return unicodedata.normalize('NFC',text)

exclude = string.punctuation
def remove_puncs(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.translate(str.maketrans("","",exclude))

def remove_spaces(text: str) -> str:
    return text.strip()

def normalize_text(text: str) -> str:
    text = unicode_normalize(text)
    text = lowercase(text)
    text = remove_puncs(text)
    text = remove_spaces(text)
    return text