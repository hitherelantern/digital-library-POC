import re

def clean_text(text: str) -> str:
    """
    Cleans text by encoding, decoding, and removing unwanted characters.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    text = text.encode('utf-8', 'replace').decode('utf-8')
    text = ''.join(c for c in text if ord(c) <= 0xFFFF)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    return text