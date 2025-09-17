import re

def clean_text(text: str) -> str:
    """
    Keep letters, digits, @ . + - _ and whitespace.
    Replace '/' with space. Collapse whitespace.
    This preserves emails, versions (3.10), C++, cloud-native, snake_case.
    Removes '#' characters.
    """
    if not text:
        return ""

    # Turn slashes into spaces (e.g., Backend/Frontend)
    text = text.replace("/", " ")

    # Whitelist important symbols; map others to space
    # \w == [A-Za-z0-9_]
    text = re.sub(r"[^A-Za-z0-9@.\+\-_\s]", " ", text)

    # Drop sentence-ending dots but keep dots inside tokens (emails, versions, domains)
    text = re.sub(r"(?<![A-Za-z0-9])\.(?![A-Za-z0-9])", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
