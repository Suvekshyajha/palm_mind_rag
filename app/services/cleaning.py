import re
import unicodedata

import ftfy


def clean_text(text: str) -> str:
    """Fix broken characters and tidy up spacing."""
    text = ftfy.fix_text(text)  # repairs garbled characters like â€™
    text = unicodedata.normalize("NFKC", text)  # e.g. turns "ﬁ" into "fi"
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control characters
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join words split across lines
    text = re.sub(r"[ \t]+", " ", text)  # many spaces -> one
    text = re.sub(r" ?\n ?", "\n", text)  # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()