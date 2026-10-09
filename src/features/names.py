"""Player-name normalisation, for matching ESPN's names to nba_api's."""
from __future__ import annotations

import re
import unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """'Nikola Jokić' / 'Jaren Jackson Jr.' -> 'nikola jokic' / 'jaren jackson' (accents, punctuation, suffixes dropped)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'`’\-]", "", s.lower())
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
