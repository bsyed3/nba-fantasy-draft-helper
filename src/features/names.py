"""Player-name normalisation, for matching ESPN's names to nba_api's."""
from __future__ import annotations

import re
import unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def search_key(name: str) -> str:
    """
    The plain-letter spelling someone types when looking a player up: accents folded, apostrophes / periods / commas dropped, hyphens as
    spaces, suffixes kept. "Day'Ron Sharpe" -> "dayron sharpe", "Nikola Jokić" -> "nikola jokic", "P.J. Washington" -> "pj washington".
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.,'`’]", "", s).replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def normalize_name(name: str) -> str:
    """'Nikola Jokić' / 'Jaren Jackson Jr.' -> 'nikola jokic' / 'jaren jackson' (accents, punctuation, suffixes dropped)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'`’\-]", "", s.lower())
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()
