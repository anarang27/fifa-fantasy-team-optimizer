"""FIFA country-code handling.

The user price list uses FIFA 3-letter codes (ENG, FRA, KSA, ...). FBref uses
full English nation names. `normalize_country` maps either representation to a
single canonical lowercase name so the entity matcher can restrict candidates by
country and the optimizer's per-country cap counts consistently.
"""

from __future__ import annotations

# FIFA 3-letter code -> canonical English nation name (FBref-style).
FIFA_CODE_TO_NAME = {
    "ALG": "Algeria",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CIV": "Ivory Coast",
    "COD": "DR Congo",
    "COL": "Colombia",
    "CPV": "Cape Verde",
    "CRO": "Croatia",
    "CUW": "Curacao",
    "CZE": "Czechia",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "ENG": "England",
    "ESP": "Spain",
    "FRA": "France",
    "GER": "Germany",
    "GHA": "Ghana",
    "HAI": "Haiti",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "JOR": "Jordan",
    "JPN": "Japan",
    "KOR": "South Korea",
    "KSA": "Saudi Arabia",
    "MAR": "Morocco",
    "MEX": "Mexico",
    "NED": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PAN": "Panama",
    "PAR": "Paraguay",
    "POR": "Portugal",
    "QAT": "Qatar",
    "RSA": "South Africa",
    "SCO": "Scotland",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "TUR": "Turkey",
    "URU": "Uruguay",
    "USA": "United States",
    "UZB": "Uzbekistan",
}

# Common FBref / alternative spellings -> canonical name.
_ALIASES = {
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "ir iran": "Iran",
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "congo dr": "DR Congo",
    "united states of america": "United States",
    "usa": "United States",
    "czech republic": "Czechia",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
}

_NAME_LOOKUP = {v.lower(): v for v in FIFA_CODE_TO_NAME.values()}


def normalize_country(value: str) -> str:
    """Return a canonical lowercase nation name from a code or a name."""
    if value is None:
        return ""
    raw = str(value).strip()
    upper = raw.upper()
    if upper in FIFA_CODE_TO_NAME:
        return FIFA_CODE_TO_NAME[upper].lower()
    low = raw.lower()
    if low in _ALIASES:
        return _ALIASES[low].lower()
    if low in _NAME_LOOKUP:
        return low
    return low
