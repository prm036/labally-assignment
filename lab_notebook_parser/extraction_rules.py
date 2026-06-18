"""
Deterministic text-level extraction rules for lab notebook content.

Covers:
  - Scientific text normalisation (units, symbols, Greek letters, sub/superscripts)
  - Quantity extraction (values + units)
  - Chemical formula mention extraction
  - Ratio extraction (v/v, mol%, w/w, etc.)
  - Chemistry NER via ChemDataExtractor (optional)
  - PubChem name resolution
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Quantity:
    raw_text: str
    value: Optional[float]
    unit: str
    quantity_type: str
    source: str
    confidence: float = 0.85
    notes: List[str] = field(default_factory=list)


@dataclass
class FormulaMention:
    raw_text: str
    formula: str
    source: str
    confidence: float = 0.75
    validation: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class Ratio:
    raw_text: str
    components: List[str]
    values: List[float]
    ratio_type: str          # "v/v", "w/w", "mol%", "mole_ratio", etc.
    source: str
    confidence: float = 0.8
    notes: List[str] = field(default_factory=list)


@dataclass
class ChemEntity:
    raw_text: str
    normalized_name: Optional[str]
    entity_type: str          # "solute", "solvent", "electrode", "additive", "product", "other"
    formula: Optional[str] = None
    formula_source: Optional[str] = None  # "explicit_text"|"structure_model"|"pubchem_lookup"|"vlm_inferred"
    pubchem_cid: Optional[int] = None
    concentration: Optional[str] = None
    source: str = "ner"
    confidence: float = 0.75
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Greek / symbol / sub-superscript normalisation
# ---------------------------------------------------------------------------

# Ordered: longest match first to avoid partial replacements
_GREEK_MAP: List[tuple] = [
    # Theta / XRD angle
    (r"\b2\s*[Tt]heta\b",       "2θ"),
    (r"\b2\s*Θ\b",              "2θ"),
    (r"\btheta\b",              "θ"),
    (r"\bTheta\b",              "θ"),
    # Lambda (wavelength)
    (r"\blambda\b",             "λ"),
    (r"\bLambda\b",             "λ"),
    # Delta
    (r"\bDelta\b",              "Δ"),
    (r"\bdelta\b",              "δ"),
    # Sigma
    (r"\bSigma\b",              "Σ"),
    (r"\bsigma\b",              "σ"),
    # Alpha, beta, gamma
    (r"\balpha\b",              "α"),
    (r"\bbeta\b",               "β"),
    (r"\bgamma\b",              "γ"),
    # Micro prefix
    (r"\bmu\b(?=-?[A-Za-z])",   "μ"),
]

_SYMBOL_MAP: List[tuple] = [
    # Arrows
    (r"->",    " → "),
    (r"=>",    " → "),
    (r"<->",   " ⇌ "),
    (r"<=>",   " ⇌ "),
    # Plus/minus
    (r"\+-",   "±"),
    (r"-\+",   "±"),
    # Degree sign misread as zero or letter O
    (r"(\d)\s*oC\b",            r"\1 °C"),
    (r"(\d)\s*0C\b",            r"\1 °C"),
    (r"(\d)\s*deg\s*[Cc]\b",    r"\1 °C"),
    (r"(\d)\s*°[Cc]\b",         r"\1 °C"),
    # Volume units
    (r"\buL\b",  "μL"),
    (r"\bul\b",  "μL"),
    (r"\bµL\b",  "μL"),
    (r"\bml\b",  "mL"),
    # Electrode shorthand typos
    (r"\bAg/AgCI\b",  "Ag/AgCl"),   # capital I → l
    (r"\bAg/Ag[Cc]l\b",  "Ag/AgCl"),
    # Common misreads
    (r"\bH20\b",  "H₂O"),
    (r"\bH2 0\b", "H₂O"),
    (r"\bH2O\b",  "H₂O"),
    (r"\bLii\+",  "Li⁺"),
    (r"\bLi\+\b", "Li⁺"),
    # Current density
    (r"mA/cm\^?2\b",  "mA/cm²"),
    (r"mA/cm2\b",     "mA/cm²"),
    # Scientific notation: 1.5E-4 → 1.5×10⁻⁴  (preserved as-is, just normalise spacing)
    (r"(\d)\s*[Ee]\s*([+-]?\d+)", r"\1E\2"),
]

# Subscript number map for chemical formulas
_SUB_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP_MAP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")


def _apply_unicode_subscripts_in_formulas(text: str) -> str:
    """
    Render trailing digits in known chemical formula patterns as Unicode subscripts.
    E.g. LiTFSI: C2F6LiNO4S2 → C₂F₆LiNO₄S₂
    This is applied only to tokens that look like molecular formulas.
    """
    def subscript_formula(m: re.Match) -> str:
        token = m.group(0)
        result = re.sub(r"([A-Z][a-z]?)(\d+)", lambda x: x.group(1) + x.group(2).translate(_SUB_MAP), token)
        # Handle charge markers: Li+, Li2+, etc.
        result = re.sub(r"([A-Z][a-z]?\d*)([+-]\d*)\b", lambda x: x.group(1) + x.group(2).translate(_SUP_MAP), result)
        return result

    # Match likely chemical formulas: start with capital, contain element+digit pattern
    return re.sub(r"\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+[+-]?\b", subscript_formula, text)


def normalize_greek_symbols(text: str) -> str:
    for pattern, replacement in _GREEK_MAP:
        text = re.sub(pattern, replacement, text)
    return text


def normalize_symbols(text: str) -> str:
    for pattern, replacement in _SYMBOL_MAP:
        text = re.sub(pattern, replacement, text)
    return text


def normalize_scientific_text(text: str) -> str:
    """
    Full normalisation pipeline:
      1. Whitespace collapse
      2. Greek letter words → Unicode symbols
      3. Arrow / degree / unit symbol fixes
      4. Number+unit spacing
      5. Chemical formula subscript rendering
    """
    if text is None:
        return ""

    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()

    # Greek symbols
    s = normalize_greek_symbols(s)

    # Arrows, degree, electrode names, etc.
    s = normalize_symbols(s)

    # Ensure space between number and unit
    s = re.sub(
        r"(\d)("
        r"mA/cm²|mA/cm2|mA/cm\^2|"
        r"mg|g|kg|mL|μL|uL|L|μM|uM|mM|M|ppm|"
        r"mg/mL|mol|mmol|min|sec|s|hr|h|K|%|"
        r"mA|A|mV|V|C|rpm|cm²|cm2|cm\^2|°C|nm|μm|mm"
        r")\b",
        r"\1 \2",
        s,
    )

    # Chemical subscripts for plain formulas (e.g. H2O → H₂O)
    s = _apply_unicode_subscripts_in_formulas(s)

    return s


# ---------------------------------------------------------------------------
# Quantity extraction
# ---------------------------------------------------------------------------

UNIT_TYPES: Dict[str, str] = {
    "mg": "mass", "g": "mass", "kg": "mass",
    "mL": "volume", "L": "volume", "μL": "volume", "uL": "volume",
    "M": "concentration", "mM": "concentration", "μM": "concentration", "ppm": "concentration",
    "mol": "amount_of_substance", "mmol": "amount_of_substance",
    "°C": "temperature", "K": "temperature",
    "s": "time", "sec": "time", "min": "time", "h": "time", "hr": "time",
    "%": "percentage",
    "mA": "current", "A": "current",
    "mA/cm²": "current_density", "mA/cm2": "current_density", "mA/cm^2": "current_density",
    "V": "potential", "mV": "potential",
    "C": "charge",
    "rpm": "rotation_speed",
    "nm": "length", "μm": "length", "mm": "length", "cm": "length",
    "mol%": "mole_fraction",
    "w/w": "weight_fraction",
    "v/v": "volume_fraction",
}

QUANTITY_PATTERN = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:\s*[Ee]\s*[-+]?\d+)?)
    \s*
    (?P<unit>
        mA\s*/\s*cm(?:²|2|\^2)|
        mol%|v/v|w/w|
        °C|μL|uL|mL|L|μM|uM|mM|M|ppm|
        mg/mL|mg|g|kg|mmol|mol|min|sec|s|hr|h|K|%|
        mA|A|mV|V|C|rpm|nm|μm|mm|cm
    )
    (?![A-Za-z])
    """,
    re.VERBOSE,
)


def parse_numeric_value(raw: str) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).replace("−", "-").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None


def normalize_unit(unit: str) -> str:
    unit = str(unit).strip().replace(" ", "")
    _MAP = {
        "uL": "μL", "ul": "μL",
        "ml": "mL",
        "mA/cm2": "mA/cm²", "mA/cm^2": "mA/cm²",
    }
    return _MAP.get(unit, unit)


def extract_quantities(text: str, source: str = "merged_transcription") -> List[Quantity]:
    normalized = normalize_scientific_text(text)
    quantities: List[Quantity] = []

    for match in QUANTITY_PATTERN.finditer(normalized):
        left_context = normalized[max(0, match.start() - 30):match.start()].lower()
        # Skip page number references
        if "page" in left_context or "pg" in left_context:
            continue

        raw_value = match.group("value")
        unit = normalize_unit(match.group("unit"))
        raw_text = match.group(0)

        quantities.append(Quantity(
            raw_text=raw_text,
            value=parse_numeric_value(raw_value),
            unit=unit,
            quantity_type=UNIT_TYPES.get(unit, "unknown"),
            source=source,
        ))

    return quantities


# ---------------------------------------------------------------------------
# Ratio extraction  (e.g. "4:1 v/v", "5 mol%")
# ---------------------------------------------------------------------------

RATIO_PATTERN = re.compile(
    r"""
    (?P<a>\d+(?:\.\d+)?)\s*:\s*(?P<b>\d+(?:\.\d+)?)     # a:b ratio
    \s*(?P<type>v/v|w/w|w/v|mol/mol)?                   # optional type label
    |
    (?P<pct>\d+(?:\.\d+)?)\s*(?P<pct_type>mol%|wt%|vol%) # percentage form
    """,
    re.VERBOSE | re.IGNORECASE,
)


def extract_ratios(text: str, source: str = "merged_transcription") -> List[Ratio]:
    ratios: List[Ratio] = []
    normalized = normalize_scientific_text(text)

    for m in RATIO_PATTERN.finditer(normalized):
        if m.group("a") is not None:
            a, b = float(m.group("a")), float(m.group("b"))
            rtype = m.group("type") or "ratio"
            ratios.append(Ratio(
                raw_text=m.group(0).strip(),
                components=[],
                values=[a, b],
                ratio_type=rtype,
                source=source,
            ))
        elif m.group("pct") is not None:
            pct = float(m.group("pct"))
            pct_type = m.group("pct_type")
            ratios.append(Ratio(
                raw_text=m.group(0).strip(),
                components=[],
                values=[pct],
                ratio_type=pct_type,
                source=source,
            ))

    return ratios


# ---------------------------------------------------------------------------
# Chemical formula extraction
# ---------------------------------------------------------------------------

# Full periodic table (118 elements — avoids the old 14-element whitelist bug)
_ELEMENT_SYMBOLS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S",
    "Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
    "Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
    "Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm",
    "Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os",
    "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa",
    "U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg",
    "Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
}

FORMULA_PATTERN = re.compile(r"(?<![a-zA-Z])(?:[A-Z][a-z]?\d*){2,}(?:[+-])?\b")


def validate_formula_string(formula: str) -> Dict[str, Any]:
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not tokens:
        return {"valid": False, "reason": "no tokens"}
    invalid = [el for el, _ in tokens if el not in _ELEMENT_SYMBOLS]
    if invalid:
        return {"valid": False, "invalid_elements": invalid}
    return {"valid": True, "tokens": tokens}


def extract_explicit_formula_mentions(text: str,
                                      source: str = "merged_transcription") -> List[FormulaMention]:
    normalized = normalize_scientific_text(text)
    mentions: List[FormulaMention] = []
    seen: set = set()

    for match in FORMULA_PATTERN.finditer(normalized):
        formula = match.group(0)
        # Skip very short tokens without digits or charge markers
        if len(formula) <= 2 and not re.search(r"\d|\+|-", formula):
            continue
        if formula in seen:
            continue
        seen.add(formula)

        validation = validate_formula_string(formula)
        if validation.get("valid"):
            mentions.append(FormulaMention(
                raw_text=formula,
                formula=formula,
                source=source,
                confidence=0.80,
                validation=validation,
                notes=["Formula appears explicitly in the transcription."],
            ))

    return mentions


# ---------------------------------------------------------------------------
# Chemistry NER via ChemDataExtractor (optional)
# ---------------------------------------------------------------------------

def run_chemistry_ner(text: str) -> List[ChemEntity]:
    """
    Extract chemical entity mentions using ChemDataExtractor (CDE).
    Falls back gracefully if CDE is not installed.
    """
    try:
        from chemdataextractor import Document
        from chemdataextractor.doc import Paragraph
    except ImportError:
        return []

    try:
        doc = Document(Paragraph(text))
        entities: List[ChemEntity] = []
        seen: set = set()

        for record in doc.records.serialize():
            names = record.get("names") or []
            labels = record.get("labels") or []
            all_names = list(set(names + labels))
            for name in all_names:
                if not name or name in seen:
                    continue
                seen.add(name)
                entities.append(ChemEntity(
                    raw_text=name,
                    normalized_name=name,
                    entity_type="other",
                    source="chemdataextractor",
                ))

        return entities
    except Exception as e:
        return []


# ---------------------------------------------------------------------------
# PubChem name resolution
# ---------------------------------------------------------------------------

def resolve_chemical_with_pubchem(name: str, timeout: int = 8) -> Dict[str, Any]:
    try:
        import requests
        import urllib.parse
    except ImportError:
        return {"resolved": False, "reason": "requests unavailable"}

    if not name or not str(name).strip():
        return {"resolved": False, "reason": "empty name"}

    encoded = urllib.parse.quote(str(name).strip())
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}"
        f"/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON"
    )

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200:
            return {"resolved": False, "reason": f"HTTP {resp.status_code}"}

        props = resp.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return {"resolved": False, "reason": "no properties"}

        first = props[0]
        return {
            "resolved": True,
            "cid": first.get("CID"),
            "molecular_formula": first.get("MolecularFormula"),
            "iupac_name": first.get("IUPACName"),
            "canonical_smiles": first.get("CanonicalSMILES"),
            "source": "PubChem PUG REST",
        }
    except Exception as e:
        return {"resolved": False, "reason": str(e)}
