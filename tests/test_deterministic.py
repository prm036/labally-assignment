from lab_notebook_parser.extraction_rules import normalize_scientific_text, extract_quantities
from lab_notebook_parser.transcription import is_bad_placeholder_line


def test_normalization():
    text = "Apply -0.4SV vs Ag/AgCI, 90min, H20 < 1ppm, 30 C, 0.SmA"
    norm = normalize_scientific_text(text)

    assert "Ag/AgCl" in norm
    assert "90 min" in norm
    assert "H2O" in norm
    assert "1 ppm" in norm
    assert "30 °C" in norm
    assert "0.5 mA" in norm


def test_quantity_extraction():
    text = "Page 57 Project. Apply -0.45 V vs Ag/AgCl, 90 min, w=1600 rpm, J=0.50 mA/cm², Q=0.81 C"
    qs = extract_quantities(text)

    units = {q.unit for q in qs}
    raws = {q.raw_text for q in qs}

    assert "V" in units
    assert "min" in units
    assert "rpm" in units
    assert "mA/cm²" in units
    assert "C" in units
    assert not any("57" in r for r in raws)


def test_placeholder_filter():
    assert is_bad_placeholder_line("transcribed line")
    assert is_bad_placeholder_line("same line with obvious symbols normalized")
    assert not is_bad_placeholder_line("Apply -0.45 V vs Ag/AgCl")
