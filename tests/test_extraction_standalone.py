"""
tests/test_extraction_standalone.py

Tests for extraction_rules.py that run without requiring cv2, torch,
or any model weights.  Safe to run locally or in CI.

Usage:
    python tests/test_extraction_standalone.py
"""
import re
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline the key functions so this test has zero heavy dependencies
# ---------------------------------------------------------------------------

# We replicate just enough of extraction_rules to verify correctness.
# (The real module can't be imported without cv2.)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch cv2 before anything else
import types
for mod_name in ['cv2', 'numpy', 'PIL', 'PIL.Image', 'matplotlib',
                 'matplotlib.pyplot', 'qwen_vl_utils', 'torch',
                 'transformers']:
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

# numpy needs a real-ish interface for the preprocess/layout modules
import numpy as _np_real
sys.modules['numpy'] = _np_real

# Minimal cv2 stub
cv2_stub = types.ModuleType('cv2')
cv2_stub.cvtColor = lambda *a, **k: None
cv2_stub.COLOR_BGR2GRAY = 0
sys.modules['cv2'] = cv2_stub

# PIL.Image stub
pil_stub = types.ModuleType('PIL')
pil_image_stub = types.ModuleType('PIL.Image')
pil_image_stub.open = lambda *a: None
pil_stub.Image = pil_image_stub
sys.modules['PIL'] = pil_stub
sys.modules['PIL.Image'] = pil_image_stub

# matplotlib stub
mpl_stub = types.ModuleType('matplotlib')
mpl_plt_stub = types.ModuleType('matplotlib.pyplot')
mpl_stub.pyplot = mpl_plt_stub
sys.modules['matplotlib'] = mpl_stub
sys.modules['matplotlib.pyplot'] = mpl_plt_stub

# Now it's safe to import just extraction_rules
import importlib.util, os
spec = importlib.util.spec_from_file_location(
    "lab_notebook_parser.extraction_rules",
    Path(__file__).parent.parent / "lab_notebook_parser" / "extraction_rules.py",
)
er = importlib.util.module_from_spec(spec)
sys.modules["lab_notebook_parser.extraction_rules"] = er
spec.loader.exec_module(er)

normalize_scientific_text = er.normalize_scientific_text
extract_quantities = er.extract_quantities
extract_ratios = er.extract_ratios
extract_explicit_formula_mentions = er.extract_explicit_formula_mentions
validate_formula_string = er.validate_formula_string


# ---------------------------------------------------------------------------
# Simulated transcription of Page 57 content
# ---------------------------------------------------------------------------

PAGE57_TEXT = """Page 57 Project: Li electrodeposition - glyme electrolytes
June 4 Goal: Screen electrolyte 240604-B for stable Li plating at 30C
Electrolyte: 1M LiTFSI in diglyme:EtOH (4:1 v/v) 20 mL
Add 5 mol% 12-crown-4 as additive. Stir 20min
-> T@ 22.4C, glovebox H2O < 1 ppm
Working elec.: glassy C RDE (0.3 cm2). CE: Li foil; ref: Ag/AgCl
Deposition run 240609-B1
Apply -0.45V vs Ag/AgCl, 90 min, w=1600rpm
J = 0.50 mA/cm2 and 0.3 cm2
Q = 1.5E-4 A . 5400s = 0.81 A.s = 0.81 C
n = Q / 2F = 0.81C / 96485 C/mol
1e- = 1 Li+
Electrode Temperature test: hot plate at 30C
0 min 22.4C  20 min 30.1C  Film looks grey + dull
1 min 23.1C  40 min 31.5C  XRD min peak at 2theta = 2.1 (low intens.)
5 min 25.6C  1 hr 32.0C    Shoulder at 2theta = 4.7
10 min 27.9C 1 hr 30 min 32.6C"""


class TestNormalization(unittest.TestCase):

    def setUp(self):
        self.norm = normalize_scientific_text(PAGE57_TEXT)

    def test_degree_celsius(self):
        self.assertIn('°C', self.norm, "Degree Celsius not normalized")

    def test_xrd_theta(self):
        self.assertIn('2θ', self.norm, "2theta not converted to 2θ")

    def test_arrow(self):
        self.assertIn('→', self.norm, "-> not converted to →")

    def test_current_density(self):
        self.assertIn('mA/cm²', self.norm, "mA/cm2 not normalized to mA/cm²")

    def test_water_formula(self):
        self.assertIn('H₂O', self.norm, "H2O not normalized to H₂O")

    def test_ag_agcl(self):
        self.assertIn('Ag/AgCl', self.norm, "Ag/AgCl not preserved")

    def test_litfsi(self):
        self.assertIn('LiTFSI', self.norm, "LiTFSI not preserved")

    def test_no_page_number_ambiguity(self):
        # "Page 57" should remain, not be mangled
        self.assertIn('57', self.norm)

    def test_crown_ether(self):
        self.assertIn('12-crown-4', self.norm, "12-crown-4 not preserved")


class TestQuantityExtraction(unittest.TestCase):

    def setUp(self):
        norm = normalize_scientific_text(PAGE57_TEXT)
        self.quantities = extract_quantities(norm)
        self.unit_map = {}
        for q in self.quantities:
            self.unit_map.setdefault(q.unit, []).append(q.value)

    def test_extracts_molarity(self):
        self.assertIn('M', self.unit_map, "Molarity (M) not extracted")
        self.assertIn(1.0, self.unit_map['M'], "1M not found")

    def test_extracts_volume(self):
        self.assertIn('mL', self.unit_map, "Volume (mL) not extracted")
        self.assertIn(20.0, self.unit_map['mL'], "20 mL not found")

    def test_extracts_current_density(self):
        self.assertIn('mA/cm²', self.unit_map, "Current density not extracted")
        self.assertAlmostEqual(min(abs(v - 0.5) for v in self.unit_map['mA/cm²']), 0.0, places=2)

    def test_extracts_rpm(self):
        self.assertIn('rpm', self.unit_map, "RPM not extracted")
        self.assertIn(1600.0, self.unit_map['rpm'], "1600 rpm not found")

    def test_extracts_time(self):
        self.assertIn('min', self.unit_map, "Time (min) not extracted")
        self.assertIn(90.0, self.unit_map['min'], "90 min not found")

    def test_extracts_charge(self):
        self.assertIn('C', self.unit_map, "Charge (C) not extracted")
        charge_vals = self.unit_map.get('C', [])
        self.assertTrue(any(abs(v - 0.81) < 0.05 for v in charge_vals), "0.81 C not found")

    def test_page_number_not_extracted(self):
        # "Page 57" should not yield a spurious quantity with value=57
        for q in self.quantities:
            if q.value == 57.0:
                self.fail(f"Page number 57 was spuriously extracted as quantity: {q}")

    def test_area_cm2(self):
        self.assertIn('cm²', self.unit_map, "Area (cm²) not extracted")
        self.assertAlmostEqual(min(abs(v - 0.3) for v in self.unit_map['cm²']), 0.0, places=2)


class TestRatioExtraction(unittest.TestCase):

    def setUp(self):
        norm = normalize_scientific_text(PAGE57_TEXT)
        self.ratios = extract_ratios(norm)

    def test_4_to_1_ratio(self):
        found = any(
            set(r.values) == {4.0, 1.0}
            for r in self.ratios
        )
        self.assertTrue(found, "4:1 diglyme:EtOH ratio not extracted")

    def test_5_mol_percent(self):
        found = any(
            abs(r.values[0] - 5.0) < 0.01 and 'mol%' in r.ratio_type.lower()
            for r in self.ratios
        )
        self.assertTrue(found, "5 mol% not extracted")


class TestFormulaExtraction(unittest.TestCase):

    def setUp(self):
        norm = normalize_scientific_text(PAGE57_TEXT)
        self.formulas = [f.formula for f in extract_explicit_formula_mentions(norm)]

    def test_water_formula(self):
        # H₂O uses unicode subscript after normalization
        found = any('H' in f and 'O' in f for f in self.formulas)
        self.assertTrue(found, "Water formula (H₂O / H2O) not found in formula mentions")


class TestFormulaValidator(unittest.TestCase):

    def test_valid_formula(self):
        result = validate_formula_string("LiC6")
        self.assertTrue(result['valid'])

    def test_invalid_formula_fake_element(self):
        result = validate_formula_string("XyZ3")
        self.assertFalse(result['valid'])

    def test_all_118_elements(self):
        # Fluorine was missing in V3 — ensure it's now included
        result = validate_formula_string("CF3")
        self.assertTrue(result['valid'], "Fluorine not in element set (V3 regression)")

    def test_lithium(self):
        result = validate_formula_string("Li2O")
        self.assertTrue(result['valid'])


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity=2)
