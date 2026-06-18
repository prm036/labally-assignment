"""
Chemical structure recognition for hand-drawn molecular drawings.

Supports two backends:
  - molscribe: transformer model trained specifically for hand-drawn chemical structures
  - decimer:   CNN-based, handles sketch-style drawings well

Validates output SMILES with RDKit and optionally resolves to canonical
name/formula via PubChem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from PIL import Image


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class StructureResult:
    region_id: str
    backend: str
    smiles: Optional[str] = None
    confidence: float = 0.0
    is_valid_smiles: bool = False
    canonical_smiles: Optional[str] = None
    molecular_formula: Optional[str] = None
    iupac_name: Optional[str] = None
    pubchem_cid: Optional[int] = None
    nearby_labels: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    raw_output: Optional[str] = None


# ---------------------------------------------------------------------------
# RDKit validation
# ---------------------------------------------------------------------------

def validate_and_canonicalize_smiles(smiles: str) -> Dict[str, Any]:
    """
    Validate SMILES using RDKit. Returns:
      {"valid": bool, "canonical_smiles": str, "molecular_formula": str}
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        return {"valid": False, "reason": "rdkit not installed"}

    if not smiles or not smiles.strip():
        return {"valid": False, "reason": "empty smiles"}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "reason": "rdkit rejected SMILES"}

    canonical = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return {
        "valid": True,
        "canonical_smiles": canonical,
        "molecular_formula": formula,
    }


# ---------------------------------------------------------------------------
# PubChem lookup by SMILES
# ---------------------------------------------------------------------------

def pubchem_lookup_by_smiles(smiles: str, timeout: int = 10) -> Dict[str, Any]:
    try:
        import requests, urllib.parse
    except ImportError:
        return {"resolved": False, "reason": "requests unavailable"}

    encoded = urllib.parse.quote(smiles)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded}/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON"
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
        }
    except Exception as e:
        return {"resolved": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# MolScribe backend
# ---------------------------------------------------------------------------

class MolScribeBackend:
    """
    Uses MolScribe (https://github.com/thomas0809/MolScribe) to convert
    hand-drawn chemical structure images to SMILES.
    """

    def __init__(self, device: str = "cuda"):
        from molscribe import MolScribe as _MolScribe
        import huggingface_hub
        ckpt_path = huggingface_hub.hf_hub_download(
            "yujieq/MolScribe", "swin_base_char_aux_1m680k.pth"
        )
        self._model = _MolScribe(ckpt_path, device=device)
        print("[MolScribe] Model loaded.")

    def predict(self, image_path: str) -> Dict[str, Any]:
        result = self._model.predict_image_file(image_path, return_atoms_bonds=False)
        return {
            "smiles": result.get("smiles"),
            "confidence": result.get("confidence", 0.0),
        }


# ---------------------------------------------------------------------------
# DECIMER backend
# ---------------------------------------------------------------------------

class DECIMERBackend:
    """
    Uses DECIMER (https://github.com/Kohulan/DECIMER-Image_Transformer) to
    convert hand-drawn chemical structure images to SMILES.
    """

    def __init__(self):
        from DECIMER import predict_SMILES
        self._predict = predict_SMILES
        print("[DECIMER] Ready.")

    def predict(self, image_path: str) -> Dict[str, Any]:
        smiles = self._predict(str(image_path))
        return {"smiles": smiles, "confidence": 0.7}


# ---------------------------------------------------------------------------
# Main recognizer
# ---------------------------------------------------------------------------

class StructureRecognizer:
    """
    High-level chemical structure recognizer.

    Args:
        backend:    "molscribe" | "decimer" | "both"  (both = cascade)
        device:     torch device string ("cuda" or "cpu")
        use_pubchem: whether to do PubChem lookup after recognition
    """

    def __init__(self,
                 backend: str = "molscribe",
                 device: str = "cuda",
                 use_pubchem: bool = True):
        self.backend_name = backend
        self.use_pubchem = use_pubchem
        self._backends: List[Any] = []

        if backend in ("molscribe", "both"):
            try:
                self._backends.append(("molscribe", MolScribeBackend(device=device)))
            except Exception as e:
                print(f"[StructureRecognizer] MolScribe unavailable: {e}")

        if backend in ("decimer", "both"):
            try:
                self._backends.append(("decimer", DECIMERBackend()))
            except Exception as e:
                print(f"[StructureRecognizer] DECIMER unavailable: {e}")

        if not self._backends:
            print("[StructureRecognizer] WARNING: No structure backend loaded. "
                  "Structures will not be recognised.")

    def recognize(self, image_path: str, region_id: str = "unknown",
                  nearby_labels: Optional[List[str]] = None) -> StructureResult:
        """
        Attempt to recognise a chemical structure in the given image crop.
        Tries backends in order; accepts the first valid SMILES.
        Falls back to the second backend if RDKit validation fails.
        """
        nearby_labels = nearby_labels or []
        result = StructureResult(region_id=region_id, backend="none",
                                 nearby_labels=nearby_labels)

        for name, backend in self._backends:
            try:
                raw = backend.predict(image_path)
            except Exception as e:
                result.notes.append(f"{name} prediction error: {e}")
                continue

            smiles = raw.get("smiles")
            confidence = float(raw.get("confidence", 0.0))
            result.raw_output = smiles
            result.backend = name
            result.confidence = confidence

            if not smiles:
                result.notes.append(f"{name}: returned empty SMILES")
                continue

            # Validate with RDKit
            validation = validate_and_canonicalize_smiles(smiles)
            if validation["valid"]:
                result.smiles = smiles
                result.is_valid_smiles = True
                result.canonical_smiles = validation.get("canonical_smiles")
                result.molecular_formula = validation.get("molecular_formula")

                # PubChem lookup
                if self.use_pubchem and result.canonical_smiles:
                    pc = pubchem_lookup_by_smiles(result.canonical_smiles)
                    if pc.get("resolved"):
                        result.pubchem_cid = pc.get("cid")
                        result.molecular_formula = pc.get("molecular_formula") or result.molecular_formula
                        result.iupac_name = pc.get("iupac_name")
                        result.notes.append(f"PubChem resolved: CID={pc.get('cid')}")
                break  # success — stop trying backends

            else:
                result.notes.append(
                    f"{name}: invalid SMILES ({validation.get('reason', 'unknown')})"
                )

        return result

    def recognize_batch(self, image_paths: List[str],
                        region_ids: Optional[List[str]] = None,
                        nearby_labels_list: Optional[List[List[str]]] = None,
                        ) -> List[StructureResult]:
        """Recognise multiple structure crops."""
        region_ids = region_ids or [f"r{i}" for i in range(len(image_paths))]
        nearby_labels_list = nearby_labels_list or [[] for _ in image_paths]
        return [
            self.recognize(p, rid, labels)
            for p, rid, labels in zip(image_paths, region_ids, nearby_labels_list)
        ]
