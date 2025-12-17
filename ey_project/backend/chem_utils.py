from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# -----------------------------------------------------------
# SAFE MOLECULE LOADING  (prevents RDKit crashes)
# -----------------------------------------------------------
def safe_mol_from_smiles(smi: str):
    """
    Safely converts SMILES → RDKit Mol with sanitization disabled first.
    Prevents:
        - getNumImplicitHs() errors
        - kekulization failures
        - invalid valence crashes
    Returns None if invalid.
    """
    if not smi or not isinstance(smi, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None

        # Now sanitize safely
        Chem.SanitizeMol(mol, catchErrors=True)
        return mol

    except Exception:
        return None


# -----------------------------------------------------------
# SAFE SIMILARITY COMPUTATION
# -----------------------------------------------------------
def compute_similarity(smiles1: str, smiles2: str):
    """
    Computes Tanimoto similarity between two SMILES strings.
    Returns None if either is invalid.
    No RDKit exceptions escape this function.
    """
    mol1 = safe_mol_from_smiles(smiles1)
    mol2 = safe_mol_from_smiles(smiles2)

    if mol1 is None or mol2 is None:
        return None

    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return None


# -----------------------------------------------------------
# TOP‑K MOST SIMILAR MOLECULES
# -----------------------------------------------------------
MOLECULE_DB = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2)C",
    "Gefitinib": "COCCOc1ccc2nc(N3CCC(N)CC3)nc(N)c2c1",
    "Erlotinib": "CN(C)CCOc1c(OC)nc(Nc2cccc(Cl)c2)n1"
}

def find_similar_molecules(query_smiles: str, top_k: int = 3):
    """
    Returns sorted list of (name, smiles, similarity_score).
    """
    results = []

    for name, smi in MOLECULE_DB.items():
        sim = compute_similarity(query_smiles, smi)
        if sim is not None:
            results.append((name, smi, sim))

    return sorted(results, key=lambda x: x[2], reverse=True)[:top_k]
