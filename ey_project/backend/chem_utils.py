from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Small internal molecule database for similarity checking
MOLECULE_DB = {
    "Gefitinib": "COC1=CC2=C(C=C1)N(C=N2)CC3=CC(=C(C=C3)F)O",
    "Erlotinib": "COC1=CC=C2N=CN(C2=C1)C3=CC=C(C=C3)OCC#C",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C",
    "Imatinib": "CC1=NC(=CN=C1N)NCC2=CC=C(C=C2)NC3=NC=CC(=N3)N",
    "Bevacizumab": ""  # Antibody — no SMILES (will be skipped)
}


def compute_similarity(smiles1: str, smiles2: str):
    """
    Computes Tanimoto similarity between two SMILES strings.
    Returns None if either molecule cannot be parsed.
    """
    if not smiles1 or not smiles2:
        return None

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if not mol1 or not mol2:
        return None

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

    return DataStructs.TanimotoSimilarity(fp1, fp2)


def find_similar_molecules(query_smiles: str, top_k: int = 3):
    """
    Returns top-K most similar molecules from MOLECULE_DB.
    """

    results = []
    for name, smiles in MOLECULE_DB.items():
        if not smiles:
            continue  # Skip biologics such as Bevacizumab (no SMILES)

        sim = compute_similarity(query_smiles, smiles)
        if sim:
            results.append((name, smiles, sim))

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]
