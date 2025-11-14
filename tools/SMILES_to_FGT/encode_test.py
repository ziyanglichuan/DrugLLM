from encode_smiles import read_file,encode_smiles  
from rdkit import Chem


input_smiles = "c1nnc(CN2CCOCC2)o1"

output_smiles = encode_smiles(input_smiles)
print(output_smiles)
