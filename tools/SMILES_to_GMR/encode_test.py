from encode_smiles import read_file,encode_smiles  
from rdkit import Chem


input_smiles = "Nc1nnc(-c2c(Cl)c(Cl)cc(Cl)c2)c(N)n1"

output_smiles = encode_smiles(input_smiles)
print(output_smiles)
