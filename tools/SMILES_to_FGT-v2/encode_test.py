from encode_smiles import encode_smiles  


input_smiles = "CS(=O)(=O)N1CCC2CC(C(=O)NC3CC3)OC2C1"
output_FGT = encode_smiles(input_smiles)
print(output_FGT)
