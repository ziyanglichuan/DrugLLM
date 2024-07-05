from encode_smiles import read_file,encode_smiles  

vocab_data =read_file('/home/guoyan/project/qlora/smiles_reconstruction/vocab/train_vocab.txt')

input_smiles = "Nc1nnc(-c2c(Cl)c(Cl)cc(Cl)c2)c(N)n1"
output_smiles = encode_smiles(input_smiles,vocab_data)
print(output_smiles)
