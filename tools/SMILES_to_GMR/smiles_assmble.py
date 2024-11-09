from smiles_change import *

smiles = "O=C(NCc1cc(-c2ccc(Cl)cc2)on1)c1ccco1"
smiles = "O=C(NCc1cc(-c2ccccc2)on1)c1ccco1"
canonical_smiles = replace_smiles(smiles)
mol = Chem.MolFromSmiles(canonical_smiles)
print("smiles:",canonical_smiles)

atom_groups = get_atom_groups(mol)
#print(atom_groups)

for groups in atom_groups:
    smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(groups))
    print("atom_groups:",smiles)


#atom_groups.reverse()

removed_groups, start_groups = remove_groups(mol, atom_groups)

for group, connected_atoms, connected_group_atoms in removed_groups:
    group_smiles = Chem.MolToSmiles(group)
    print(f"Removed group: {group_smiles}, connected to: {connected_atoms},group_connect:{connected_group_atoms}")

new_mol = start_groups
new_smiles = Chem.MolToSmiles(new_mol)
print(new_smiles)

for group, connected_atoms, connected_group_atoms in reversed(removed_groups):
   
    new_mol = add_group(new_mol, group, connected_atoms, connected_group_atoms,bond_type=1)
    
    new_smiles = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)

    new_smiles = Chem.MolToSmiles(new_mol)
    print(new_smiles)
    
end_smiles = replace_smiles(new_smiles)
print(end_smiles)