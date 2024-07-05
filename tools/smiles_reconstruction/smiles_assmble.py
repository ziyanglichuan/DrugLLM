# -*- coding:gbk -*-
from smiles_change import *


# 从SMILES字符串创建分子
smiles = "O=C(NCc1cc(-c2ccc(Cl)cc2)on1)c1ccco1"
smiles = "O=C(NCc1cc(-c2ccccc2)on1)c1ccco1"
canonical_smiles = replace_smiles(smiles)
mol = Chem.MolFromSmiles(canonical_smiles)
print("smiles:",canonical_smiles)

atom_groups = get_atom_groups(mol)
#print(atom_groups)
#部分基团转换为SMILES字符串
for groups in atom_groups:
    smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(groups))
    print("atom_groups:",smiles)


#atom_groups.reverse()

# 调用remove_groups函数
removed_groups, start_groups = remove_groups(mol, atom_groups)

# 打印结果
for group, connected_atoms, connected_group_atoms in removed_groups:
    group_smiles = Chem.MolToSmiles(group)
    print(f"Removed group: {group_smiles}, connected to: {connected_atoms},group_connect:{connected_group_atoms}")

# 逆反remove_groups函数的过程
new_mol = start_groups
new_smiles = Chem.MolToSmiles(new_mol)
print(new_smiles)

for group, connected_atoms, connected_group_atoms in reversed(removed_groups):
   
    # 如果connected_atoms不为空，则将原子组按照connected_atoms的位置连接到起始分子上生成新分子
    new_mol = add_group(new_mol, group, connected_atoms, connected_group_atoms,bond_type=1)
    
    # 获取new_mol分子的SMILES字符串
    new_smiles = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)

    new_smiles = Chem.MolToSmiles(new_mol)
    print(new_smiles)
    
end_smiles = replace_smiles(new_smiles)
print(end_smiles)
'''
with open('/home/guoyan/project/qlora/data/molecule/test_smiles.txt', 'r') as file:
    smiles_list = file.read().split('\n')


with open('success_rate.txt', 'w') as success_file, open('invalid_failed_cases.txt', 'w') as invalid_failed_file:
    success_count = 0
    counter = 0
    for smiles in smiles_list:

        if not is_valid_smiles(smiles) or smiles == "":
            invalid_failed_file.write(f"Invalid SMILES: {smiles}\n")
            continue

        counter += 1
        # 从SMILES字符串创建分子
        canonical_smiles = replace_smiles(smiles)
        mol = Chem.MolFromSmiles(canonical_smiles)

        atom_groups = get_atom_groups(mol)

        success = False
        for i in range(2):
            # 调用remove_groups函数
            removed_groups, start_groups = remove_groups(mol, atom_groups)

            # 逆反remove_groups函数的过程
            new_mol = start_groups

            for group, connected_atoms, connected_group_atoms in reversed(removed_groups):
                new_mol = add_group(new_mol, group, connected_atoms, connected_group_atoms, bond_type=1)
                new_smiles = Chem.MolToSmiles(new_mol)
                new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)

            end_smiles = Chem.MolToSmiles(new_mol)
            end_smiles = replace_smiles(end_smiles)

            if canonical_smiles == end_smiles:
                success_count += 1
                success = True
                break
            else:
                atom_groups.reverse()

        if not success:
            invalid_failed_file.write(f"Failed case: original SMILES: {canonical_smiles}, transformed SMILES: {end_smiles}\n")

        if counter % 5000 == 0:
            success_rate = success_count / counter
            success_file.write(f"Success rate after processing {counter} SMILES: {success_rate}\n")
            success_file.flush()

    success_rate = success_count / len(smiles_list)
    success_file.write(f"Final success rate: {success_rate}\n")
'''
