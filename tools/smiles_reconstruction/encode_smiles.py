from smiles_change import *
import random
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        smiles, code = line.strip().split()
        data[smiles] = code
    return data

def encode_smiles(smiles, vocab_data):
    # 检查smiles是否有效
    if not is_valid_smiles(smiles) or smiles == "":
        print(f"Invalid SMILES: {smiles}")
        return None

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
            success = True
            break
        else:
            atom_groups.reverse()

    if not success:
        #print(f"Failed case: original SMILES: {canonical_smiles}, transformed SMILES: {end_smiles}")
        return None
    else:
        result_str = ''
        start_smiles = Chem.MolToSmiles(start_groups)
        if start_smiles in vocab_data:
            code = vocab_data[start_smiles]
            result_str += code
        else:
            #print(f'{start_smiles} not found in vocab_file')
            return None
        for group, connected_atoms, connected_group_atoms in reversed(removed_groups):
            group_smiles = Chem.MolToSmiles(group)
            if group_smiles in vocab_data:
                code = vocab_data[group_smiles]
                result_str += f'{connected_atoms[0]}/{connected_group_atoms[0]}'
                result_str += code
            else:
                #print(f'{group_smiles} not found in vocab_file')
                return None

        return result_str

        
        
        
        
