from smiles_change import *
import random
import json
from tqdm import tqdm
from multiprocessing import Pool, Manager
import os
import sys


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        smiles, code = line.strip().split()
        data[smiles] = code
    return data

# 动态添加模块路径（
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 使用相对路径加载数据文件
vocab_path = os.path.join(current_dir, 'vocab/vocab.txt')
error_path = os.path.join(current_dir, './vocab/add_vocab/ext_error.txt')
vocab_data = read_file(vocab_path)


def write_unique_smiles_to_error_file(smiles, filepath=error_path):
    if not os.path.exists(filepath):
        existing_smiles = set()
    else:
        with open(filepath, 'r') as f:
            existing_smiles = set(line.strip() for line in f if line.strip())

    if smiles not in existing_smiles:
        with open(filepath, 'a') as f:
            f.write(f'{smiles}\n')

def encode_smiles(smiles):
    if not is_valid_smiles(smiles) or smiles == "":
        print(f"Invalid SMILES: {smiles}")
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()  # 获取原子数
        canonical_smiles = atom_group_to_smiles(mol, list(range(num_atoms)))
    except AttributeError:
        print("Warning: SMILES is invalid or None.")
        return None

    # if not is_valid_smiles(canonical_smiles) or smiles == "":
    #     print(f"Invalid SMILES: {smiles}")
    #     return None

    mol = Chem.MolFromSmiles(canonical_smiles)

    success = False

    removed_groups, start_groups = remove_groups(mol)

    new_mol = start_groups
    try:
        for group, connected_atoms, connected_group_atoms, bond_type in reversed(removed_groups):
            new_mol = add_group(new_mol, group, connected_atoms, connected_group_atoms, bond_type=bond_type)
            new_smiles = Chem.MolToSmiles(new_mol)

            new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)
            # print(new_smiles)
        end_smiles = Chem.MolToSmiles(new_mol)
    except Exception as e:
        #print("Encode error")
        return None

    if canonical_smiles == end_smiles:
        success = True
    else:
        end_smiles = replace_smiles(end_smiles)

    if canonical_smiles == end_smiles:
        success = True

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
            #write_unique_smiles_to_error_file(start_smiles)
            return None

        for group, connected_atoms, connected_group_atoms, bond_type in reversed(removed_groups):
            group_smiles = Chem.MolToSmiles(group)
            if group_smiles in vocab_data:
                code = vocab_data[group_smiles]

                if bond_type == 1:
                    bond_symbol = '-'
                elif bond_type == 2:
                    bond_symbol = '='
                elif bond_type == 3:
                    bond_symbol = '#'
                elif bond_type == 4:
                    bond_symbol = '.'
                elif bond_type == 5:
                    bond_symbol = '<'
                elif bond_type == 6:
                    bond_symbol = '>'
                else:
                    bond_symbol = '-'
                    
                parts = []
                if connected_atoms[0] != 0:
                    parts.append(str(connected_atoms[0]))
                parts.append(bond_symbol)
                if connected_group_atoms[0] != 0:
                    parts.append(str(connected_group_atoms[0]))
                result_str += ''.join(parts) + code
                
                # result_str += f'{connected_atoms[0]}{bond_symbol}{connected_group_atoms[0]}'
                # result_str += code
            else:
                #print(f'{group_smiles} not found in vocab_file')
                #write_unique_smiles_to_error_file(group_smiles)
                return None

        return result_str


