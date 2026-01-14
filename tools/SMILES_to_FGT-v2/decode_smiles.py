import os
import re
import sys
from smiles_change import *
from rdkit import Chem

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        smiles, code = line.strip().split()
        data[code] = smiles
    return data

# 动态添加模块路径（
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

vocab_path = os.path.join(current_dir, 'vocab/vocab.txt')
vocab_data = read_file(vocab_path)


def process_string(s):
    start_groups = s[:3]
    s = s[3:]
    removed_groups = re.findall(r'[a-zA-Z]{3}', s)
    groups = re.split(r'[a-zA-Z]{3}', s)
    
    connected_atoms = []
    connected_group_atoms = []
    bond_types = []
    
    # Define possible bond types as separators
    bond_symbols = ['-', '=', '#', '.', '<', '>']
    for group in groups:
        if group:
            for symbol in bond_symbols:
                if symbol in group:
                    a, b = group.split(symbol)
                    # 如果为空字符串，则赋值为 0
                    a = int(a) if a else 0
                    b = int(b) if b else 0

                    connected_atoms.append(a)
                    connected_group_atoms.append(b)
                    bond_types.append(symbol)
                    break

    return start_groups, removed_groups, connected_atoms, connected_group_atoms,bond_types


def decode_smiles(code):
    try:
        start_groups, group, connected_atoms, connected_group_atoms, bond_types = process_string(code)
        start_groups = vocab_data.get(start_groups, "null")
        for i in range(len(group)):
            group[i] = vocab_data.get(group[i], "null")
        start_groups = Chem.MolFromSmiles(start_groups)
        new_mol = start_groups

        for i in range(len(group)):
            group_item = group[i]
            group_item = Chem.MolFromSmiles(group_item, sanitize=False)
            connected_atoms_item = connected_atoms[i]
            connected_group_atoms_item = connected_group_atoms[i]
            connected_atoms_item = [connected_atoms_item]
            connected_group_atoms_item = [connected_group_atoms_item]
            
            # Set bond_type based on bond_types[i]
            if bond_types[i] == "-":
                bond_type = 1
            elif bond_types[i] == "=":
                bond_type = 2
            elif bond_types[i] == "#":
                bond_type = 3
            elif bond_types[i] == ".":
                bond_type = 4
            elif bond_types[i] == "<":
                bond_type = 5
            elif bond_types[i] == ">":
                bond_type = 6
            else:
                bond_type = 1  # Default

            new_mol = add_group(new_mol, group_item, connected_atoms_item, connected_group_atoms_item, bond_type=bond_type)
            new_smiles = Chem.MolToSmiles(new_mol)
            new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)

        end_smiles = Chem.MolToSmiles(new_mol)
        end_smiles = replace_smiles(end_smiles)
        return end_smiles
    except Exception as e:
        # print(f"Error in code_to_smiles: {e}")
        return None

