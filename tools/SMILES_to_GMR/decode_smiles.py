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

# 使用相对路径加载数据文件
vocab_path = os.path.join(current_dir, 'vocab/train_vocab.txt')
vocab_data = read_file(vocab_path)


def process_string(s):
    start_groups = s[:3]
    s = s[3:]
    removed_groups = re.findall(r'[a-zA-Z]{3}', s)
    groups = re.split(r'[a-zA-Z]{3}', s)
    connected_atoms = []
    connected_group_atoms = []
    for group in groups:
        if group:
            a, b = group.split('/')
            connected_atoms.append(int(a))
            connected_group_atoms.append(int(b))
    return start_groups, removed_groups, connected_atoms, connected_group_atoms

def code_to_smiles(code):
    try:
        start_groups, group, connected_atoms, connected_group_atoms = process_string(code)
        start_groups = vocab_data.get(start_groups, "null")
        for i in range(len(group)):
            group[i] = vocab_data.get(group[i], "null")
        start_groups = Chem.MolFromSmiles(start_groups, sanitize=False)
        new_mol = start_groups

        for i in range(len(group)):
            group_item = group[i]
            group_item = Chem.MolFromSmiles(group_item, sanitize=False)
            connected_atoms_item = connected_atoms[i]
            connected_group_atoms_item = connected_group_atoms[i]
            connected_atoms_item = [connected_atoms_item]
            connected_group_atoms_item = [connected_group_atoms_item]
            new_mol = add_group(new_mol, group_item, connected_atoms_item, connected_group_atoms_item, bond_type=1)
            new_smiles = Chem.MolToSmiles(new_mol)
            new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)

        end_smiles = Chem.MolToSmiles(new_mol)
        end_smiles = replace_smiles(end_smiles)
        return end_smiles
    except Exception as e:
        print(f"Error in code_to_smiles: {e}")
        return None
