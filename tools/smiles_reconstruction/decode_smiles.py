# -*- coding:gbk -*-
from smiles_change import *
import re

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    data = {}
    for line in lines:
        smiles, code = line.strip().split()
        data[code] = smiles
    return data

vocab_data = read_file('/home/guoyan/project/qlora/smiles_reconstruction/vocab/train_vocab.txt')

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
            connected_atoms.append((int(a)))
            connected_group_atoms.append((int(b)))
    return start_groups, removed_groups, connected_atoms, connected_group_atoms

def code_to_smiles(code):
    try:
        start_groups, group, connected_atoms, connected_group_atoms = process_string(code)
        start_groups = vocab_data.get(start_groups, "null")
        for i in range(len(group)):
            group[i] = vocab_data.get(group[i], "null")
        # �ؽ�����
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
    except:
        return None

'''

result = []
for smiles, code in encode_data.items():
    result.append(code_to_smiles(code))

success_rate = 0
for end_smiles in result:
    print(end_smiles)
    if end_smiles in encode_data:
        success_rate += 1
print(success_rate / len(encode_data))
'''
