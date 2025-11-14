import os
import re
import sys
from smiles_change import *
from rdkit import Chem

from rdkit import RDLogger

# 关闭RDKit的所有日志输出
RDLogger.DisableLog('rdApp.*')

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

        # 初始化分子
        start_groups = Chem.MolFromSmiles(start_groups, sanitize=False)
        if start_groups is None:
            return None

        new_mol = start_groups

        for i in range(len(group)):
            group_item = Chem.MolFromSmiles(group[i], sanitize=False)
            if group_item is None:
                break

            connected_atoms_item = [connected_atoms[i]]
            connected_group_atoms_item = [connected_group_atoms[i]]

            prev_mol = new_mol  

            try:
                new_mol = add_group(prev_mol, group_item, connected_atoms_item, connected_group_atoms_item, bond_type=1)
                new_smiles = Chem.MolToSmiles(new_mol)
                tmp_mol = Chem.MolFromSmiles(new_smiles)
                if tmp_mol is None:
                    new_mol = prev_mol  
                    break

                new_mol = tmp_mol 

            except Exception as e:
                new_mol = prev_mol 
                break 

        if new_mol is None:
            return None

        end_smiles = Chem.MolToSmiles(new_mol)
        end_smiles = replace_smiles(end_smiles)
        return end_smiles

    except Exception as e:
        print(f"Error in code_to_smiles: {e}")
        return None
