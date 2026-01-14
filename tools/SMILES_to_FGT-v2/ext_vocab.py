import itertools
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from smiles_change import remove_groups
from rdkit import Chem 


def ext_vocab_smiles(smiles):
    vocab = set()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return vocab

        removed_groups, start_groups = remove_groups(mol)

        start_smiles = Chem.MolToSmiles(start_groups, canonical=True, isomericSmiles=False)
        vocab.add(start_smiles)

        for group, _, _, _ in removed_groups:
            group_smiles = Chem.MolToSmiles(group, canonical=True, isomericSmiles=False)
            vocab.add(group_smiles)

    except Exception as e:
        print(f"Error processing {smiles}: {e}")  # 仅打印错误，不抛出异常
        return vocab  # 返回一个可序列化的值（空 set）

    return vocab

def encode_vocab(vocab_list):
    vocab_dict = {}
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    code_iter = itertools.product(chars, repeat=3)
    for smiles, code in zip(vocab_list, code_iter):
        vocab_dict[smiles] = ''.join(code)
    return vocab_dict

def process_smiles(smiles):
    try:
        return ext_vocab_smiles(smiles)
    except Exception as e:
        # print(f"Error in process_smiles: {smiles}, {e}")
        return set()  # 返回可 pickle 的空 set

def ext_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        smiles_list = [line.strip() for line in file if line.strip()]

    # Determine optimal number of processes
    num_processes = min(cpu_count(), 250)  

    with Pool(processes=num_processes) as pool:
        # Use imap for memory efficiency
        results = list(tqdm(pool.imap(process_smiles, smiles_list, chunksize=100),
                            total=len(smiles_list),
                            desc="Processing SMILES"))

    # Combine all vocab sets
    vocab_set = set().union(*results)
    vocab_list = sorted(vocab_set)

    # Encode the vocabulary list
    vocab_dict = encode_vocab(vocab_list)

    # Write the results to the output file
    with open(output_filename, 'w') as outfile:
        for smiles, code in vocab_dict.items():
            outfile.write(f'{smiles} {code}\n')

# Main function with hardcoded file paths
if __name__ == '__main__':
    input_filename = '../test_data/train_0.2b.txt' 
    # input_filename = './smiles_test.txt'
    output_filename = './vocab/add_vocab/vocab_train_0.2b.txt'
    ext_file(input_filename, output_filename)
