from smiles_change import *
import argparse
import itertools
from multiprocessing import Pool, Manager
from tqdm import tqdm


def extVocab_smiles(smiles_list):
    vocab_set = set()
    for i, smiles in enumerate(smiles_list):
        if not is_valid_smiles(smiles) or smiles == "":
            continue

        canonical_smiles = replace_smiles(smiles)
        mol = Chem.MolFromSmiles(canonical_smiles)

        try:
            atom_groups = get_atom_groups(mol)
            removed_groups, start_groups = remove_groups(mol, atom_groups)

        except Exception as e:
            continue

        start_smiles = Chem.MolToSmiles(start_groups)
        vocab_set.add(start_smiles)

        for group, connected_atoms, connected_group_atoms in removed_groups:
            group_smiles = Chem.MolToSmiles(group)
            vocab_set.add(group_smiles)
    vocab_list = list(vocab_set)
    return vocab_list


def encode_vocab(vocab_list):
    vocab_dict = {}
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, code in enumerate(itertools.product(chars, repeat=3)):
        if i >= len(vocab_list):
            break
        vocab_dict[vocab_list[i]] = ''.join(code)
    return vocab_dict


def process_smiles(smiles, vocab_list):
    vocab = extVocab_smiles([smiles])
    vocab_list.extend(vocab)


def process_smiles_wrapper(args):
    smiles, vocab_list = args
    process_smiles(smiles, vocab_list)


def ext_file(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        smiles_list = file.read().split('\n')

    # Create a Manager to share data between processes
    manager = Manager()
    vocab_list = manager.list()

    # Create a pool of worker processes
    pool = Pool(processes=24)
    for _ in tqdm(
            pool.imap_unordered(process_smiles_wrapper, ((smile, vocab_list) for smile in smiles_list), chunksize=10000),
            total=len(smiles_list), miniters=100):
        pass
    pool.close()
    pool.join()

    # Remove duplicates and sort the vocabulary list
    vocab_set = set(vocab_list)
    vocab_list = sorted(list(vocab_set))

    # Encode the vocabulary list
    vocab_dict = encode_vocab(vocab_list)

    # Write the results to the output file
    with open(output_filename, 'w') as outfile:
        for smiles, code in vocab_dict.items():
            outfile.write(f'{smiles} {code}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filename', help='Input file name')
    parser.add_argument('output_filename', help='Output file name')
    args = parser.parse_args()

    ext_file(args.input_filename, args.output_filename)
