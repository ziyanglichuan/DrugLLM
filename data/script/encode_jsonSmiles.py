import json
from multiprocessing import Manager, Pool
from tqdm import tqdm
import sys
import os
import argparse
import re

sys.path.append("/home/guoyan/project/qlora/smiles_reconstruction/")
from encode_smiles import read_file,encode_smiles

def process_item(item, output_file, lock):
    smiles_list = item['output'].split('}')
    smiles_list = [smiles.strip('-.>{') for smiles in smiles_list if smiles.strip()]
    for smiles in smiles_list:
        encoded_smiles = encode_smiles(smiles, vocab_data)
        if encoded_smiles:
            item['output'] = item['output'].replace(smiles, encoded_smiles, 1)
        else:
            return

    instruction_smiles_list = re.findall(r'\{([^}]*)\}', item['instruction'])
    for instruction_smiles in instruction_smiles_list:
        encoded_instruction_smiles = encode_smiles(instruction_smiles, vocab_data)
        if encoded_instruction_smiles:
            item['instruction'] = item['instruction'].replace('{' + instruction_smiles + '}', '{' + encoded_instruction_smiles + '}', 1)
        else:
            return

    with lock:
        with open(output_file, 'a') as f:
            f.write(json.dumps(item))
            f.write('\n')


# def process_item(item, output_file, lock):
#     smiles_list = item['output'].split('}')
#     smiles_list = [smiles.strip('-.>{') for smiles in smiles_list if smiles.strip()]
#     for smiles in smiles_list:
#         encoded_smiles = encode_smiles(smiles, vocab_data)
#         if encoded_smiles:
#             item['output'] = item['output'].replace(smiles, encoded_smiles, 1)
#         else:
#             return

#     instruction_smiles_list = item['instruction'].split('Output: ')[1].split('}')
#     instruction_smiles_list = [smiles.strip('-.>{') for smiles in instruction_smiles_list if smiles.strip()]
#     for instruction_smiles in instruction_smiles_list:
#         encoded_instruction_smiles = encode_smiles(instruction_smiles, vocab_data)
#         if encoded_instruction_smiles:
#             item['instruction'] = item['instruction'].replace(instruction_smiles, encoded_instruction_smiles, 1)
#         else:
#             return

#     with lock:
#         with open(output_file, 'a') as f:
#             f.write(json.dumps(item))
#             f.write('\n')

def process_item_star(args):
    return process_item(*args)

vocab_data =read_file('/home/guoyan/project/qlora/smiles_reconstruction/vocab/train_vocab.txt') 

def process_file(filename, input_dir, output_dir):
    input_file = os.path.join(input_dir, filename)

    with open(input_file, 'r') as f:
        data = json.load(f)

    output_file = os.path.join(output_dir, 'encode_' + filename)

    with Manager() as manager:
        lock = manager.Lock()
        pool = Pool()
        for _ in tqdm(pool.imap_unordered(process_item_star, [(item, output_file, lock) for item in data]), total=len(data)):
            pass
    
    data = []
    with open(output_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
            
    with open(output_file, 'w') as f:
        f.write('[\n')
        for i in range(len(data)):  # 使用处理后的数据
            item = data[i]
            if i != len(data) - 1:
                f.write(json.dumps(item,indent=4) + ',\n')
            else:
                f.write(json.dumps(item,indent=4) + '\n')
        f.write(']')

# 解析命令行参数
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('filename', type=str, help='The name of the file to process')
parser.add_argument('input_dir', type=str, help='The directory of the input file')
parser.add_argument('output_dir', type=str, help='The directory of the output file')

args = parser.parse_args()

# 调用函数
process_file(args.filename, args.input_dir, args.output_dir)