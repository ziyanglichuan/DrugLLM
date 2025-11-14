import json
import os
import re
import sys
from multiprocessing import Manager, Pool
from tqdm import tqdm
import argparse

# Add the path for the smiles_reconstruction tools
sys.path.append("../tools/SMILES_to_FGT/")
from encode_smiles import encode_smiles

# Function to process each item in the JSON file
def process_item(item, output_file, lock):
    smiles_list = item['output'].split('}')
    smiles_list = [smiles.strip('-.>{') for smiles in smiles_list if smiles.strip()]
    for smiles in smiles_list:
        encoded_smiles = encode_smiles(smiles)
        if encoded_smiles:
            item['output'] = item['output'].replace(smiles, encoded_smiles, 1)
        else:
            return

    instruction_smiles_list = re.findall(r'\{([^}]*)\}', item['instruction'])
    for instruction_smiles in instruction_smiles_list:
        encoded_instruction_smiles = encode_smiles(instruction_smiles)
        if encoded_instruction_smiles:
            item['instruction'] = item['instruction'].replace('{' + instruction_smiles + '}', '{' + encoded_instruction_smiles + '}', 1)
        else:
            return

    # Write the processed item back to the output file
    with lock:
        with open(output_file, 'a') as f:
            f.write(json.dumps(item))
            f.write('\n')

# Helper function for multiprocessing
def process_item_star(args):
    return process_item(*args)

# Function to process all JSON files in the directory
def process_all_files(input_dir, output_dir):
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    with Manager() as manager:
        lock = manager.Lock()
        pool = Pool()
        tasks = []

        # Read and process each JSON file
        for json_file in json_files:
            input_file = os.path.join(input_dir, json_file)
            output_file = os.path.join(output_dir, f"FGT_{json_file}")
            with open(input_file, 'r') as f:
                data = json.load(f)
                tasks.extend([(item, output_file, lock) for item in data])

        # Process each item using a multiprocessing pool
        for _ in tqdm(pool.imap_unordered(process_item_star, tasks), total=len(tasks)):
            pass

    # Write the processed data back to the respective output files in a pretty JSON format
    for json_file in json_files:
        output_file = os.path.join(output_dir, f"FGT_{json_file}")
        data = []
        with open(output_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)

        with open(output_file, 'w') as f:
            f.write('[\n')
            for i in range(len(data)):
                item = data[i]
                if i != len(data) - 1:
                    f.write(json.dumps(item, indent=4) + ',\n')
                else:
                    f.write(json.dumps(item, indent=4) + '\n')
            f.write(']')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process all JSON files in a directory.')
parser.add_argument('input_dir', type=str, help='The input directory containing JSON files to process')
parser.add_argument('output_dir', type=str, help='The output directory')

args = parser.parse_args()

# Call the function with the provided arguments
process_all_files(args.input_dir, args.output_dir)
