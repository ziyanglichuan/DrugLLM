import os
import json
import re
import sys
import csv
import transformers
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedModel
import typing

sys.path.append("../tools/calculate_props/")
sys.path.append("../tools/SMILES_to_FGT/")
import decode_smiles
from success_rate import *

model_path = "../model/DrugLLM"
token_path = model_path
model_max_length = 1200

def get_models_from_pretrained_resume(model_path: str, token_path: str, model_max_length: int) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model:PreTrainedModel = transformers.LlamaForCausalLM.from_pretrained(
        model_path,
        cache_dir=None,
    )

    tokenizer:PreTrainedTokenizer = transformers.LlamaTokenizer.from_pretrained(
        token_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    return model, tokenizer

model, tokenizer = get_models_from_pretrained_resume(model_path, token_path, model_max_length)
model.half().eval().cuda()
print("Load checkpoint")

max_new_tokens = 128
top_p = 0.8
temperature=0.9

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

prompt = "Below are molecule modifications:{user_question}"

# Read json file
input_file = sys.argv[1] 
with open(input_file, 'r') as f:
    data = json.load(f)

input_filename = os.path.basename(input_file).split('_')[1]

# Create output directory
output_dir = './success_results'
os.makedirs(output_dir, exist_ok=True)

results_file_path = os.path.join(output_dir, f'{input_filename}.csv')
success_file_path = os.path.join(output_dir, f'{input_filename}_success_rate.txt')

calculator = PropertyCalculator()
correct_count = 0
total_count = 0

with open(results_file_path, 'w', newline='') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(['SMILES_before', 'SMILES_after', 'Input_value', 'Success'])

    for item in data:
        instruction = item['instruction']
        input_value = item['input']
        match_source = re.search(r'\{([^}]*)\}(?!.*\{)', instruction)

        if match_source:
            result_source = match_source.group(1)
        else:
            writer.writerow(['none', 'none', input_value, 'False'])
            print('No match found in input\n')
            continue
        
        output = generate(model, instruction)

        # Extract content using regex
        match_generated = re.search(r'{(.*?)}', output[len('Below are molecule modifications:') + len(instruction):])
        
        if match_generated:
            result_generated = match_generated.group(1)

            # Decode SMILES
            smiles_before = decode_smiles.code_to_smiles(result_source)
            smiles_after = decode_smiles.code_to_smiles(result_generated)
            
            success_flag = False
            if smiles_before is not None and smiles_after is not None and smiles_before != smiles_after:
                total_count += 1
                prop1_func = next(key for key, value in calculator.prop_names.items() if value == f'{input_filename}')
                comparison = [f'{input_value}']
                is_success = calculator.ifSuccess(smiles_before, smiles_after, comparison, prop1=prop1_func, prop2=None)
                correct_count += is_success
                success_flag = bool(is_success)
                success_rate = (correct_count / total_count) * 100 if total_count > 0 else 0
                print(f"Input SMILES: {smiles_before} -> Generated SMILES: {smiles_after}")
                print(f"Success Rate: {success_rate:.2f}%")
            
            writer.writerow([smiles_before, smiles_after, input_value, success_flag])
            results_file.flush()
        else:
            writer.writerow(['none', 'none', input_value, 'False'])
            print('No match found in generated output\n')
            continue

# Calculate success rate and write to file
success_rate = (correct_count / total_count) * 100 if total_count > 0 else 0
print(f"Final Success Rate: {success_rate:.2f}%")

with open(success_file_path, 'w') as success_file:
    success_file.write(f"Success Rate: {success_rate:.2f}%\n")
