import os
import json
import re
import sys
import typing
from chemprop.data import MoleculeDatapoint, MoleculeDataset, MoleculeDataLoader
from chemprop.utils import load_checkpoint
from chemprop.train import predict
from rdkit import Chem

sys.path.append("../tools/calculate_props/")
sys.path.append("../tools/SMILES_to_FGT/")

import transformers
from transformers import GenerationConfig
import decode_smiles


model_path = "../model/DrugLLM"
token_path = model_path
model_max_length = 1200

def get_models_from_pretrained_resume(model_path, token_path, model_max_length):
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_path,
        cache_dir=None,
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        token_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    return model, tokenizer

llm_model, tokenizer = get_models_from_pretrained_resume(model_path, token_path, model_max_length)
llm_model.half().eval().cuda()


def load_chemprop_best_model(keyword):
    best_model_dir = f'../chemprop_train/best_model/{keyword}_activities_best'
    model_files = []
    for fname in os.listdir(best_model_dir):
        if fname.startswith("model_"):
            model_files.append(os.path.join(best_model_dir, fname))
    if not model_files:
        raise FileNotFoundError(f"No model_i found in {best_model_dir}")
    models = [load_checkpoint(os.path.join(m, "model_0")) for m in model_files]
    return models

def predict_smiles(models, smiles_string):
    """Predict activity of SMILES using ChemProp ensemble"""
    if Chem.MolFromSmiles(smiles_string) is None:
        return None

    data = MoleculeDataset([MoleculeDatapoint(smiles=[smiles_string])])
    data_loader = MoleculeDataLoader(dataset=data, batch_size=1)

    all_preds = []
    for model in models:
        preds = predict(model, data_loader)
        all_preds.append(preds)

    # Ensemble mean
    import numpy as np
    return float(np.mean(all_preds))


max_new_tokens = 128
top_p = 0.8
temperature = 0.9

prompt = "Below are molecule modifications:{user_question}"

def generate(user_question):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
    outputs = llm_model.generate(
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


input_file = sys.argv[1]
with open(input_file, 'r') as f:
    data = json.load(f)

input_filename = os.path.splitext(os.path.basename(input_file))[0]
keyword = input_filename.split('_')[1]

chem_models = load_chemprop_best_model(keyword)

os.makedirs('results', exist_ok=True)
with open(f'results/{input_filename}_results.txt', 'w') as results_file:
    success_count = 0
    total_count = 0

    for item in data:
        instruction = item['instruction']
        input_value = item['input']

        # Generate molecule modification with DrugLLM
        output = generate(instruction)
        # Extract content within {}
        pat = r'\{[^\{\}]*\}'
        match = re.findall(pat, output[len('Below are molecule modifications:') + len(instruction):])
        if not match:
            results_file.write('none\n')
            continue

        result = match[0][1:-1]
        smiles_after = decode_smiles.code_to_smiles(result)
        if smiles_after is None:
            results_file.write('none\n')
            continue

        # Original SMILES
        match2 = re.search(r'\{([^}]*)\}(?!.*\{)', instruction)
        if not match2:
            results_file.write('none\n')
            continue
        smiles_before = decode_smiles.code_to_smiles(match2.group(1))
        if smiles_before is None:
            results_file.write('none\n')
            continue

        # Predict activity using ChemProp
        pred_before = predict_smiles(chem_models, smiles_before)
        pred_after = predict_smiles(chem_models, smiles_after)

        if pred_before is not None and pred_after is not None:
            if input_value == 'increase' and pred_after > pred_before:
                success_count += 1
            elif input_value == 'decrease' and pred_after < pred_before:
                success_count += 1
        total_count += 1

        results_file.write(f'{smiles_before} {smiles_after}\n')
        results_file.flush()

success_rate = success_count / total_count if total_count > 0 else 0
print(f"Success rate: {success_rate}")

with open('target_success_rate.txt', 'a') as f:
    f.write(f"{keyword}: Success rate: {success_rate}\n")
