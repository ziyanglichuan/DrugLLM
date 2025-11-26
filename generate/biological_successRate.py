import os
import sys
import json
import re
import subprocess
import pandas as pd
import numpy as np
import tempfile
from rdkit import Chem
import torch

sys.path.append("../tools/calculate_props/")
sys.path.append("../tools/SMILES_to_FGT/")

import transformers
from transformers import GenerationConfig
import decode_smiles


model_path = "../model/DrugLLM"
token_path = model_path
model_max_length = 1200
max_new_tokens = 128
top_p = 0.8
temperature = 0.9
prompt_template = "Below are molecule modifications:{user_question}"

def get_models_from_pretrained_resume(model_path, token_path, model_max_length):
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_path, 
        cache_dir=None,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        token_path, cache_dir=None, model_max_length=model_max_length,
        padding_side="right", use_fast=False
    )
    return model, tokenizer

def batch_predict_with_cli(unique_smiles_list, model_dirs):
    if not unique_smiles_list:
        return {}

    tmp_input = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv", dir=".")
    tmp_input.write("smiles\n")
    for smi in unique_smiles_list:
        tmp_input.write(f"{smi}\n")
    tmp_input_path = tmp_input.name
    tmp_input.close()

    tmp_output_path = tmp_input_path.replace(".csv", "_preds.csv")
    
    ensemble_results = {smi: [] for smi in unique_smiles_list}

    print(f"[INFO] Running batch prediction on {len(unique_smiles_list)} molecules...")

    try:
        for i, model_dir in enumerate(model_dirs):
            model_name = os.path.basename(model_dir)
            
            base_cmd = [
                "chemprop", "predict",
                "--test-path", tmp_input_path,
                "--model-path", model_dir,
                "--preds-path", tmp_output_path,
                "--accelerator", "gpu",  
                "--devices", "1", 
                "--batch-size", "256",   
                "--num-workers", "0"     
            ]
            
            feat_cmd = base_cmd + ["--features-generators", "v1_rdkit_2d_normalized"]

            success = False
            
            try:
                subprocess.run(feat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                success = True
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(base_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    success = True
                except subprocess.CalledProcessError as e:
                    err_msg = e.stderr.decode('utf-8').strip().split('\n')[-1]
                    print(f"[WARNING] Model {model_name} failed prediction: {err_msg[:100]}...")
                    continue

            # 读取结果并存入集合
            if success and os.path.exists(tmp_output_path):
                try:
                    df_pred = pd.read_csv(tmp_output_path)
                    # 排除 smiles 列，剩下的即为预测值
                    pred_cols = [c for c in df_pred.columns if c != "smiles"]
                    if pred_cols:
                        target_col = pred_cols[0]
                        for _, row in df_pred.iterrows():
                            smi = row['smiles']
                            val = row[target_col]
                            if smi in ensemble_results:
                                ensemble_results[smi].append(val)
                except Exception as e:
                    print(f"[ERROR] Failed to parse results for {model_name}: {e}")
                
                os.remove(tmp_output_path)

    finally:
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)

    final_preds = {}
    for smi, scores in ensemble_results.items():
        if scores:
            final_preds[smi] = np.mean(scores)
        else:
            final_preds[smi] = None
            
    return final_preds

def find_chemprop_dirs(keyword):
    best_model_root = f'../chemprop_train/best_model/{keyword}_activities_best'
    
    if not os.path.exists(best_model_root):
        print(f"[WARNING] 目录不存在: {best_model_root}")
        return []

    model_dirs = []
    items = os.listdir(best_model_root)
    for item in items:
        full_path = os.path.join(best_model_root, item)
        if os.path.isdir(full_path) and "model_" in item:
            model_dirs.append(full_path)
    
    if not model_dirs:
        model_dirs = [best_model_root]
        
    return model_dirs


print("[INFO] Loading DrugLLM...")
llm_model, tokenizer = get_models_from_pretrained_resume(model_path, token_path, model_max_length)
llm_model.eval() 

def generate(user_question):
    inputs = tokenizer(prompt_template.format(user_question=user_question), return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            generation_config=GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if len(sys.argv) < 2:
    print("Usage: python script.py <input.json>")
    sys.exit(1)

input_file = sys.argv[1]
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

input_filename = os.path.splitext(os.path.basename(input_file))[0]
try:
    keyword = input_filename.split('_')[1]
except:
    keyword = input_filename

print(f"[INFO] Task: {keyword}, Total samples: {len(data)}")

print("[INFO] Step 1: Generating molecules with LLM...")
generated_entries = [] 
all_unique_smiles = set()

for i, item in enumerate(data):
    if i % 10 == 0: print(f"  Generating {i}/{len(data)}...")
    
    instruction = item['instruction']
    target_direction = item['input']
    
    match_orig = re.search(r'\{([^}]*)\}', instruction)
    if not match_orig:
        generated_entries.append(None)
        continue
    smiles_before = decode_smiles.code_to_smiles(match_orig.group(1))
    
    output_text = generate(instruction)
    
    prompt_end_idx = output_text.find(instruction) + len(instruction)
    gen_content = output_text[prompt_end_idx:]
    matches = re.findall(r'\{([^\{\}]*)\}', gen_content)
    
    if matches:
        smiles_after = decode_smiles.code_to_smiles(matches[0])
    else:
        smiles_after = None
        
    if smiles_before and smiles_after:
        if Chem.MolFromSmiles(smiles_before) and Chem.MolFromSmiles(smiles_after):
            generated_entries.append({
                "before": smiles_before,
                "after": smiles_after,
                "direction": target_direction
            })
            all_unique_smiles.add(smiles_before)
            all_unique_smiles.add(smiles_after)
        else:
            generated_entries.append(None)
    else:
        generated_entries.append(None)

print(f"[INFO] Step 2: Predicting properties for {len(all_unique_smiles)} unique molecules...")
model_dirs = find_chemprop_dirs(keyword)

if not model_dirs:
    print(f"[ERROR] No ChemProp models found for {keyword}!")
    # sys.exit(1)
    smiles_score_map = {}
else:
    smiles_score_map = batch_predict_with_cli(list(all_unique_smiles), model_dirs)

print("[INFO] Step 3: Calculating success rates and saving...")
os.makedirs('results', exist_ok=True)
result_path = f'results/{input_filename}_results.txt'

success_count = 0
valid_count = 0

with open(result_path, 'w', encoding='utf-8') as f_out:
    for entry in generated_entries:
        if entry is None:
            f_out.write('none\n')
            continue
            
        s_before = entry['before']
        s_after = entry['after']
        direction = entry['direction']
        
        score_before = smiles_score_map.get(s_before)
        score_after = smiles_score_map.get(s_after)
        
        if score_before is not None and score_after is not None:
            is_success = False
            if direction == 'increase' and score_after > score_before:
                is_success = True
            elif direction == 'decrease' and score_after < score_before:
                is_success = True
            
            if is_success:
                success_count += 1
            
            f_out.write(f'{s_before} {s_after}\n')
            valid_count += 1
        else:
            f_out.write('none\n')

final_rate = success_count / valid_count if valid_count > 0 else 0
print(f"[RESULT] Valid Pairs: {valid_count}, Success: {success_count}, Rate: {final_rate:.4f}")

with open('target_success_rate.txt', 'a') as f:
    f.write(f"{keyword}: Success rate: {final_rate:.4f}\n")