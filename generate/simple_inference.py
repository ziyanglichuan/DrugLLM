import json
import re
import sys
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM

# -------------------------------------------------
# Add tool paths
# -------------------------------------------------
sys.path.append("../tools/SMILES_to_FGT/")

from encode_smiles import encode_smiles
import decode_smiles

# -------------------------------------------------
# Load model and tokenizer
# -------------------------------------------------
model_path = "../model/DrugLLM"
tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
model = LlamaForCausalLM.from_pretrained(model_path).half().eval().cuda()

# -------------------------------------------------
# Generation parameters
# -------------------------------------------------
max_new_tokens = 128
top_p = 0.8
temperature = 0.9

prompt_template = "Below are molecule modifications:{user_question}"

def generate(model, tokenizer, instruction):
    inputs = tokenizer(
        prompt_template.format(user_question=instruction),
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
        )
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------------------------
# Only preprocess INPUT: SMILES -> FGT
# -------------------------------------------------
def smiles_to_fgt(item):
    instr_smiles_list = re.findall(r'\{([^}]*)\}', item['instruction'])
    for s in instr_smiles_list:
        fgt = encode_smiles(s)
        if fgt:
            item['instruction'] = item['instruction'].replace(
                f'{{{s}}}', f'{{{fgt}}}', 1
            )
    return item

# -------------------------------------------------
# Main (print-only inference)
# -------------------------------------------------
def main(input_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    for idx, item in enumerate(tqdm(data, desc="Processing molecules")):
        item = smiles_to_fgt(item)

        instruction = item['instruction']
        input_value = item.get('input', '')

        # Extract source FGT (last {...} in instruction)
        match_source = re.search(r'\{([^}]*)\}(?!.*\{)', instruction)
        if not match_source:
            print(f"\n[{idx}] ❌ 无法解析 source FGT")
            continue

        fgt_source = match_source.group(1)

        # Model generation
        output_text = generate(model, tokenizer, instruction)

        # Only search in newly generated part
        gen_part = output_text[
            len('Below are molecule modifications:') + len(instruction):
        ]
        match_gen = re.search(r'\{(.*?)}', gen_part)

        print("\n" + "=" * 90)
        print(f"[{idx}] Input property : {input_value}")
        print(f"[{idx}] Instruction    : {instruction}")
        print(f"[{idx}] Model output   :\n{output_text}")

        if not match_gen:
            print(f"[{idx}] ❌ 未生成合法 FGT")
            continue

        fgt_generated = match_gen.group(1)

        # Decode FGT -> SMILES
        smiles_before = decode_smiles.code_to_smiles(fgt_source)
        smiles_after = decode_smiles.code_to_smiles(fgt_generated)

        print(f"[{idx}] SMILES before  : {smiles_before}")
        print(f"[{idx}] SMILES after   : {smiles_after}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DrugLLM inference (input → output only, print mode)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to inference JSON file"
    )
    args = parser.parse_args()

    main(args.input)
