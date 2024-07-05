import os
import json
import re
import sys
sys.path.append("../tools/")
sys.path.append("../tools/smiles_reconstruction/")
sys.path.append("../tools/calculate_props/")
import decode_smiles
from success_rate import *
import transformers
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedModel
import typing

model_path = "../model/checkpoint-last"
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

print("加载checkpoint")

max_new_tokens = 128
top_p = 0.9
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


prompt = (
    "Below are molecule modifications:{user_question}"
)

# 读取json文件
#input_file = sys.argv[1] # 获取第一个命令行参数
input_file = "../data/encode_data/encode_change5_offset9_TPSA_15000.json"
with open(input_file, 'r') as f:
    data = json.load(f)

input_filename = os.path.splitext(os.path.basename(input_file))[0]

with open(f'./output/Umap_data/TPSA/source_space.txt', 'w') as source_file, open(f'./output/Umap_data/TPSA/target_space.txt', 'w') as target_file, open(f'./output/Umap_data/TPSA/generated_space.txt', 'w') as generated_file:
    for item in data:
        print("----------------------------")
        instruction = item['instruction']
        input_value = item['input']
        output_value = item['output']
        match_source = re.search(r'\{([^}]*)\}(?!.*\{)', instruction)
        match_target = re.search(r'{(.*?)}', output_value)
        if match_source and match_target:
            result_source = match_source.group(1)
            result_target = match_target.group(1)
            print(result_source)
            print(result_target)
            result_source = decode_smiles.code_to_smiles(result_source)
            result_target = decode_smiles.code_to_smiles(result_target)
            if result_source == None or result_target == None:
                print('decode1 error\n')
                source_file.write('none\n')
                target_file.write('none\n')
                generated_file.write('none\n')
                continue
        else:
            source_file.write('none\n')
            target_file.write('none\n')
            generated_file.write('none\n')
            print('Input no match found\n')
            continue
        
        output = generate(model, instruction)

        # 使用正则表达式提取内容
        match_generated = re.search(r'{(.*?)}', output[len('Below are molecule modifications:') + len(instruction):])
        result_generated = match_generated.group(1)
        if result_generated:
            print(result_generated)
            result_generated = decode_smiles.code_to_smiles(result_generated)
            if result_generated == None:
                print('decode2 error\n')
                source_file.write('none\n')
                target_file.write('none\n')
                generated_file.write('none\n')
                continue
            source_file.write(result_source  + '\n')
            target_file.write(result_target  + '\n')
            generated_file.write(result_generated  + '\n')
        else:
            source_file.write('none\n')
            target_file.write('none\n')
            generated_file.write('none\n')
            print('Generated no match found\n')
            continue