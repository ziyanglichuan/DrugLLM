import json
import sys
import os

def add_input(data, count_increase, count_decrease, count_to_keep):
    new_data = []
    for d in data:
        if 'increase' in d['instruction'] and count_increase > count_to_keep:
            count_increase -= 1
        elif 'decrease' in d['instruction'] and count_decrease > count_to_keep:
            count_decrease -= 1
        else:
            instruction = d['instruction']
            if 'increase' in instruction:
                d['input'] = 'increase'
            elif 'decrease' in instruction:
                d['input'] = 'decrease'

            new_data.append(d)
    return new_data

def add_TwoInput(data, count_to_keep):
    new_data = []
    for d in data:
        if len(new_data) >= count_to_keep:
            break
        instruction = d['instruction'].split()
        d['input'] = ' '.join(word for word in instruction if word in ['increase', 'decrease'])
        new_data.append(d)
    return new_data

def change_instruction(data, offset, count_increase, count_decrease, count_to_keep):
    new_data = []
    for d in data:
        new_d = d.copy()
        if 'increase' in d['instruction'] and count_increase > count_to_keep:
            count_increase -= 1
        elif 'decrease' in d['instruction'] and count_decrease > count_to_keep:
            count_decrease -= 1
        else:
            instruction = d['instruction']
            output_index = instruction.find('Output:')
            if output_index != -1:
                output_str = d['output'].split('.')
                context = '.'.join(output_str[1:offset+1])
                if context:
                    context += '.'
                new_instruction = 'increase a certain molecular property. Output: ' + context + instruction[output_index+8:]
                if 'increase' in instruction:
                    new_d['input'] = 'increase'
                elif 'decrease' in instruction:
                    new_d['input'] = 'decrease'
                new_d['instruction'] = new_instruction
            new_data.append(new_d)
    return new_data

def add_offset(data, offset, count_increase, count_decrease, count_to_keep):
    new_data = []
    for d in data:
        new_d = d.copy()  # 创建一个新的字典
        if 'increase' in new_d['instruction'] and count_increase > count_to_keep:
            count_increase -= 1
        elif 'decrease' in new_d['instruction'] and count_decrease > count_to_keep:
            count_decrease -= 1
        else:
            instruction = new_d['instruction']
            output_index = instruction.find('Output:')
            if output_index != -1:
                output_str = new_d['output'].split('.')
                context = '.'.join(output_str[1:offset+1])
                if context:
                    context += '.'
                new_instruction = instruction[:output_index] + 'Output: '+context+instruction[output_index+8:]
                if 'increase' in instruction:
                    new_d['input'] = 'increase'
                elif 'decrease' in instruction:
                    new_d['input'] = 'decrease'
                new_d['instruction'] = new_instruction
            new_data.append(new_d)
    return new_data

def change_instruction_noBalance(data, offset):
    new_data = []
    for d in data:
        new_d = d.copy()
        instruction = d['instruction']
        output_index = instruction.find('Output:')
        if output_index != -1:
            output_str = d['output'].split('.')
            context = '.'.join(output_str[1:offset+1])
            if context:
                context += '.'
            new_instruction = 'increase a certain molecular property. Output: ' + context + instruction[output_index+8:]
            if 'increase' in instruction:
                new_d['input'] = 'increase'
            elif 'decrease' in instruction:
                new_d['input'] = 'decrease'
            new_d['instruction'] = new_instruction
        new_data.append(new_d)
    return new_data



input_filename = sys.argv[1]
count_increase = 0
count_decrease = 0

with open(input_filename, 'r') as f:
    data = json.load(f)

for item in data:
    if 'instruction' in item:
        if 'increase' in item['instruction']:
            count_increase += 1
        if 'decrease' in item['instruction']:
            count_decrease += 1

count_to_keep = min(count_increase, count_decrease)
count_to_keep = 400
print(f"Increase count: {count_increase}")
print(f"Decrease count: {count_decrease}")
print(f"Keep count: {count_to_keep}")

if sys.argv[2] == '0':
    new_data = add_input(data, count_increase, count_decrease, count_to_keep)
    input_basename = os.path.basename(input_filename)
    output_filename = 'changeInstru_data/change'  + sys.argv[2] + '_' + input_basename
    with open(output_filename, 'w') as f:
        json.dump(new_data, f, indent=4)

elif sys.argv[2] == '1':
    offset = 9
    new_data = add_offset(data, offset, count_increase, count_decrease, count_to_keep)
    input_basename = os.path.basename(input_filename)
    output_filename = 'changeInstru_data/change'  + sys.argv[2] + '_' + input_basename
    with open(output_filename, 'w') as f:
        json.dump(new_data, f, indent=4)

elif sys.argv[2] == '2':
    for offset in range(10):
        new_data = add_offset(data, offset, count_increase, count_decrease, count_to_keep)
        input_basename = os.path.basename(input_filename)
        output_filename = 'changeInstru_data/change'  + sys.argv[2] + '_offset' + str(offset) + '_' + input_basename
        with open(output_filename, 'w') as f:
            json.dump(new_data, f, indent=4)

# 无性质说明，多个文件
elif sys.argv[2] == '3':
    for offset in range(10):
        new_data = change_instruction(data, offset, count_increase, count_decrease, count_to_keep)
        input_basename = os.path.basename(input_filename)
        output_filename = 'changeInstru_data/change'  + sys.argv[2] + '_offset' + str(offset) + '_' + input_basename
        with open(output_filename, 'w') as f:
            json.dump(new_data, f, indent=4)

elif sys.argv[2] == '4':
    count_to_keep = 800
    new_data = add_TwoInput(data, count_to_keep)
    input_basename = os.path.basename(input_filename)
    output_filename = 'changeInstru_data/change'  + sys.argv[2] + '_' + input_basename
    with open(output_filename, 'w') as f:
        json.dump(new_data, f, indent=4)

# 无性质说明，单文件，无正负平衡
elif sys.argv[2] == '5':
    offset = 9
    new_data = change_instruction_noBalance(data, offset)
    input_basename = os.path.basename(input_filename)
    output_filename = 'changeInstru_data/change'  + sys.argv[2] + '_offset' + str(offset) + '_' + input_basename
    with open(output_filename, 'w') as f:
        json.dump(new_data, f, indent=4)