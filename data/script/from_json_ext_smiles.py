import json
import multiprocessing

def process_item(item, output_file, lock):
    # ��ȡoutput����Ӧ��ֵ�е�SMILES�ַ���
    smiles_list = item['output'].split('}')
    smiles_list = [smiles.strip('-.>{') for smiles in smiles_list if smiles.strip()]
    
    # ��ȡinstruction����Ӧ��ֵ�е�SMILES�ַ���
    if 'Output:' in item['instruction']:
        smiles = item['instruction'].split('Output: ')[1].strip('{}')
        smiles_list.append(smiles)

    # ʹ������ͬ��д�����
    with lock:
        with open(output_file, 'a') as f:
            for smiles in smiles_list:
                f.write(smiles + '\n')

if __name__ == '__main__':
    with open('ext_data/temp_test.json', 'r') as f:
        data = json.load(f)

    output_file = 'ext_data/temp_smiles.txt'

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        pool = multiprocessing.Pool()
        pool.starmap(process_item, [(item, output_file, lock) for item in data])

        pool.close()
        pool.join()
