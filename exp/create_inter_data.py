import os
import json
from pathlib import Path

DATASETS = ['wmtchat2020', 'openSub_ende', 'openSub_enes', 'openSub_enfr', 'openSub_enru']
PARTITIONS = ['train', 'dev', 'test']

def create_data(base_path='data', type_c='agnostic'):

    base_path = Path(base_path)
    raw_path = base_path / 'raw'
    inter_path = base_path / 'inter'
    for dataset in DATASETS:
        print(f'Processing {dataset} ...')
        if dataset == 'wmtchat2020':
            lang_src = 'en'
            lang_tgt = 'de'
            spk_b = 'agent'
            spk_a = 'customer'
        elif 'openSub' in dataset:
            pair_lang = dataset.split('_')[1]
            lang_src = pair_lang[:2]
            lang_tgt = pair_lang[2:]
            spk_b = '<en>'
            spk_a = '<2en>'
        else:
            raise NotImplementedError
            
        for partition in PARTITIONS:
            print(f'Reading {partition} ...')
            with open(raw_path / dataset / f'{partition}.json') as f:
                data = json.load(f)
                
            data_src = [] # Always english
            data_tgt = [] 
            for conversation in data:
                for block in data[conversation]:
                    if type_c == 'agnostic' or partition=='test':
                        if block['speaker'] == spk_a:
                            data_src.append(block['source'])
                            data_tgt.append(block['target'])
                        elif block['speaker'] == spk_b:
                            data_src.append(block['source'])
                            data_tgt.append(block['target'])
                        else:
                            raise NotImplementedError
                    elif type_c == 'multilingual':
                        if block['speaker'] == spk_a:
                            data_src.append(block['source'])
                            data_tgt.append(block['target'])
                        elif block['speaker'] == spk_b:
                            data_src.append(block['target'])
                            data_tgt.append(block['source'])
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                  
            os.makedirs(inter_path / type_c / dataset, exist_ok=True)
            
            if type_c == 'agnostic':
                with open(inter_path / type_c / dataset / f'{partition}.input', 'w') as f:
                    for line in data_src:
                        if len(line):
                            print(line, file=f)

                with open(inter_path / type_c / dataset / f'{partition}.output', 'w') as f:
                    for line in data_tgt:
                        if len(line):
                            print(line, file=f)
            elif type_c == 'multilingual':
                list_langs = [[lang_src, lang_tgt, data_src, data_tgt], [lang_tgt, lang_src, data_tgt, data_src]]
                    
                for lang_1, lang_2, temp_1, temp_2 in list_langs:
                    with open(inter_path / type_c / dataset / f'{partition}.{lang_1}-{lang_2}.{lang_1}', 'w') as f:
                        for line in temp_1:
                            if len(line):
                                print(line, file=f)

                    with open(inter_path / type_c / dataset / f'{partition}.{lang_1}-{lang_2}.{lang_2}', 'w') as f:
                        for line in temp_2:
                            if len(line):
                                print(line, file=f)
            else:
                print(type_c)
                raise NotImplementedError
                            
                            
        with open(raw_path / dataset / f'{partition}.json') as f:
            data = json.load(f)

def main():
    create_data(base_path='data', type_c='agnostic')
    create_data(base_path='data', type_c='multilingual')
            
main()
