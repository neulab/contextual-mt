from fire import Fire
from pathlib import Path
from collections import Counter
import os

def main(pretrain_dict_file, data_dir, dest_dir):
    data_dir = Path(data_dir)
    dest_dir = Path(dest_dir)
    
    pretrain_dict = []
    with open(pretrain_dict_file, 'r') as f:
        for l in f.read().split('\n'):
            if len(l):
                pretrain_dict.append(l.split(' ')[0])
    print('len pretrain_dict:', len(pretrain_dict))
    
    files = os.listdir(data_dir)
    vocab = []
    for file in files:
        if '.ipy' not in file:
            with open(data_dir / file, 'r') as f:
                vocab.extend(f.read().split())
    print('len vocab:', len(vocab))  
    
    vocab = set(vocab)
    print('len set vocab:', len(vocab))
    with open(dest_dir / 'dict.txt', 'w') as f:
        for word in vocab:
            print(f'{word} 1', file=f)
            
if __name__ == '__main__':
    Fire(main)