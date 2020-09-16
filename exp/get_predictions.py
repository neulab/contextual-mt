import os

for corpus in ['openSub_ende','openSub_enfr','openSub_enru','openSub_enes']:
    
    print(corpus)
    command = f"""
     cat data/inter/agnostic/pass/{corpus}/test.input \
     | CUDA_VISIBLE_DEVICES=1  fairseq-interactive data/preprocessed/{corpus}/ \
     --path checkpoint/{corpus}/checkpoint_best.pt \
     --buffer-size 2000 --batch-size 128 \
     --beam 5 --remove-bpe > hyp_{corpus}.sys
    """
    os.system(command)
    
    print('filtering')
    hip = []
    with open(f'hyp_{corpus}.sys', 'r') as f:
        for l in f.read().split('\n'):
            if l.count('\t') == 2:
                mod, n, sent = l.split('\t')
                if 'D-' in mod:
                    hip.append(sent)

    print('Saving')
    with open(f'hyp_prec_{corpus}.sys', 'w') as f:
        for l in hip:
            if len(l):
                print(l, file=f)
    print('-->')