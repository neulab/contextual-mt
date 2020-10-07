"""
Calculate the ratio of number of tokens in two sets of sentences
"""
import argparse, csv
from mosestokenizer import *
import sacrebleu
import comet
from comet.models import download_model
from bleurt import score

if __name__ == "__main__":

    data_dir = "/projects/tir4/users/kayoy/dialogue-translation/dialogue-mt/outputs/"

    checkpoint = "/projects/tir4/users/kayoy/dialogue-translation/bleurt/bleurt-base-128"
    #bleurtscorer = score.BleurtScorer(checkpoint)
    comet_model = download_model("wmt-large-da-estimator-1719", "/projects/tir4/users/kayoy/dialogue-translation/comet/")
    
    tokenize = MosesTokenizer('en')
    
    results = [['Experiment', 'Length ratio (hyp/target)', 'BLEU', 'BLEURT', 'COMET']]
    for data_file in ['enfr00_3', 'enfr21_3', 'enfradd_4', 'enfrencode_3', 'enfr00_4', 'enfr21_4', 'enfradd_5', 'enfradd2', 'enfrencode_4', 'enfr00_7', 'enfr21_6', 'enfr22_5', 'enfradd_6', 'enfrencode_6']:
        result = [data_file]
        with open(data_dir + data_file + '.txt', 'r') as file:
            lines = file.readlines()

        src = []
        tgt = []
        hyp = []
        for line in lines:
            if line.startswith("S-"):
                src.append(line)
            elif line.startswith("T-"):
                tgt.append(" ".join(line.split()[1:]) + "\n")
            elif line.startswith("D-"):
                hyp.append(" ".join(line.split()[2:]) + "\n")


        print(len(src))
        print(len(tgt))
        print(len(hyp))

        ratios = [len(h)/len(t) for h,t in zip(map(tokenize, hyp), map(tokenize, tgt))]
        
        try:
            result.append(sum(ratios)/len(ratios))
        except:
            result.append(0)

        bleu = sacrebleu.corpus_bleu(hyp, [tgt]).score
        result.append(bleu)

        bleurt = bleurtscorer.score(tgt, hyp)[0] * 100
        result.append(bleurt)

        comet_input = {"src": src, "mt": hyp, "ref": tgt}
        comet_input = [dict(zip(comet_input, t)) for t in zip(*comet_input.values())]
        out = comet_model.predict(comet_input, cuda=True, show_progress=False)
        print(out)
        _, comet = out
        try:
            comet = sum(comet)/len(comet)
        except:
            comet = 0
        print(comet)
        result.append(comet)
        

        results.append(result)

    with open("scores3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)
