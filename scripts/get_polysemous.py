import re
import abc
import argparse
import spacy
import spacy_stanza
import json
from collections import defaultdict

def normalize(word):
    return re.sub(r"^\W+|\W+$", "", word.lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--target-file", required=True, help="file to be translated")
    parser.add_argument("--alignments-file", required=True, help="file with word alignments")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    src_aligned_tgt = defaultdict(lambda: defaultdict(lambda: 0))

    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.target_file, "r") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    # with open(args.source_file.replace("tok", "detok"), "r") as src_f:
    #     detok_srcs = [line.strip() for line in src_f]
    # with open(args.target_file.replace("tok", "detok"), "r") as tgt_f:
    #     detok_tgts = [line.strip() for line in tgt_f]
    with open(args.alignments_file, "r") as file:
        alignments = file.readlines()

    alignments = list(map(lambda x: dict(list(map(lambda y: list(map(int,y.split("-"))), x.strip().split(" ")))), alignments))
    src_tagger = spacy_stanza.load_pipeline(args.source_lang, processors="tokenize,pos,lemma")
    tgt_tagger = spacy_stanza.load_pipeline(args.target_lang.split("_")[0], processors="tokenize,pos,lemma")

    for src, tgt, align in zip(srcs, tgts, alignments):
        src_doc = src_tagger(src)
        tgt_doc = tgt_tagger(tgt)
        src_lemmas = [tok.lemma_ if not tok.is_stop and not tok.is_punct else None for tok in src_doc]
        tgt_lemmas = [tok.lemma_ if not tok.is_stop and not tok.is_punct else None for tok in tgt_doc]

        for s, t in align.items():
            try:
                src_lemma = src_lemmas[s]
                tgt_lemma = tgt_lemmas[t]
                if src_lemma is not None and tgt_lemma is not None:
                    src_aligned_tgt[src_lemma][tgt_lemma] += 1
            except:
                continue

    poly_lemmas = [src for src, tgts in src_aligned_tgt.items() if len(tgts) > 1]

    with open(args.output, "w") as file:
        for lemma in poly_lemmas:
            file.write(f"{lemma}\n")
    with open(args.output + ".json", "w") as file:
        json.dump(src_aligned_tgt, file)


            










if __name__ == "__main__":
    main()
