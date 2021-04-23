
import argparse
import os
import sentencepiece as sp
import contextual_mt
from fairseq import utils, hub_utils
from contextual_mt.utils import parse_documents, token_to_word_cxmi
from contextual_mt.docmt_cxmi import p_compute_cxmi
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="path to model checkpoint")
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--target-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    package = hub_utils.from_pretrained(
        args.path, checkpoint_file="checkpoint_best.pt"
    )
    models = package["models"]
    for model in models:
        model.cuda()
        model.eval()

    # load dict, params and generator from task
    src_dict = package["task"].src_dict
    tgt_dict = package["task"].tgt_dict

    # load sentencepiece models (assumes they are in the checkpoint dirs)
    # FIXME: is there someway to have it in `package`
    if os.path.exists(os.path.join(args.path, "spm.model")):
        spm = sp.SentencePieceProcessor()
        spm.Load(os.path.join(args.path, "spm.model"))
        src_spm = spm
        tgt_spm = spm
    else:
        src_spm = sp.SentencePieceProcessor()
        src_spm.Load(os.path.join(args.path, f"spm.{args.source_lang}.model"))
        tgt_spm = sp.SentencePieceProcessor()
        tgt_spm.Load(os.path.join(args.path, f"spm.{args.target_lang}.model"))

    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [int(idx) for idx in docids_f]
    with open(args.target_file, "r") as tgt_f:
        refs = [line.strip() for line in tgt_f]

    documents = parse_documents(srcs, refs, docids)

    token_cxmis, ids = p_compute_cxmi(
        documents,
        models,
        src_spm,
        src_dict,
        tgt_spm,
        tgt_dict,
        args.source_context_size,
        args.target_context_size,
        batch_size=8,
        word_level=True,
        output_ll=False
    )

    word_cxmis = token_to_word_cxmi(token_cxmis, documents, ids, tgt_spm)

    sorted_word_cxmis = [x for _,x in sorted(zip(ids,word_cxmis))]
    sorted_word_cxmis = [[sum(s).item() for s in score] for score in sorted_word_cxmis] 

    with open(args.output, "w") as file:
        for word_cxmi in sorted_word_cxmis:
            file.write(" ".join(list(map(str, word_cxmi))) + "\n")


if __name__ == "__main__":
    main()