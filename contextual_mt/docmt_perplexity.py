import argparse
import os
import random

import numpy as np
import torch

from fairseq import utils, hub_utils
from fairseq.sequence_scorer import SequenceScorer

import sentencepiece as sp

import contextual_mt  # noqa: F401
from contextual_mt.contextual_dataset import collate
from contextual_mt.utils import encode, create_context, parse_documents


def compute_crossentropy(
    documents,
    models,
    src_spm,
    src_dict,
    tgt_spm,
    tgt_dict,
    source_context_size,
    target_context_size,
    batch_size,
    no_source=False
):
    ids = []
    src_context_lines = [[] for _ in range(batch_size)]
    tgt_context_lines = [[] for _ in range(batch_size)]

    scorer = SequenceScorer(tgt_dict)

    # info necessary to create batches and recreate docs
    doc_idx = 0
    current_docs = [None for _ in range(batch_size)]
    current_docs_ids = [-1 for _ in range(batch_size)]
    current_docs_pos = [0 for _ in range(batch_size)]
    xes = []
    while True:
        samples = []
        for idx in range(batch_size):
            # if any of the docs in the batch has finished replace by a new one
            if current_docs[idx] is None or current_docs_pos[idx] >= len(
                current_docs[idx]
            ):
                if doc_idx < len(documents):
                    current_docs[idx] = documents[doc_idx]
                    current_docs_ids[idx] = doc_idx
                    current_docs_pos[idx] = 0
                    src_context_lines[idx] = []
                    tgt_context_lines[idx] = []
                    doc_idx += 1
                else:
                    current_docs[idx] = None
                    continue

            src_l, tgt_l = current_docs[idx][current_docs_pos[idx]]

            ids.append((current_docs_ids[idx], current_docs_pos[idx]))

            # binarize source and create input with context and target
            source_noeos = encode(src_l, src_spm, src_dict) if not no_source else torch.tensor([src_dict.index("<mask>")])
            source = torch.stack([*source_noeos, torch.tensor(src_dict.eos())])
            target_noeos = encode(tgt_l, tgt_spm, tgt_dict)
            target = torch.stack([*target_noeos, torch.tensor(tgt_dict.eos())])

            src_context = src_context_lines[idx]
            tgt_context = tgt_context_lines[idx]

            contextual_src_context = create_context(
                src_context,
                source_context_size,
                break_id=src_dict.index("<brk>"),
                eos_id=src_dict.eos(),
            )
            contextual_tgt_context = create_context(
                tgt_context,
                target_context_size,
                break_id=tgt_dict.index("<brk>"),
                eos_id=tgt_dict.eos(),
            )
            samples.append(
                {
                    "id": 0,
                    "src_context": contextual_src_context,
                    "source": source,
                    "tgt_context": contextual_tgt_context,
                    "target": target,
                }
            )

            src_context_lines[idx].append(source_noeos)
            tgt_context_lines[idx].append(target_noeos)

            current_docs_pos[idx] += 1

        # while exit condition
        if all(chat is None for chat in current_docs):
            break

        # create batch
        sample = collate(samples, src_dict.pad(), src_dict.eos())
        sample = utils.move_to_cuda(sample)

        contextual_output = scorer.generate(models, sample)
        for batch_idx in range(len(samples)):
            # decode hypothesis
            key = "score"
            xes.append(contextual_output[batch_idx][0][key].cpu())

        return xes, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument(
        "--reference-file",
        required=True,
        help="file with reference translations",
    )
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument(
        "--path", required=True, metavar="FILE", help="path to model file"
    )

    parser.add_argument("--source-context-size", default=None, type=int)
    parser.add_argument("--target-context-size", default=None, type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=("number of sentences to inference in parallel"),
    )
    parser.add_argument(
        "--no-source",
        action="store_true",
        help=""
    )
    args = parser.parse_args()

    # load files needed
    with open(args.source_file, "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.reference_file, "r", encoding="utf-8") as tgt_f:
        refs = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r", encoding="utf-8") as docids_f:
        docids = [int(idx) for idx in docids_f]

    pretrained = hub_utils.from_pretrained(
        args.path, checkpoint_file="checkpoint_best.pt"
    )
    models = pretrained["models"]
    for model in models:
        model.cuda()
        model.eval()

    # load dict, params and generator from task
    src_dict = pretrained["task"].src_dict
    tgt_dict = pretrained["task"].tgt_dict
    source_context_size = (
        pretrained["task"].args.source_context_size
        if args.source_context_size is None
        else args.source_context_size
    )
    target_context_size = (
        pretrained["task"].args.target_context_size
        if args.target_context_size is None
        else args.target_context_size
    )

    # load sentencepiece models (assume they are in the checkpoint dirs)
    if os.path.exists(os.path.join(args.path, "spm.model")):
        spm = sp.SentencePieceProcessor()
        spm.Load(os.path.join(args.path, "spm.model"))
        src_spm = spm
        tgt_spm = spm
    else:
        assert args.source_lang is not None and args.target_lang is not None
        src_spm = sp.SentencePieceProcessor()
        src_spm.Load(os.path.join(args.path, f"spm.{args.source_lang}.model"))
        tgt_spm = sp.SentencePieceProcessor()
        tgt_spm.Load(os.path.join(args.path, f"spm.{args.target_lang}.model"))

    documents = parse_documents(srcs, refs, docids)
    sample_xes, _ = compute_crossentropy(
        documents,
        models,
        src_spm,
        src_dict,
        tgt_spm,
        tgt_dict,
        source_context_size,
        target_context_size,
        batch_size=args.batch_size,
        no_source=args.no_source,
    )
    print(f"Perplexity: {np.exp(-np.mean(sample_xes)):.03f}")


if __name__ == "__main__":
    main()
