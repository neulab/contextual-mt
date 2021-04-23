import argparse
import os

import numpy as np
import torch
torch.manual_seed(0)

from fairseq import utils, hub_utils
from fairseq.sequence_scorer import SequenceScorer

import sentencepiece as sp

import contextual_mt  # noqa: F401
from contextual_mt.contextual_dataset import collate
from contextual_mt.utils import encode, create_context, parse_documents, decode_scores, decode, token_to_word_cxmi, SequenceGapScorer


def p_compute_cxmi(
    documents,
    models,
    src_spm,
    src_dict,
    tgt_spm,
    tgt_dict,
    source_context_size,
    target_context_size,
    batch_size,
    word_level=False,
    output_ll=False,
):
    ids = []
    src_context_lines = [[] for _ in range(batch_size)]
    tgt_context_lines = [[] for _ in range(batch_size)]

    scorer = SequenceScorer(tgt_dict)
    generator = SequenceGapScorer(tgt_dict)

    # info necessary to create batches and recreate docs
    doc_idx = 0
    current_docs = [None for _ in range(batch_size)]
    current_docs_ids = [-1 for _ in range(batch_size)]
    current_docs_pos = [0 for _ in range(batch_size)]
    baseline_xes = []
    contextual_xes = []
    preds = []
    while True:
        baseline_samples = []
        contextual_samples = []
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
            source_noeos = encode(src_l, src_spm, src_dict)
            source = torch.stack([*source_noeos, torch.tensor(src_dict.eos())])
            target_noeos = encode(tgt_l, tgt_spm, tgt_dict)
            target = torch.stack([*target_noeos, torch.tensor(tgt_dict.eos())])

            baseline_src_context = create_context(
                src_context_lines[idx],
                0,
                break_id=src_dict.index("<brk>"),
                eos_id=src_dict.eos(),
            )
            baseline_tgt_context = create_context(
                tgt_context_lines[idx],
                0,
                break_id=tgt_dict.index("<brk>"),
                eos_id=tgt_dict.eos(),
            )
            baseline_samples.append(
                {
                    "id": 0,
                    "src_context": baseline_src_context,
                    "source": source,
                    "tgt_context": baseline_tgt_context,
                    "target": target,
                }
            )

            contextual_src_context = create_context(
                src_context_lines[idx],
                source_context_size,
                break_id=src_dict.index("<brk>"),
                eos_id=src_dict.eos(),
            )
            contextual_tgt_context = create_context(
                tgt_context_lines[idx],
                target_context_size,
                break_id=tgt_dict.index("<brk>"),
                eos_id=tgt_dict.eos(),
            )
            contextual_samples.append(
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
        baseline_sample = collate(baseline_samples, src_dict.pad(), src_dict.eos())
        baseline_sample = utils.move_to_cuda(baseline_sample)
        contextual_sample = collate(contextual_samples, src_dict.pad(), src_dict.eos())
        contextual_sample = utils.move_to_cuda(contextual_sample)

        baseline_output = scorer.generate(models, baseline_sample)
        contextual_output = scorer.generate(models, contextual_sample)
        baseline_pred = generator.generate(models, baseline_sample)
        for batch_idx in range(len(baseline_samples)):
            # decode hypothesis
            key = "positional_scores" if word_level else "score"
            baseline_xes.append(baseline_output[batch_idx][0][key].cpu())
            contextual_xes.append(contextual_output[batch_idx][0][key].cpu())
            preds.append(baseline_pred[batch_idx][0]["pred"].cpu())

    cxmis = [c_xes - b_xes for c_xes, b_xes in zip(contextual_xes, baseline_xes)]
    #assert output_ll == word_level, "TODO"

    if output_ll:
        return cxmis, baseline_xes, contextual_xes, preds, ids
    else:
        return cxmis, ids



def compute_cxmi(
    documents,
    models,
    src_spm,
    src_dict,
    tgt_spm,
    tgt_dict,
    source_context_size,
    target_context_size,
    batch_size,
    word_level=False,
    token_level=False,
    norm_words=False,
    norm_cxmi=False,
    context_ll=False,
    base_ll=False
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
    baseline_xes = []
    contextual_xes = []
    while True:
        baseline_samples = []
        contextual_samples = []
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
            source_noeos = encode(src_l, src_spm, src_dict)
            source = torch.stack([*source_noeos, torch.tensor(src_dict.eos())])
            target_noeos = encode(tgt_l, tgt_spm, tgt_dict)
            target = torch.stack([*target_noeos, torch.tensor(tgt_dict.eos())])

            baseline_src_context = create_context(
                src_context_lines[idx],
                0,
                break_id=src_dict.index("<brk>"),
                eos_id=src_dict.eos(),
            )
            baseline_tgt_context = create_context(
                tgt_context_lines[idx],
                0,
                break_id=tgt_dict.index("<brk>"),
                eos_id=tgt_dict.eos(),
            )
            baseline_samples.append(
                {
                    "id": 0,
                    "src_context": baseline_src_context,
                    "source": source,
                    "tgt_context": baseline_tgt_context,
                    "target": target,
                }
            )

            contextual_src_context = create_context(
                src_context_lines[idx],
                source_context_size,
                break_id=src_dict.index("<brk>"),
                eos_id=src_dict.eos(),
            )
            contextual_tgt_context = create_context(
                tgt_context_lines[idx],
                target_context_size,
                break_id=tgt_dict.index("<brk>"),
                eos_id=tgt_dict.eos(),
            )
            contextual_samples.append(
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
        baseline_sample = collate(baseline_samples, src_dict.pad(), src_dict.eos())
        baseline_sample = utils.move_to_cuda(baseline_sample)
        contextual_sample = collate(contextual_samples, src_dict.pad(), src_dict.eos())
        contextual_sample = utils.move_to_cuda(contextual_sample)

        baseline_output = scorer.generate(models, baseline_sample)
        contextual_output = scorer.generate(models, contextual_sample)
        for batch_idx in range(len(baseline_samples)):
            # decode hypothesis
            # if word_level:
            #     assert len(baseline_samples[batch_idx]['target']) == len(baseline_output[batch_idx][0]["positional_scores"].cpu())
            #     baseline_xes.append(decode_scores(baseline_samples[batch_idx]['target'], tgt_spm, tgt_dict, baseline_output[batch_idx][0]["positional_scores"].cpu(), normalize=norm_words))
            #     assert len(contextual_samples[batch_idx]['target']) == len(contextual_output[batch_idx][0]["positional_scores"].cpu())
            #     contextual_xes.append(decode_scores(contextual_samples[batch_idx]['target'], tgt_spm, tgt_dict, contextual_output[batch_idx][0]["positional_scores"].cpu(), normalize=norm_words))
            if token_level or word_level:
                baseline_xes.append(-baseline_output[batch_idx][0]["positional_scores"].cpu())
                contextual_xes.append(-contextual_output[batch_idx][0]["positional_scores"].cpu())
            else:
                baseline_xes.append(-baseline_output[batch_idx][0]["score"].cpu())
                contextual_xes.append(-contextual_output[batch_idx][0]["score"].cpu())
    if word_level:
        #return [[c - b for c,b in zip(c_xes, b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids
        # if norm_cxmi:            
        #     return [[[(ci - bi)/bi for ci, bi in zip(c,b)] for c,b in zip(c_xes, b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids
        # else:
        #     return [[[ci - bi for ci, bi in zip(c,b)] for c,b in zip(c_xes, b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids
        if context_ll:
            token_scores = contextual_xes
        elif base_ll:
            token_scores = baseline_xes
        elif norm_cxmi:
            token_scores = [[(bi - ci)/bi for ci, bi in zip(c_xes,b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)]
        else:
            token_scores = [[(bi - ci) for ci, bi in zip(c_xes,b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)]
        return token_to_word_cxmi(token_scores, documents, ids, tgt_spm), ids
    if token_level:
        if context_ll:
            return contextual_xes, ids
        if base_ll:
            return baseline_xes, ids
        if norm_cxmi:
            return [[(bi - ci)/bi for ci, bi in zip(c_xes,b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids
        else:
            return [[(bi - ci) for ci, bi in zip(c_xes,b_xes)] for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids

    if norm_cxmi:
        return [(c_xes - b_xes)/b_xes for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids
    else:
        return [c_xes - b_xes for c_xes, b_xes in zip(contextual_xes, baseline_xes)], ids


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

    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=("number of sentences to inference in parallel"),
    )
    parser.add_argument(
        "--word-level",
        default=False,
        action="store_true",
        help="if set, calculates word-level CXMI",
    )
    args = parser.parse_args()

    # load files needed
    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.reference_file, "r") as tgt_f:
        refs = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [idx for idx in docids_f]

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
    sample_cxmis, ids = compute_cxmi(
        documents,
        models,
        src_spm,
        src_dict,
        tgt_spm,
        tgt_dict,
        source_context_size,
        target_context_size,
        batch_size=args.batch_size,
        word_level=args.word_level
    )
    print(f"CXMI: {np.mean(sample_cxmis):.05f}")


if __name__ == "__main__":
    main()
