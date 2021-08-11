import argparse
import os
import tqdm

import torch

from fairseq import utils, hub_utils

import sentencepiece as sp

import contextual_mt  # noqa: F401
from contextual_mt.contextual_dataset import collate
from contextual_mt import ContextualSequenceGenerator

from contextual_mt.utils import encode, decode, create_context, parse_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument(
        "--predictions-file", required=True, help="file to save the predictions"
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="reference file, used if with --gold-target-context",
    )
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument(
        "--path", required=True, metavar="FILE", help="path to model file"
    )
    parser.add_argument("--beam", default=5, type=int, metavar="N", help="beam size")
    parser.add_argument(
        "--max-len-a",
        default=0,
        type=float,
        metavar="N",
        help=(
            "generate sequences of maximum length ax + b, "
            "where x is the source length"
        ),
    )
    parser.add_argument(
        "--max-len-b",
        default=200,
        type=int,
        metavar="N",
        help=(
            "generate sequences of maximum length ax + b, "
            "where x is the source length"
        ),
    )
    parser.add_argument(
        "--min-len",
        default=1,
        type=float,
        metavar="N",
        help=("minimum generation length"),
    )
    parser.add_argument(
        "--lenpen",
        default=1,
        type=float,
        help="length penalty: <1.0 favors shorter, >1.0 favors longer sentences",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=("number of sentences to inference in parallel"),
    )
    parser.add_argument(
        "--gold-target-context",
        default=False,
        action="store_true",
        help="if set, model will use ground-truth targets as context",
    )
    parser.add_argument("--source-context-size", default=None, type=int)
    parser.add_argument("--target-context-size", default=None, type=int)

    args = parser.parse_args()

    if args.gold_target_context:
        assert args.reference_file is not None

    # load pretrained model, set eval and send to cuda
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
    generator = pretrained["task"].build_generator(
        models, args, seq_gen_cls=ContextualSequenceGenerator
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

    # load files needed
    with open(args.source_file, "r", encoding='utf-8') as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.docids_file, "r", encoding='utf-8') as docids_f:
        docids = [int(idx) for idx in docids_f]
    if args.reference_file is not None:
        with open(args.reference_file, "r", encoding='utf-8') as tgt_f:
            refs = [line.strip() for line in tgt_f]
    else:
        refs = [None for _ in srcs]

    documents = parse_documents(srcs, refs, docids)

    preds = []
    ids = []
    scores = []
    bar = tqdm.tqdm(total=sum(1 for _ in srcs))
    src_context_lines = [[] for _ in range(args.batch_size)]
    tgt_context_lines = [[] for _ in range(args.batch_size)]

    # info necessary to create batches and recreate docs
    doc_idx = 0
    current_docs = [None for _ in range(args.batch_size)]
    current_docs_ids = [-1 for _ in range(args.batch_size)]
    current_docs_pos = [0 for _ in range(args.batch_size)]
    while True:
        batch_map = []
        batch_targets = []
        samples = []
        for idx in range(args.batch_size):
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

            # this is need to be able to remap to
            # the correct objects if true batch size < batch_size
            # and in order to save the correct target context
            batch_map.append(idx)
            if args.reference_file is not None:
                batch_targets.append(encode(tgt_l, tgt_spm, tgt_dict))

            ids.append((current_docs_ids[idx], current_docs_pos[idx]))

            # binarize source and create input with context and target
            source_noeos = encode(src_l, src_spm, src_dict)
            source = torch.stack([*source_noeos, torch.tensor(src_dict.eos())])
            src_context = create_context(
                src_context_lines[idx],
                source_context_size,
                break_id=src_dict.index("<brk>"),
                eos_id=src_dict.eos(),
            )
            tgt_context = create_context(
                tgt_context_lines[idx],
                target_context_size,
                break_id=tgt_dict.index("<brk>"),
                eos_id=tgt_dict.eos(),
            )

            samples.append(
                {
                    "id": 0,
                    "src_context": src_context,
                    "source": source,
                    "tgt_context": tgt_context,
                }
            )

            src_context_lines[idx].append(source_noeos)

            current_docs_pos[idx] += 1

        # while exit condition
        if all(chat is None for chat in current_docs):
            break

        # create batch
        sample = collate(samples, src_dict.pad(), src_dict.eos())
        sample = utils.move_to_cuda(sample)
        output = pretrained["task"].inference_step(generator, models, sample)
        for batch_idx in range(len(samples)):
            # decode hypothesis
            hyp_ids = output[batch_idx][0]["tokens"].cpu()
            preds.append(decode(hyp_ids, tgt_spm, tgt_dict))
            scores.append(output[batch_idx][0]["positional_scores"].cpu().tolist())

            # collect output to be prefix for next utterance
            idx = batch_map[batch_idx]
            if args.gold_target_context:
                tgt_context_lines[idx].append(batch_targets[batch_idx])
            else:
                tgt_context_lines[idx].append(
                    hyp_ids[:-1] if hyp_ids[-1] == tgt_dict.eos() else hyp_ids
                )

        bar.update(len(samples))

    bar.close()

    assert len(preds) == len(ids)
    _, preds = zip(*sorted(zip(ids, preds)))

    with open(args.predictions_file, "w", encoding='utf-8') as f:
        for pred in preds:
            print(pred, file=f)


if __name__ == "__main__":
    main()
