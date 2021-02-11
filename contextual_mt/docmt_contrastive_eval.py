import argparse
import os
import json
import tqdm

import torch

from fairseq import utils, hub_utils
from fairseq.sequence_scorer import SequenceScorer

import sentencepiece as sp

import contextual_mt  # noqa: F401
from contextual_mt import DocumentTranslationTask
from contextual_mt.contextual_dataset import collate as contextual_collate
from fairseq.data.language_pair_dataset import collate as raw_collate
from contextual_mt.utils import create_context, encode


def load_contrastive(
    source_file,
    target_file,
    src_context_file,
    tgt_context_file,
    dataset="contrapro",
):
    """
    reads the contrapro or bawden contrastive dataset
    """
    if dataset == "contrapro":
        pronouns = ("Er", "Sie", "Es")
    elif dataset == "bawden":
        pronouns = None
    else:
        raise ValueError("should not get here")

    # load files needed
    # and binarize
    with open(source_file, "r") as src_f, open(target_file, "r") as tgt_f, open(
        src_context_file
    ) as src_ctx_f, open(tgt_context_file) as tgt_ctx_f:
        srcs = []
        srcs_context = []
        tgts_context = []
        all_tgts = []
        tgt_labels = []
        src_lines = src_f.readlines()
        src_ctx_lines = src_ctx_f.readlines()
        tgt_lines = tgt_f.readlines()
        tgt_ctx_lines = tgt_ctx_f.readlines()
        assert len(src_lines) == len(
            tgt_lines
        ), "source and target files have different sizes"
        assert len(src_ctx_lines) == len(
            tgt_ctx_lines
        ), "src_content and tgt_context files have different_sizes"
        assert (
            len(src_ctx_lines) % len(src_lines) == 0
        ), "src_context file lines aren't multiple of source lines"
        included_context_size = len(src_ctx_lines) // len(src_lines)

        index = 0
        while index < len(src_lines):
            i = 0
            src = None
            tgts = []
            while (index + i) < len(src_lines) and (
                (
                    dataset == "contrapro"
                    and (src is None or src == src_lines[index + i])
                )
                or (dataset == "bawden" and (i < 2))
            ):
                src = src_lines[index + i]
                tgt = tgt_lines[index + i]
                src_context = [
                    src_ctx_lines[(index + i) * included_context_size + j].strip()
                    for j in range(included_context_size)
                ]
                tgt_context = [
                    tgt_ctx_lines[(index + i) * included_context_size + j].strip()
                    for j in range(included_context_size)
                ]
                tgts.append(tgt.strip())
                i += 1

            lower_gold = tgts[0].lower()
            tokenized_gold = lower_gold.split(" ")
            # if for some reason simple tokenization
            # doesn't work, just count the pron that
            # appears more times
            max_count, best_pron = 0, None
            if pronouns is not None:
                for pron in pronouns:
                    if pron.lower() in tokenized_gold:
                        best_pron = pron
                        max_count = 1
                        break
                    count = lower_gold.count(pron.lower())
                    if count > max_count:
                        best_pron = pron
                        max_count = count
                if max_count == 0:
                    raise ValueError(
                        f"no pronoun found in one of the sentences: {tgts[0]}"
                    )
            else:
                best_pron = None

            tgt_labels.append(best_pron)

            srcs.append(src.strip())
            all_tgts.append(tgts)
            srcs_context.append(src_context)
            tgts_context.append(tgt_context)
            index += i

    assert len([t for tgt in all_tgts for t in tgt]) == len(
        tgt_lines
    ), "ended up with differnt number of lines..."
    return srcs, all_tgts, tgt_labels, srcs_context, tgts_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--src-context-file", required=True)
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--tgt-context-file", required=True)
    parser.add_argument("--source-context-size", default=0, type=int)
    parser.add_argument("--target-context-size", default=0, type=int)
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument(
        "--dataset", choices=("contrapro", "bawden"), default="contrapro"
    )
    parser.add_argument(
        "--path", required=True, metavar="FILE", help="path to model file"
    )
    parser.add_argument("--checkpoint-file", default="checkpoint_best.pt")
    parser.add_argument("--save-scores", default=None, type=str)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=("number of sentences to inference in parallel"),
    )
    args = parser.parse_args()

    # load pretrained model, set eval and send to cuda
    pretrained = hub_utils.from_pretrained(
        args.path, checkpoint_file=args.checkpoint_file
    )
    models = pretrained["models"]
    for model in models:
        model.cuda()
        model.eval()

    # load dict, params and generator from task
    src_dict = pretrained["task"].src_dict
    tgt_dict = pretrained["task"].tgt_dict
    if isinstance(pretrained["task"], DocumentTranslationTask):
        concat_model = False
        source_context_size = pretrained["task"].args.source_context_size
        target_context_size = pretrained["task"].args.target_context_size
    else:
        concat_model = True
        source_context_size = args.source_context_size
        target_context_size = args.target_context_size

    scorer = SequenceScorer(tgt_dict)

    # load sentencepiece models (assume they are in the checkpoint dirs)
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

    # load files
    srcs, all_tgts, tgt_labels, srcs_contexts, tgts_contexts = load_contrastive(
        args.source_file,
        args.target_file,
        args.src_context_file,
        args.tgt_context_file,
        dataset=args.dataset,
    )
    # and binarize
    srcs = [encode(s, src_spm, src_dict) for s in srcs]
    all_tgts = [[encode(s, tgt_spm, tgt_dict) for s in tgts] for tgts in all_tgts]
    srcs_context = [
        [encode(s, src_spm, src_dict) for s in context] for context in srcs_contexts
    ]
    tgts_context = [
        [encode(s, tgt_spm, tgt_dict) for s in context] for context in tgts_contexts
    ]

    label_corrects = {label: [] for label in set(tgt_labels)}
    bar = tqdm.tqdm(total=sum(1 for _ in srcs))
    corrects = []
    attentions, src_log, src_context_log, tgt_context_log, tgt_log = [], [], [], [], []
    all_scores = []
    for src, src_ctx, contr_tgts, tgt_ctx, label in zip(
        srcs, srcs_context, all_tgts, tgts_context, tgt_labels
    ):
        samples = []
        for tgt in contr_tgts:
            if concat_model:
                src_ctx_tensor = create_context(
                    src_ctx, source_context_size, src_dict.index("<brk>")
                )
                if len(src_ctx_tensor) > 0:
                    src_ctx_tensor = torch.cat(
                        [src_ctx_tensor, torch.tensor([src_dict.index("<brk>")])]
                    )
                full_src = torch.cat(
                    [src_ctx_tensor, src, torch.tensor([src_dict.eos()])]
                )
                tgt_ctx_tensor = create_context(
                    tgt_ctx, target_context_size, tgt_dict.index("<brk>")
                )
                if len(tgt_ctx_tensor) > 0:
                    tgt_ctx_tensor = torch.cat(
                        [tgt_ctx_tensor, torch.tensor([tgt_dict.index("<brk>")])]
                    )
                full_tgt = torch.cat(
                    [tgt_ctx_tensor, tgt, torch.tensor([tgt_dict.eos()])]
                )
                sample = {"id": 0, "source": full_src, "target": full_tgt}
            else:
                src_ctx_tensor = create_context(
                    src_ctx,
                    source_context_size,
                    src_dict.index("<brk>"),
                    src_dict.eos(),
                )

                full_src = torch.cat([src, torch.tensor([src_dict.eos()])])
                tgt_ctx_tensor = create_context(
                    tgt_ctx,
                    target_context_size,
                    tgt_dict.index("<brk>"),
                    tgt_dict.eos(),
                )
                full_tgt = torch.cat([tgt, torch.tensor([tgt_dict.eos()])])
                sample = {
                    "id": 0,
                    "source": full_src,
                    "src_context": src_ctx_tensor,
                    "target": full_tgt,
                    "tgt_context": tgt_ctx_tensor,
                }
            samples.append(sample)

        if concat_model:
            sample = raw_collate(
                samples, pad_idx=src_dict.pad(), eos_idx=src_dict.eos()
            )
        else:
            sample = contextual_collate(
                samples,
                pad_id=src_dict.pad(),
                eos_id=src_dict.eos(),
            )

        sample = utils.move_to_cuda(sample)
        hyps = scorer.generate(models, sample)
        scores = [h[0]["score"] for h in hyps]
        all_scores = all_scores + scores

        most_likely = torch.argmax(torch.stack(scores))
        correct = most_likely == 0
        corrects.append(correct)
        label_corrects[label].append(correct)

        # save info for attention visualization
        attentions.append(hyps[most_likely][0]["attention"])
        src_log.append(src_dict.string(samples[most_likely]["source"]) + " <eos>")
        src_context_log.append(
            src_dict.string(samples[most_likely]["src_context"]) + " <eos>"
        )
        tgt_context_log.append(
            "<eos> " + tgt_dict.string(samples[most_likely]["tgt_context"])
        )
        tgt_log.append("<eos> " + tgt_dict.string(samples[most_likely]["target"]))

        bar.update(1)

    bar.close()

    if None not in label_corrects:
        print("Pronoun accs...")
        for label, l_corrects in label_corrects.items():
            print(f" {label}: {torch.stack(l_corrects).float().mean().item()}")
    print(f"Total Acc: {torch.stack(corrects).float().mean().item()}")

    print("Saving info")
    with open("log.json", "w") as f:
        for src, src_context, tgt, tgt_context, attention, correct in zip(
            src_log, src_context_log, tgt_log, tgt_context_log, attentions, corrects
        ):
            d = json.dumps(
                {
                    "correct": correct.item(),
                    "source": src,
                    "source_context": src_context,
                    "target": tgt,
                    "target_context": tgt_context,
                    "attention": attention.tolist(),
                }
            )
            print(d, file=f)

    if args.save_scores is not None:
        with open(args.save_scores, "w") as f:
            for score in all_scores:
                print(score.item(), file=f)


if __name__ == "__main__":
    main()
