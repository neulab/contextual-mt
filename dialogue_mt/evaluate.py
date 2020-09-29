import argparse
import os
import json
import tqdm

import torch

from fairseq import utils, hub_utils
from fairseq.data import data_utils

import sacrebleu
from comet.models import download_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        help="colon separated path to data directories list, \
                    will be iterated upon during epochs in round-robin manner; \
                    however, valid and test data are always in the first directory to \
                    avoid the need for repeating them in all directories",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="split do inference on"
    )
    parser.add_argument("--path", metavar="FILE", help="path to model file")
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
        default=8,
        help=("number of chats to inference in parallel"),
    )
    parser.add_argument(
        "--ignore-previous-targets",
        default=False,
        action="store_true",
        help="if set, model will ignore previously generated targets and re-generate a new context",
    )
    parser.add_argument(
        "--comet-model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--comet-path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--print-output",
        type=str,
        default=None,
        help="if set, saves the outpus to a file",
    )

    args = parser.parse_args()

    # load pretrained model, set eval and send to cuda
    pretrained = hub_utils.from_pretrained(
        args.path, checkpoint_file="checkpoint_best.pt"
    )
    models = pretrained["models"]
    for model in models:
        model.cuda()
        model.eval()

    bpe = pretrained["task"].bpe
    vocab = pretrained["task"].src_dict
    tokenizer = pretrained["task"].tokenizer
    split_source_context = pretrained["task"].args.split_source_context
    source_context_size = pretrained["task"].args.source_context_size
    target_context_size = pretrained["task"].args.target_context_size
    generator = pretrained["task"].build_generator(models, args)

    data_path = os.path.join(args.data, f"{args.split}.json")

    def encode(s):
        """ applies tokenization and bpe """
        if tokenizer is not None:
            s = tokenizer.encode(s)
        if bpe is not None:
            s = bpe.encode(s)
        return s

    def decode(x):
        """ removes bpe and detokenizes """
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    def binarize(s, speaker):
        """ binarizes a sentence by applying bpe and tokenization and adding a speaker tag """
        s = encode(s)
        tokens = vocab.encode_line(s, append_eos=False, add_if_not_exist=False).long()
        spk_tensor = torch.tensor([vocab.index(speaker)])
        tokens = torch.cat([spk_tensor, tokens])
        return tokens

    with open(data_path, "r") as f:
        chat_dict = json.load(f)

    chat_list = list(chat_dict.values())
    srcs = []
    refs = []
    preds = []
    bar = tqdm.tqdm(total=sum(1 for chat in chat_list for turn in chat))
    src_context = [[] for _ in range(args.batch_size)]
    tgt_context = [[] for _ in range(args.batch_size)]
    current_chats = [[] for _ in range(args.batch_size)]
    while True:
        src_ctx_tokens = []
        src_ctx_lengths = []
        src_tokens = []
        src_lengths = []
        targets = []
        for idx in range(args.batch_size):
            # if any of the chats in the batch has finished replace by a new one
            if not current_chats[idx]:
                if chat_list:
                    current_chats[idx] = list(chat_list.pop(0))
                    src_context[idx] = []
                    tgt_context[idx] = []
                else:
                    current_chats[idx] = None
                    continue

            turn = current_chats[idx].pop(0)

            # normalize references to match the way `fairseq-generate` computes scores
            srcs.append(turn["source"])
            refs.append(decode(encode(turn["target"])))

            # binarize source and create input with context and target
            src_ids = binarize(turn["source"], speaker=turn["speaker"])
            src_ctx_ids = [
                i
                for ctx in src_context[idx][
                    len(src_context[idx]) - source_context_size :
                ]
                for i in [*ctx, torch.tensor(vocab.index("<brk>"))]
            ]
            tgt_ctx_ids = [
                i
                for ctx in tgt_context[idx][
                    len(tgt_context[idx]) - target_context_size :
                ]
                for i in [*ctx, torch.tensor(vocab.index("<brk>"))]
            ]

            # if context separate from source, encode in different tensors (removing the last break)
            if split_source_context:
                src_ctx_tokens.append(
                    torch.stack([*src_ctx_ids[:-1], torch.tensor(vocab.eos())])
                )
                src_ctx_lengths.append(len(src_ctx_tokens[-1]))
                src_tokens.append(torch.stack([*src_ids, torch.tensor(vocab.eos())]))
                src_lengths.append(len(src_tokens[-1]))
            # otherwise just concat with source
            else:
                src_tokens.append(
                    torch.stack([*src_ctx_ids, *src_ids, torch.tensor(vocab.eos())])
                )
                src_lengths.append(len(src_tokens[-1]))

            # TODO: add split_target_context
            targets.append(
                torch.stack(tgt_ctx_ids) if tgt_ctx_ids else torch.tensor([])
            )
            src_context[idx].append(src_ids)

        # while exit condition
        if all(chat is None for chat in current_chats):
            break
        # create batch
        src_tokens = data_utils.collate_tokens(src_tokens, vocab.pad(), vocab.eos())
        src_lengths = torch.tensor(src_lengths)
        targets = data_utils.collate_tokens(targets, vocab.pad(), vocab.eos())

        # build batch and run inference
        sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths},
            "target": targets.long() if not args.ignore_previous_targets else None,
        }
        if split_source_context:
            src_ctx_tokens = data_utils.collate_tokens(
                src_ctx_tokens, vocab.pad(), vocab.eos()
            ).long()
            src_ctx_lengths = torch.tensor(src_ctx_lengths)
            sample["net_input"].update(
                {"src_context": src_ctx_tokens, "src_ctx_lengths": src_ctx_lengths}
            )

        sample = utils.move_to_cuda(sample)
        output = pretrained["task"].inference_step(generator, models, sample)
        for idx in range(len(src_lengths)):
            # decode hypothesis
            hyp_ids = output[idx][0]["tokens"].cpu()
            hyp_str = vocab.string(hyp_ids.int())
            preds.append(decode(hyp_str))

            # collect output to be prefix for next utterance
            tgt_context[idx].append(
                hyp_ids[:-1] if hyp_ids[-1] == vocab.eos() else hyp_ids
            )

        bar.update(len(src_lengths))

    bar.close()
    # print BLEU
    print(sacrebleu.corpus_bleu(preds, [refs]).format())

    if args.comet_model is not None:
        assert (
            args.comet_path is not None
        ), "need to provide a path to download/load comet"
        # download comet and load
        comet_model = download_model(args.comet_model, args.comet_path)
        print("running comet evaluation....")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref}
            for src, mt, ref in zip(srcs, preds, refs)
        ]
        _, scores = comet_model.predict(comet_input, cuda=True, show_progress=True)
        print(f"COMET = {sum(scores)/len(scores):.4f}")

    if args.print_output is not None:
        with open(args.print_output, "w") as f:
            for hyp in preds:
                print(hyp, file=f)


if __name__ == "__main__":
    main()
