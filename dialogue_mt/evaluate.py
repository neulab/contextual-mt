import argparse
import os
import json
import sacrebleu
import tqdm

import dialogue_mt

import torch

from fairseq import hub_utils
from fairseq import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        help="colon separated path to data directories list, \
                    will be iterated upon during epochs in round-robin manner; \
                    however, valid and test data are always in the first directory to \
                    avoid the need for repeating them in all directories",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--path", metavar="FILE", help="path to model file")
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

    args = parser.parse_args()

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
    source_context_size = pretrained["task"].args.source_context_size
    target_context_size = pretrained["task"].args.target_context_size
    generator = pretrained["task"].build_generator(models, args)

    data_path = os.path.join(args.data, f"{args.split}.json")

    def binarize(s, speaker=None):
        """ binarizes a sentence by applying bpe and tokenization and adding a speaker tag """
        s = tokenizer.encode(s)
        s = bpe.encode(s)
        tokens = vocab.encode_line(s, append_eos=False, add_if_not_exist=False).long()
        if speaker is not None:
            spk_tensor = torch.tensor([vocab.index(speaker)])
            tokens = torch.cat([spk_tensor, tokens])
        return tokens

    with open(data_path, "r") as f:
        chat_dict = json.load(f)

    refs = []
    preds = []
    pbar = tqdm.tqdm(total=sum(1 for chat in chat_dict.values() for turn in chat))
    for chat in chat_dict.values():
        src_context = []
        tgt_context = []
        for turn in chat:
            # binarize source and create input with context and target
            src_ids = binarize(turn["source"], speaker=turn["speaker"])
            src_ctx_ids = [
                idx
                for l in src_context[len(src_context) - source_context_size :]
                for idx in [*l, torch.tensor(vocab.index("<brk>"))]
            ]
            tgt_ctx_ids = [
                idx
                for l in tgt_context[len(tgt_context) - target_context_size :]
                for idx in [*l, torch.tensor(vocab.index("<brk>"))]
            ]
            # create batch with size 1
            inputs = torch.stack([*src_ctx_ids, *src_ids, torch.tensor(vocab.eos())])
            target = torch.stack(tgt_ctx_ids) if tgt_ctx_ids else None

            # run inference and collect output for next prefix
            sample = {
                "net_input": {
                    "src_tokens": torch.unsqueeze(inputs, 0),
                    "src_lengths": torch.tensor([len(inputs)]),
                },
                "target": target,
            }
            sample = utils.move_to_cuda(sample)
            hyp_ids = pretrained["task"].inference_step(generator, models, sample)[0][
                0
            ]["tokens"]
            hyp_str = vocab.string(hyp_ids.int().cpu(), bpe_symbol="@@ ")

            refs.append(turn["target"])
            preds.append(hyp_str)

            src_context.append(src_ids)
            tgt_context.append(hyp_ids[:-1])
            pbar.update(1)

    print(f"BLEU score = {sacrebleu.corpus_bleu(preds, [refs]).score}\n")


if __name__ == "__main__":
    main()
