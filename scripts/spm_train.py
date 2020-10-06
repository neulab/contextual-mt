import argparse
import tempfile
import json
import os
from collections import Counter

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--model-prefix", type=str, required=True)
    parser.add_argument("--vocab-file", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--vocab-sample-size", type=int, default=None)
    parser.add_argument("--is-raw", action="store_true", default=False)
    parser.add_argument("--include-target", action="store_true", default=False)
    parser.add_argument("--special-symbols", type=str, nargs="+", default=[])
    args = parser.parse_args()

    special_symbols = ["<brk>", *args.special_symbols]
    if not args.is_raw:
        raw_file = tempfile.NamedTemporaryFile(delete=False).name
        with open(args.data, "r") as chat_file, open(raw_file, "w") as raw:
            chat_dataset = json.load(chat_file)
            for chat in chat_dataset.values():
                for turn in chat:
                    print(turn["source"], file=raw)
                    if args.include_target:
                        print(turn["target"], file=raw)
                    if turn["speaker"] not in special_symbols:
                        special_symbols.append(turn["speaker"])
    else:
        raw_file = args.data

    kwargs = {}
    if args.vocab_sample_size is not None:
        kwargs.update(
            {
                "input_sentence_size": args.vocab_sample_size,
                "shuffle_input_sentence": True,
            }
        )
    spm.SentencePieceTrainer.Train(
        input=raw_file,
        model_prefix=args.model_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        user_defined_symbols=special_symbols,
        **kwargs,
    )

    # since SentencePieces generates a dictionary non-compatible
    # with fairseq, we allow generating our own vocabulary
    # WARNING: this might be slow
    if args.vocab_file is not None:
        vocab = Counter()
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{args.model_prefix}.model")
        with open(raw_file, "r") as raw:
            for line in raw:
                pieces = sp.encode(line.strip(), out_type=str)
                for p in pieces:
                    if p not in special_symbols:
                        vocab[p] += 1

        with open(args.vocab_file, "w") as vocab_f:
            for symbol in special_symbols:
                print(f"{symbol} 0", file=vocab_f)
            for word, freq in vocab.most_common():
                print(f"{word} {freq}", file=vocab_f)

    if not args.is_raw:
        os.remove(raw_file)


if __name__ == "__main__":
    main()
