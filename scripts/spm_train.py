import argparse
from collections import Counter

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--model-prefix", type=str, required=True)
    parser.add_argument("--vocab-file", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--vocab-sample-size", type=int, default=None)
    parser.add_argument("--special-symbols", type=str, nargs="+", default=["<brk>"])
    args = parser.parse_args()

    kwargs = {}
    if args.vocab_sample_size is not None:
        kwargs.update(
            {
                "input_sentence_size": args.vocab_sample_size,
                "shuffle_input_sentence": True,
            }
        )
    spm.SentencePieceTrainer.Train(
        input=args.data,
        model_prefix=args.model_prefix,
        model_type="bpe",
        vocab_size=args.vocab_size,
        user_defined_symbols=args.special_symbols,
        **kwargs,
    )

    # since SentencePieces generates a dictionary non-compatible
    # with fairseq, we allow generating our own vocabulary
    # WARNING: this might be slow
    if args.vocab_file is not None:
        vocab = Counter()
        sp = spm.SentencePieceProcessor()
        sp.Load(f"{args.model_prefix}.model")
        with open(args.data, "r") as raw:
            for line in raw:
                pieces = sp.encode(line.strip(), out_type=str)
                for p in pieces:
                    if p not in args.special_symbols:
                        vocab[p] += 1

        with open(args.vocab_file, "w") as vocab_f:
            for symbol in args.special_symbols:
                print(f"{symbol} 0", file=vocab_f)
            for word, freq in vocab.most_common(
                args.vocab_size - len(args.special_symbols)
            ):
                print(f"{word} {freq}", file=vocab_f)


if __name__ == "__main__":
    main()
