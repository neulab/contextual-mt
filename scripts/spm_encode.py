import argparse
import sys

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", default=[None])
    parser.add_argument("--outputs", type=str, nargs="+", default=[None])
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    for inp, out in zip(args.inputs, args.outputs):
        if inp is not None and out is not None:
            inp_f = open(inp, "r")
            out_f = open(out, "w")
        else:
            inp_f = sys.stdin
            out_f = sys.stdout

        for line in inp_f:
            pieces = sp.encode(line, out_type=str)
            print(" ".join(pieces), file=out_f)


if __name__ == "__main__":
    main()
