import argparse
import fasttext
from itertools import takewhile


def length_filter(line, max_length=250):
    return len(line.split(" ")) > max_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pref", type=str)
    parser.add_argument("output_pref", type=str)
    parser.add_argument("--source-lang", type=str)
    parser.add_argument("--target-lang", type=str)
    parser.add_argument("--fasttext-model", default=None)
    parser.add_argument("--max-length", default=250)
    parser.add_argument("--chunk-size", default=10000)
    args = parser.parse_args()

    model = None
    if args.fasttext_model is not None:
        model = fasttext.load_model(args.fasttext_model)

    source_in = f"{args.input_pref}.{args.source_lang}"
    target_in = f"{args.input_pref}.{args.target_lang}"
    source_out = f"{args.output_pref}.{args.source_lang}"
    target_out = f"{args.output_pref}.{args.target_lang}"

    with open(source_in, "r") as src_in, open(target_in, "r") as tgt_in, open(source_out, "w") as src_out, open(target_out, "w") as tgt_out:
        # read size chunk_size into memory
        while True:
            src_lines, tgt_lines = [], []
            lines_read = 0
            for src_l, tgt_l in takewhile(lambda _: lines_read <= args.chunk_size, zip(src_in, tgt_in)):
                src_lines.append(src_l[:-1])
                tgt_lines.append(tgt_l[:-1])
                lines_read += 1

            # break if there is nothing else to read
            if not src_lines:
                break

            # apply length filters
            filter_flags = [False for _ in range(len(src_lines))]

            assert len(filter_flags) == len(src_lines) and len(filter_flags) == len(tgt_lines)

            for idx, (src_l, tgt_l) in enumerate(zip(src_lines, tgt_lines)):
                filter_flags[idx] = filter_flags[idx] or length_filter(src_l, args.max_length)
                filter_flags[idx] = filter_flags[idx] or length_filter(tgt_l, args.max_length)

            # apply langid filters
            src_labels = model.predict(src_lines)
            tgt_labels = model.predict(tgt_lines)
            for idx, (src_lbl, tgt_lbl) in enumerate(zip(src_labels, tgt_labels)):
                filter_flags[idx] = filter_flags[idx] or (src_lbl != f"__{args.source_lang}__") or (tgt_lbl != f"__{args.target_lang}__")

            for filter_f, src_l, tgt_l in zip(filter_flags, src_lines, tgt_lines):
                if not filter_f:
                    print(src_l, file=src_out)
                    print(tgt_l, file=tgt_out)


if __name__ == "__main__":
    main()
