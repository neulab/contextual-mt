import argparse
import stanza
import spacy_stanza
from mosestokenizer import MosesTokenizer, MosesDetokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--detok", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]

    with open(args.output, "w", encoding="utf-8") as output_f:
        lang = args.lang.split("_")[0] if args.lang != "zh_tw" else "zh-hant"

        try:
            tokenize = spacy_stanza.load_pipeline(lang, processors="tokenize,mwt")
        except FileNotFoundError:
            tokenize = spacy_stanza.load_pipeline(lang, processors="tokenize")

        for line in lines:
            doc = tokenize(line)
            line = (
                " ".join([token.text for token in doc])
                if not args.detok
                else "".join([token.text_with_ws for token in doc])
            )
            print(line, file=output_f)


if __name__ == "__main__":
    main()
