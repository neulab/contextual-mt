import argparse
from mosestokenizer import MosesTokenizer, MosesDetokenizer
import stanza
import spacy_stanza

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.input, "r") as file:
        lines = [line.strip().strip('"') for line in file]

    with open(args.output, "w") as output_file:
        detokenize = MosesDetokenizer(args.lang.split('_')[0])
        lang = args.lang.split("_")[0] if args.lang != "zh_tw" else  "zh-hant"
        tokenize = spacy_stanza.load_pipeline(lang, processors='tokenize')
        for line in lines:
            line = detokenize(line.split())
            doc = tokenize(line)
            line = " ".join([token.text for token in doc])
            print(line, file=output_file)

    # with open(args.output, "w") as output_file:
    #     if args.lang not in []
    #     with MosesTokenizer(args.lang.split('_')[0]) as tokenize:
    #         for line in lines:
    #             print(" ".join(tokenize(line)), file=output_file)


if __name__ == "__main__":
    main()
