import argparse
import pandas as pd
import numpy as np

from mosestokenizer import MosesDetokenizer

import os

drop_columns = [
    "calv",
    "kk",
    "be",
    "bn",
    "eu",
    "ms",
    "zh",
    "bs",
    "az",
    "ur",
    "ta",
    "eo",
    "mn",
    "mr",
    "gl",
    "ku",
    "et",
    "ka",
    "nb",
    "hi",
    "sl",
    "fr-ca",
    "hy",
    "my",
    "fi",
    "mk",
    "lt",
    "sq",
    "da",
    "pt",
    "sv",
    "sk",
    "id",
    "th",
    "cs",
    "uk",
    "hr",
    "el",
    "sr",
    "hu",
    "fa",
    "vi",
    "bg",
    "pl",
    "zh-tw",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("extract_dir")
    args = parser.parse_args()

    for split in ("train", "dev", "test"):
        data = pd.read_csv(
            os.path.join(args.data_dir, f"all_talks_{split}.tsv"), sep="\t"
        )
        data = data.replace(to_replace=".*_ _ NULL _ _*", value=np.nan, regex=True)
        data = data.replace(to_replace=".*__NULL__*", value=np.nan, regex=True)
        data = data.drop(drop_columns, axis=1)
        data = data[data.isnull().sum(axis=1) == 0]
        for lang in data.columns:
            if lang == "talk_name":
                continue
            with open(os.path.join(args.extract_dir, f"{split}.{lang}"), "w") as f:
                detokenize = MosesDetokenizer(lang)
                for sample in data[lang]:
                    sample = detokenize(sample.split()) if lang != "ko" else sample
                    print(sample, file=f)

        prev_talk = None
        idx = 0
        with open(os.path.join(args.extract_dir, f"{split}.docids"), "w") as f:
            for line in data["talk_name"]:
                if line.strip() != prev_talk:
                    idx += 1
                    prev_talk = line.strip()
                print(idx, file=f)


if __name__ == "__main__":
    main()
