import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument(
        "--target-file",
        required=True,
        help="file with translations",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.source_file, "r", encoding="utf-8") as file:
        en = file.readlines()
    with open(args.target_file, "r", encoding="utf-8") as file:
        fr = file.readlines()

    with open(args.output, "w", encoding="utf-8") as file:
        for e, f in zip(en, fr):
            if len(e.strip()) > 0 and len(f.strip()) > 0:
                file.write(f"{e.strip()} ||| {f.strip()}\n")
            elif len(e.strip()) > 0:
                file.write(f"{e.strip()} ||| <blank>\n")
            else:
                file.write(f"<blank> ||| <blank>\n")


if __name__ == "__main__":
    main()
