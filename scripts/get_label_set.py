import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-file", required=True, help="file to be translated")
    parser.add_argument("--labels-file", default=None)
    args = parser.parse_args()
    with open(args.tag_file, "r") as file:
        lines = srcs = [line.strip().split(" ") for line in file]

    label_set = dict()

    for line in lines:
        for tags in line:
            for tag in tags.split("+"):
                label_set[tag] = 1

    label_set = sorted(list(label_set.keys()))

    with open(args.labels_file, "w", encoding="utf-8") as file:
        file.write("+".join(label_set))
