#!/usr/bin/env python
""" 'flattens' a chat dataset into a line-by-line example """
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--print-speakers", action="store_true", default=False)
    parser.add_argument("--src-tgt", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.data, "r") as f:
        chat_data = json.load(f)

    if args.print_speakers:
        seen = set()
        for chat in chat_data.values():
            for turn in chat:
                if turn["speaker"] not in seen:
                    print(f"{turn['speaker']} 1")
                    seen.add(turn["speaker"])
    elif args.src_tgt:
        for chat in chat_data.values():
            for turn in chat:
                print(f"src: {turn['source']}")
                print(f"tgt: {turn['target']}")
    else:
        for chat in chat_data.values():
            for turn in chat:
                print(turn["source"])



if __name__ == "__main__":
    main()
