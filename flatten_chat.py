#!/usr/bin/env python
""" 'flattens' a chat dataset into a line-by-line example """
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    args = parser.parse_args()

    with open(args.data, "r") as f:
        chat_data = json.load(f)

    for chat in chat_data.values():
        for turn in chat:
            print(turn["source"])
            print(turn["target"])


if __name__ == "__main__":
    main()
