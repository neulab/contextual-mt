"""
Calculate the ratio of number of tokens in two sets of sentences
"""
import argparse
from mosestokenizer import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypo", type=str)
    parser.add_argument("--target", type=str)
    args = parser.parse_args()

    tokenize = MosesTokenizer('en')

    with open(args.hypo, "r") as f:
        hypo = f.readlines()
        hypo = map(tokenize, hypo)

    with open(args.target, "r") as f:
        target = f.readlines()
        target = map(tokenize, target)

    ratios = [len(h)/len(t) for h,t in zip(hypo, target)]
    print(sum(ratios)/len(ratios))

    

if __name__ == "__main__":
    main()
