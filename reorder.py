
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='contra/enfr00_3.txt')
    args = parser.parse_args()

    ids = []
    scores = []
    with open(args.path, 'r') as file:
        for line in file.readlines():
            i,_ = line.split(":")
            ids.append(int(i))
            scores.append(line)
    scores = [x for _,x in sorted(zip(ids,scores))]
    ids = [x for _,x in sorted(zip(ids,ids))]

    with open(args.path, 'w') as file:
        for s in scores:
            file.write(s)
