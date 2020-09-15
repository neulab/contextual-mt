import argparse, os, json

parser = argparse.ArgumentParser()
parser.add_argument('--contra',type=str)
args = parser.parse_args()

if __name__ == "__main__":
    if True:
        with open(args.contra + ".context.src", 'r', encoding='utf-8') as file:
            src_cxt = file.readlines()
        with open(args.contra + ".context.trg", 'r', encoding='utf-8') as file:
            tgt_cxt = file.readlines()
        with open(args.contra + ".current.src", 'r', encoding='utf-8') as file:
            src = file.readlines()
        with open(args.contra + ".current.trg", 'r', encoding='utf-8') as file:
            tgt = file.readlines()
        ret = []
        
        for s, s_c, t, t_c in zip(src, src_cxt, tgt, tgt_cxt):
            ret.append({"source": src, "target": tgt, "source_context": s_c, "target_context": t_c})
        print(len(ret))
        with open(args.contra + ".contra.json", 'w', encoding='utf-8') as file:
            json.dump(ret, file)
        raise ValueError
 