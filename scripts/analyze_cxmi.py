import argparse
import sys
import re
from collections import defaultdict

order = [ "Prev-src", "Prev-tgt", "Cur-src", "Cur-tgt", "CXMI", "TAG"]

# Find intervals
def interval_vals(my_data, pos):
  my_col = [x[pos] for x in my_data]
  my_col.sort()
  lmc = len(my_col)
  return my_col[int(lmc/4)], my_col[int(lmc/2)], my_col[int(3*lmc/4)]

def interval(my_data, pos):
  lower,med,upper = interval_vals(my_data, pos)
  return f'{med:.3f} ({lower:.3f},{upper:.3f})'

def display_tup(tup, data):
  cxmi, ref, tag, i, j = tup
  for x in order:
    print(f'{x}: {" ".join(list(map(str,data[i][x])))}')
  print(tup)
  print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxmi-file", required=True, help="file with cxmi scores")
    parser.add_argument("--source-file", required=True, help="file with sources")
    parser.add_argument("--reference-file", required=True, help="file with references")
    parser.add_argument("--tag-file", required=True, help="file with tags")
    args = parser.parse_args()

    with open(args.cxmi_file, "r") as file:
        cxmis = [list(map(float, line.strip().split(" "))) for line in file]
    with open(args.source_file, "r") as file:
        sources = [line.strip().split(" ") for line in file]
    with open(args.reference_file, "r") as file:
        references = [line.strip().split(" ") for line in file]
    with open(args.tag_file, "r") as file:
        tags = [line.strip().split(" ") for line in file]
        

    cxmi_words = []
    data = []
    my_data = {}
    all_tags = {}

    for i, (cxmi, reference, tag) in enumerate(zip(cxmis, references, tags)):
      my_data["Prev-src"] = sources[i-1] if i > 0 else ""
      my_data["Prev-tgt"] = references[i-1] if i > 0 else ""
      my_data["Cur-src"] = sources[i]
      my_data["Cur-tgt"] = reference
      my_data["CXMI"] = cxmi
      my_data["TAG"] = tag
      for j, (cxmi_i, ref_i, tag_i) in enumerate(zip(cxmi, reference, tag)):
        tag_i = tag_i.split("+")
        cxmi_words.append((cxmi_i, ref_i, tag_i, len(data), j))
        for t in tag_i:
          all_tags[t] = 1
      
      data.append(my_data)
      my_data = {} 
    
    all_tags = sorted(list(all_tags.keys()))
    cxmi_words.sort(reverse=True)

    print('----------- By tag')
    print('tag\tcxmi_mean')
    for j in all_tags:
      filtered_data = [x for x in cxmi_words if j in x[2]]
      print(f'{j}\t'+'\t'.join([interval(filtered_data,i) for i in range(1)]))
    print()

    print('----------- Highest-scoring individual tokens')
    for tup in cxmi_words[:50]:
      display_tup(tup, data)

    cxmi_stats = defaultdict(lambda: 0)
    count_stats = defaultdict(lambda: 0)
    for tup in cxmi_words:
      cxmi, ref, tag, i, j = tup
      cxmi_stats[ref] += cxmi
      count_stats[ref] += 1
    THRESH = 5
    wordbyword_cxmi = []
    for k, v in count_stats.items():
      if v >= THRESH:
        wordbyword_cxmi.append( (cxmi_stats[k]/v, k) )
    wordbyword_cxmi.sort(reverse=True)
    print('----------- Highest-scoring vocabulary items')
    for tup in wordbyword_cxmi[:100]:
      cxmi, word = tup
      print(f'*** {word}\t{cxmi}')
      print()
      for tup in [x for x in cxmi_words if x[1] == word][:THRESH]:
        display_tup(tup, data)

if __name__ == "__main__":
  main()