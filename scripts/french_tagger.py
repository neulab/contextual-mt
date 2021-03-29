
import argparse
import os
import re

from contextual_mt.utils import encode, create_context, parse_documents

# import spacy
# from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
# from spacy_lefff import LefffLemmatizer, POSTagger

# tagger = spacy.load("fr_core_news_sm", pipeline=["tagger"])
# pos = POSTagger()
# french_lemmatizer = LefffLemmatizer(after_melt=True, default=True)
# tagger.add_pipe(pos, name='pos', after='tagger')
# tagger.add_pipe(french_lemmatizer, name='lefff', after='pos')

tu_words = ["tu", "ton", "ta", "tes", "toi"]
vous_words = ["vous", "votre", "vos"]

def lexical_cohesion(cur_tgt, ctx_tgt):
    tags = []
    context_words = map(lambda x: re.sub(r"^\W+|\W+$", '', x.lower()), ctx_tgt.split(" "))
    cur_tgt = cur_tgt.split(" ")
    for word in cur_tgt:
        word = re.sub(r"^\W+|\W+$", '', word.lower())
        if word not in fr_stop and word in context_words:
            tags.append(True)
        else:
            tags.append(False)
    assert len(tags) == len(cur_tgt)
    return tags

def tense_cohesion(cur_tgt, ctx_tgt):
    tags = []
    ctx_doc = tagger(ctx_tgt)
    cur_doc = tagger(cur_tgt)
    cur_tgt = cur_tgt.split(" ")

    prev_tenses = []
    for tok in ctx_doc:
        if "|" in tok.tag_:
            vform = dict(x.split("=") for x in tok.tag_.split("|")).get("VerbForm")
            if vform is not None:
                prev_tenses.append(vform)
    cur_tenses = dict()
    for d in cur_doc:
        if "|" in d.tag_:
            cur_tenses[re.sub(r"^\W+|\W+$", '', d.text.lower())] = dict(x.split("=") for x in d.tag_.split("|")).get("VerbForm")
    for word in cur_tgt:
        word = re.sub(r"^\W+|\W+$", '', word.lower())
        if cur_tenses.get(word) is not None:
            tags.append(True)
        else:
            tags.append(False)
    assert len(tags) == len(cur_tgt)
    return tags


def pronouns(src, ref, align):
    src = src.split(" ")
    ref = ref.split(" ")
    tags = [False] * len(ref)
    for s, r in align.items():
        if re.sub(r"^\W+|\W+$", '', src[s].lower()) in ["it", "they"]:
            ref_i = ref[r].lower()
            if "'" in ref_i:
                ref_i = ref_i.split("'")[-1]
            if re.sub(r"^\W+|\W+$", '', ref_i) in ["il", "ils", "elle", "elles"]:
                tags[r] = True
    return tags

def vt_cohesion(cur_tgt, ctx_tgt):
    tags = []
    context_words = map(lambda x: re.sub(r"^\W+|\W+$", '', x.lower()), ctx_tgt.split(" "))
    cur_tgt = cur_tgt.split(" ")

    t = False
    v = False
    for word in context_words:
        if word in tu_words:
            t = True
        elif word in vous_words:
            v = True
    if t and ~v:
        prev_tv = tu_words
    elif v and ~t:
        prev_tv = vous_words
    else:
        prev_tv = []

    for word in cur_tgt:
        word = re.sub(r"^\W+|\W+$", '', word.lower())
        if word in prev_tv:
            tags.append(True)
        else:
            tags.append(False)
    assert len(tags) == len(cur_tgt)
    return tags

def ellipsis(src, ref, align):
    src = src.split(" ")
    ref = ref.split(" ")
    tags = [False] * len(ref)
    for i in range(len(ref)):
        if i not in align.values():
            word = re.sub(r"^\W+|\W+$", '', ref[i].lower())
            if word not in fr_stop:
                tags[i] = True
    return tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument(
        "--reference-file",
        required=True,
        help="file with reference translations",
    )
    parser.add_argument(
        "--alignments-file",
        required=True,
    )
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_file = open(args.output, "w")
    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.reference_file, "r") as tgt_f:
        refs = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [idx for idx in docids_f]
    with open(args.alignments_file, "r") as file:
        alignments = file.readlines()
    alignments = list(map(lambda x: dict(list(map(lambda y: list(map(int,y.split("-"))), x.strip().split(" ")))), alignments))


    documents = parse_documents(srcs, refs, docids)
    doc_idx = 0
    ids = []
    cur_doc = None
    cur_doc_id = -1
    cur_doc_pos = 0
    i = 0
    while True:
        if cur_doc is None or cur_doc_pos >= len(cur_doc):
            if doc_idx < len(documents):
                cur_doc = documents[doc_idx]
                cur_doc_id = doc_idx
                cur_doc_pos = 0
                src_context_lines = []
                tgt_context_lines = []
                doc_idx += 1
            else:
                break
                
        src_l, tgt_l = cur_doc[cur_doc_pos]
        ids.append((cur_doc_id, cur_doc_pos))
        align = alignments[i]

        tgt_context = " ".join(tgt_context_lines[len(tgt_context_lines) - args.target_context_size :])
        #lc = lexical_cohesion(tgt_l, tgt_context)
        #tense = tense_cohesion(tgt_l, tgt_context)
        #vt = vt_cohesion(tgt_l, tgt_context)
        pronoun = pronouns(src_l, tgt_l, align)
        #ellip = ellipsis(src_l, tgt_l, align)
        lc = [False for _ in range(len(pronoun))]
        tense = [False for _ in range(len(lc))]
        vt = [False for _ in range(len(lc))]
        ellip = [False for _ in range(len(lc))]
        tags = []
        for l, t, v, p, e in zip(lc, tense, vt, pronoun, ellip):
            if p:
                tags.append("pronoun")
            elif v:
                tags.append("vous_tu")
            elif t:
                tags.append("verb_tense")
            elif e:
                tags.append("ellipsis")
            elif l:
                tags.append("lexical")
            else:
                tags.append("other")
        print(" ".join(tags), file=output_file)

        src_context_lines.append(src_l)
        tgt_context_lines.append(tgt_l)
        cur_doc_pos += 1
        i += 1

    output_file.close()




if __name__ == "__main__":
    main()




