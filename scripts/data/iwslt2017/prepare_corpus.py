import argparse
import os
import xmltodict
import glob


def extract_train(train_inp, train_out, docids_out=None):
    with open(train_inp, "r", encoding="utf-8") as f:
        all_docs = f.read()
        docs_xml = [f"{d}</doc>" for d in all_docs.split("</doc>")[:-1]]
        docs = [xmltodict.parse(doc) for doc in docs_xml]

    train_out_f = open(train_out, "w", encoding="utf-8")
    if docids_out is not None:
        docids_out_f = open(docids_out, "w", encoding="utf-8")

    for doc in docs:
        if "#text" not in doc["doc"]:
            continue
        doc_id = doc["doc"]["@docid"]
        for line in doc["doc"]["#text"].split("\n"):
            print(line.strip(), file=train_out_f)
            if docids_out is not None:
                print(doc_id, file=docids_out_f)


def extract_eval(eval_inp, eval_out, docids_out=None):
    with open(eval_inp, "r", encoding="utf-8") as f:
        all_docs = f.read()
        docs_xml = xmltodict.parse(all_docs)
        docs = (
            docs_xml["mteval"]["srcset"]["doc"]
            if "srcset" in docs_xml["mteval"]
            else docs_xml["mteval"]["refset"]["doc"]
        )

    eval_out_f = open(eval_out, "a")
    if docids_out is not None:
        docids_out_f = open(docids_out, "a")

    for doc in docs:
        doc_id = doc["@docid"]
        if not doc["seg"]:
            continue
        for seg in doc["seg"]:
            line = seg["#text"]
            print(line.strip(), file=eval_out_f)
            if docids_out is not None:
                print(doc_id, file=docids_out_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parser for iwslt2017 data")

    parser.add_argument("raw_data")
    parser.add_argument("out_data")
    parser.add_argument("-s", "--source-lang", required=True, type=str)
    parser.add_argument("-t", "--target-lang", required=True, type=str)
    parser.add_argument(
        "--devsets", nargs="*", default=["tst2011", "tst2012", "tst2013", "tst2014"]
    )
    parser.add_argument("--testsets", nargs="*", default=["tst2015"])
    args = parser.parse_args()

    first_l = (
        args.source_lang
        if glob.glob(f"{args.raw_data}/*.{args.source_lang}-{args.target_lang}.*")
        else args.target_lang
    )
    second_l = (
        args.target_lang
        if glob.glob(f"{args.raw_data}/*.{args.source_lang}-{args.target_lang}.*")
        else args.source_lang
    )

    train_inp_prefix = os.path.join(args.raw_data, f"train.tags.{first_l}-{second_l}")
    train_out_prefix = os.path.join(
        args.out_data, f"train.{args.source_lang}-{args.target_lang}"
    )
    extract_train(
        f"{train_inp_prefix}.{args.source_lang}",
        f"{train_out_prefix}.{args.source_lang}",
        docids_out=f"{train_out_prefix}.docids",
    )
    extract_train(
        f"{train_inp_prefix}.{args.target_lang}",
        f"{train_out_prefix}.{args.target_lang}",
    )

    # generate valid set based on devsets paseed
    valid_out_prefix = os.path.join(
        args.out_data, f"valid.{args.source_lang}-{args.target_lang}"
    )
    for devset in args.devsets:
        valid_inp_prefix = os.path.join(
            args.raw_data, f"IWSLT17.TED.{devset}.{first_l}-{second_l}"
        )
        extract_eval(
            f"{valid_inp_prefix}.{args.source_lang}.xml",
            f"{valid_out_prefix}.{args.source_lang}",
            docids_out=f"{valid_out_prefix}.docids",
        )
        extract_eval(
            f"{valid_inp_prefix}.{args.target_lang}.xml",
            f"{valid_out_prefix}.{args.target_lang}",
        )

    # generate test set based on testsets paseed
    test_out_prefix = os.path.join(
        args.out_data, f"test.{args.source_lang}-{args.target_lang}"
    )
    for testset in args.testsets:
        test_inp_prefix = os.path.join(
            args.raw_data, f"IWSLT17.TED.{testset}.{first_l}-{second_l}"
        )
        extract_eval(
            f"{test_inp_prefix}.{args.source_lang}.xml",
            f"{test_out_prefix}.{args.source_lang}",
            docids_out=f"{test_out_prefix}.docids",
        )
        extract_eval(
            f"{test_inp_prefix}.{args.target_lang}.xml",
            f"{test_out_prefix}.{args.target_lang}",
        )
