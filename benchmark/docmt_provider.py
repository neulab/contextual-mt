from typing import List
from abc import ABC, abstractmethod
import argparse

from contextual_mt.utils import parse_documents

import tqdm
 
from nltk.tokenize.util import xml_unescape, xml_escape

STOP_TOKEN = " 🛑 "

class DocumentMTProvider(ABC):
    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def unscape_and_split_source(self, source):
        """ """
        sentences = [xml_unescape(s) for s in source]
        splited_sentences =  [s.text for s in self.source_nlp(" ".join(sentences)).sents]
        return splitted_sentences, None

        mapping = []
        splitted_sentences = []
        for i, s in enumerate(sentences):
            splitted_s = [s.text for s in self.source_nlp(s).sents]
            splitted_sentences.extend(splitted_s)
            mapping.extend([i for i in splitted_s])
        return splitted_sentences, mapping


    def escape_and_desplit(self, splitted_target, mapping):
        sentences = [xml_escape(s) for s in target]
        prev = None
        target = []
        for i, s in zip(mapping, sentences):
            if i != prev:
                if prev is not None:
                    target.append(" ".join(curr_t))
                curr_t = []
            curr_t.append(s)
            prev = i
        return target


    def translate_document(self, source: List[str], context_size: int=None):
        if context_size is not None:
            target = []
            for i in range(len(source)):
                start_idx = max(i-context_size, 0)
                end_idx = i+1
                partial_source = source[start_idx:end_idx]
                output = self._translate(STOP_TOKEN.join([xml_unescape(s) for s in partial_source]))
                partial_target = [xml_escape(s) for s in output.split(STOP_TOKEN)]
                if len(partial_source) != len(partial_target):
                    import ipdb; ipdb.set_trace()
                target.append(partial_target[-1])
        else:
            source_str = STOP_TOKEN.join([xml_unescape(s) for s in source])
            target_str = self._translate(source_str)
            target = [xml_escape(s) for s in target_str.split(STOP_TOKEN)]
        
        if len(source) != len(target):
            import ipdb; ipdb.set_trace()

        return target


    @abstractmethod
    def _translate(self, source_str: str):
        raise NotImplementedError("")
    

class FreeGoogleProvider(DocumentMTProvider):
    def __init__(self, source_lang, target_lang):
        super().__init__(source_lang, target_lang)

        from googletrans import Translator
        self.translator = Translator()

    def _translate(self, source_str:str):
        return self.translator.translate(source_str, src=self.source_lang, dest=self.target_lang).text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="free_google")
    parser.add_argument("--source-file")
    parser.add_argument("--docids-file")
    parser.add_argument("--translations-file")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    args = parser.parse_args()

    # load files needed
    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [int(idx) for idx in docids_f]

    refs = [None for _ in srcs]
    documents = parse_documents(srcs, refs, docids)

    if args.provider == "free_google":
        provider = FreeGoogleProvider(args.source_lang, args.target_lang)
    else:
        raise ValueError("unknown provider")

    translations = []
    for doc in tqdm.tqdm(documents):
        sources = [s for s, _ in doc]
        translations.extend(provider.translate_document(sources))

    assert len(translations) == len(srcs)

    with open(args.translations_file, "w") as f:
        for translation in translations:
            print(translation, file=f)


if __name__ == "__main__":
    main()