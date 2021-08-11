from typing import List
from abc import ABC, abstractmethod
import argparse
import requests

from contextual_mt.utils import parse_documents

import tqdm

class DocumentMTProvider(ABC):
    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_document(self, source: List[str], context_size: int=None):
        if context_size is not None:
            target = []
            for i in range(len(source)):
                start_idx = max(i-context_size, 0)
                end_idx = i+1
                partial_source = source[start_idx:end_idx]
                output = self._translate(self.STOP_TOKEN.join(partial_source))
                partial_target = output.split(self.STOP_TOKEN)
                if len(partial_source) != len(partial_target):
                    import ipdb; ipdb.set_trace()
                target.append(partial_target[-1])
        else:
            source_str = self.STOP_TOKEN.join(source)
            target_str = self._translate(source_str)
            target = target_str.split(self.STOP_TOKEN)
        
        if len(source) != len(target):
            if context_size == 0:
                raise ValueError("something is wrong")

            print("document couldn't be translated with context-aware; trying sentence-level")
            return self.translate_document(source, context_size=0)

        return target


    @abstractmethod
    def _translate(self, source_str: str):
        raise NotImplementedError("")
    

class FreeGoogleProvider(DocumentMTProvider):
    def __init__(self, source_lang, target_lang):
        super().__init__(source_lang, target_lang)
        self.STOP_TOKEN = " 🛑."
        from googletrans import Translator
        self.translator = Translator()

    def _translate(self, source_str: str):
        return self.translator.translate(source_str, src=self.source_lang, dest=self.target_lang).text


class DeepLProvider(DocumentMTProvider):
    def __init__(self, source_lang, target_lang, api_key):
        super().__init__(source_lang, target_lang)
        self.STOP_TOKEN = "<x>STOP</x>"
        self.api_key = api_key

    def _translate(self, source_str: str):
        result = requests.post("https://api.deepl.com/v2/translate", 
            data={
                "source_lang": self.source_lang.upper(),
                "target_lang": self.target_lang.upper(),
                "text": source_str,
                "auth_key": self.api_key,
                "tag_handling": "xml",
                "ignore_tags": "x"
        })
        try:
            return result.json()['translations'][0]['text']
        except KeyError:
            print(result)
            raise Exception()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="free_google")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--source-file")
    parser.add_argument("--docids-file")
    parser.add_argument("--translations-file")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", default=None)
    args = parser.parse_args()

    # load files needed
    with open(args.source_file, "r", encoding='utf-8') as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.docids_file, "r", encoding='utf-8') as docids_f:
        docids = [int(idx) for idx in docids_f]

    refs = [None for _ in srcs]
    documents = parse_documents(srcs, refs, docids)

    if args.provider == "free_google":
        if args.target_lang == "zh_cn":
            args.target_lang = "zh-CN"
        if args.target_lang == "pt_br":
            args.target_lang = "pt"
        provider = FreeGoogleProvider(args.source_lang, args.target_lang)
    elif args.provider == "deepl":
        if args.target_lang == "zh_cn":
            args.target_lang = "zh"
        if args.target_lang == "pt_br":
            args.target_lang = "pt-br"
        provider = DeepLProvider(args.source_lang, args.target_lang, args.api_key)
    else:
        raise ValueError("unknown provider")

    translations = []
    for doc in tqdm.tqdm(documents):
        sources = [s for s, _ in doc]
        translations.extend(provider.translate_document(sources))

    assert len(translations) == len(srcs)

    with open(args.translations_file, "w", encoding='utf-8') as f:
        for translation in translations:
            print(translation, file=f)


if __name__ == "__main__":
    main()