import re
import abc
import argparse
import spacy


class Tagger(abc.ABC):
    """ Abstact class that represent a tagger for a language """

    def __init__(self):
        self.tagger = None
        self.formality_classes = {}
        self.src_neutral_pronouns = ["it", "they"]
        self.tgt_gendered_pronouns = None

    def _normalize(self, word):
        """ default normalization """
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def formality_tags(self, current, context):
        ctx_formality = None
        for word in context.split(" "):
            word = self._normalize(word)
            for formality, class_words in self.formality_classes.items():
                if word in class_words:
                    ctx_formality = formality if ctx_formality is None else "ambiguous"
                    break
            if ctx_formality == "ambiguous":
                break

        # in case of undefined formality just return everything false
        if ctx_formality is None or ctx_formality == "ambiguous":
            return [False for _ in current.split(" ")]

        # TODO: shouldn't we penalize words that are in the wrong formality class?
        tags = []
        for word in current.split(" "):
            word = self._normalize(word)
            tags.append(word in self.formality_classes[ctx_formality])

        assert len(tags) == len(current.split(" "))
        return tags

    def lexical_cohesion(self, current, context):
        tags = []
        context_words = map(self._normalize, context.split(" "))
        for word in current.split(" "):
            word = self._normalize(word)
            if len(word.split("'")) > 1:
                word = word.split("'")[1]
            tags.append(
                len(word) > 1 and word not in self.stop_words and word in context_words
            )
        assert len(tags) == len(current.split(" "))
        return tags

    def tense_cohesion(self, current, context):
        # if there is no tagger, we just pass everything false
        if self.tagger is None:
            return [False for _ in current.split(" ")]

        cur_doc = self.tagger(current)
        ctx_doc = self.tagger(context)
        prev_tenses = []
        for tok in ctx_doc:
            if tok.tag_ == "VERB":
                vform = tok.morph.get("VerbForm")
                if vform is not None:
                    prev_tenses.append(vform)
        cur_tenses = dict()
        for tok in cur_doc:
            if tok.tag_ == "VERB":
                cur_tenses[self._normalize(tok.text)] = tok.morph.get("VerbForm")

        tags = []
        for word in current.split(" "):
            word = self._normalize(word)
            if word in cur_tenses and cur_tenses[word] in prev_tenses: 
                tags.append(True)
            else:
                tags.append(False)
        assert len(tags) == len(current.split(" "))
        return tags

    def pronouns(self, src, ref, align):
        src = src.split(" ")
        ref = ref.split(" ")
        tags = [False] * len(ref)
        if self.src_neutral_pronouns is None or self.tgt_gendered_pronouns is None:
            return tags
        for s, r in align.items():
            if self._normalize(src[s]) in self.src_neutral_pronouns:
                if self._normalize(ref[r]) in self.tgt_gendered_pronouns:
                    tags[r] = True
        return tags

    def ellipsis(self, src, ref, align):
        src = src.split(" ")
        ref = ref.split(" ")
        tags = [False] * len(ref)
        for i in range(len(ref)):
            if i not in align.values():
                word = self._normalize(ref[i])
                if word not in self.stop_words:
                    tags[i] = True
        return tags

class FrenchTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {
                "tu",
                "ton",
                "ta",
                "tes",
                "toi",
                "te",
                "tien",
                "tiens",
                "tienne",
                "tiennes",
            },
            "v_class": {"vous", "votre", "vos"},
        }
        self.tgt_gendered_pronouns = ["il", "ils", "elle", "elles"]

        from spacy.lang.fr.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS
        self.tagger = spacy.load("fr_core_news_sm")


class PortugueseTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: this is incomplete
        # TODO: shit I think brazilian rules are different
        self.formality_classes = {
            "t_class": {"tu", "tua", "teu", "teus", "tuas", "te"},
            "v_class": {"você", "sua", "seu", "seus", "suas", "lhe"},
        }
        from spacy.lang.pt.stop_words import STOP_WORDS
        self.tgt_gendered_pronouns = ["ele", "ela", "eles", "elas"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy.load("pt_core_news_sm")


class SpanishTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: usted/su/sus/suyo/suya works for V class and 3rd person
        self.formality_classes = {
            "t_class": {"tú", "tu", "tus", "ti", "contigo", "tuyo", "te", "tuya"},
            "v_class": {"usted", "vosotros", "vuestro", "vuestra", "vuestras", "os"},
        }
        from spacy.lang.es.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class HebrewTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: hebrew has t-v distinction only in extreme formality cases
        
        from spacy.lang.he.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class DutchTagger(Tagger):
    def __init__(self):
        super().__init__()
        # Source: https://en.wikipedia.org/wiki/T%E2%80%93V_distinction_in_the_world%27s_languages#Dutch
        self.formality_classes = {
            "t_class": {"jij", "jouw", "jou", "jullie", "je"},
            "v_class": {"u", "men", "uw"},
        }
        from spacy.lang.nl.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class RomanianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {
                "tu",
                "el",
                "ea",
                "voi",
                "ei",
                "ele",
                "tău",
                "ta",
                "tale",
                "tine",
            },
            "v_class": {
                "dumneavoastră",
                "dumneata",
                "mata",
                "matale",
                "dânsul",
                "dânsa" "dumnealui",
                "dumneaei",
                "dumnealor",
            },
        }
        from spacy.lang.ro.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class TurkishTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"sen", "senin"},
            "v_class": {"siz", "sizin"},
        }
        from spacy.lang.tr.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class ArabicTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.ar.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class ItalianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"tu", "tuo", "tua", "tuoi"},
            "v_class": {"lei", "suo", "sua", "suoi"},
        }
        from spacy.lang.it.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class KoreanTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.ko.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class JapaneseTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.ja.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class ChineseTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class TaiwaneseTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class RussianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"ты", "тебя", "тебе", "тобой", "твой", "твоя", "твои", "тебе"},
            "v_class": {"вы", "вас", "вам", "вами", "ваш", "ваши"},
        }
        from spacy.lang.ru.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


def build_tagger(lang) -> Tagger:
    taggers = {
        "fr": FrenchTagger,
        "pt_br": PortugueseTagger,
        "es": SpanishTagger,
        "he": HebrewTagger,
        "nl": DutchTagger,
        "ro": RomanianTagger,
        "tr": TurkishTagger,
        "ar": ArabicTagger,
        "it": ItalianTagger,
        "ko": KoreanTagger,
        "ru": RussianTagger,
        "ja": JapaneseTagger,
        "zh_cn": ChineseTagger,
        "zh_tw": TaiwaneseTagger,
    }
    return taggers[lang]()


# TODO: ja, zh_tw, zh_cn taggers - need to add tokenizer. Something weird happens with ar and ko tokenization too


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--target-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument("--alignments-file", required=True, help="file with word alignments")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.target_file, "r") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [idx for idx in docids_f]
    with open(args.alignments_file, "r") as file:
        alignments = file.readlines()

    alignments = list(map(lambda x: dict(list(map(lambda y: list(map(int,y.split("-"))), x.strip().split(" ")))), alignments))

    tagger = build_tagger(args.target_lang)

    prev_docid = None
    with open(args.output, "w") as output_file:
        for source, target, docid, align in zip(srcs, tgts, docids, alignments):
            if prev_docid is None or docid != prev_docid:
                prev_docid = docid
                target_context = []

            current_tgt_ctx = " ".join(
                target_context[len(target_context) - args.target_context_size :]
            )
            lexical_tags = tagger.lexical_cohesion(target, current_tgt_ctx)
            formality_tags = tagger.formality_tags(target, current_tgt_ctx)
            tense_cohesion_tags = tagger.tense_cohesion(target, current_tgt_ctx)
            pronouns_tags = tagger.pronouns(source, target, align)
            ellipsis_tags = tagger.ellipsis(source, target, align)
            tags = []
            for i in range(len(lexical_tags)):
                if pronouns_tags[i]:
                    tags.append("pronouns")
                elif formality_tags[i]:
                    tags.append("formality")
                elif tense_cohesion_tags[i]:
                    tags.append("verbe_tense")
                elif ellipsis_tags[i]:
                    tags.append("ellipsis")
                elif lexical_tags[i]:
                    tags.append("lexical")
                else:
                    tags.append("other")

            print(" ".join(tags), file=output_file)

            target_context.append(target)


if __name__ == "__main__":
    main()
