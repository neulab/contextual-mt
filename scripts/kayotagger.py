import re
import abc
import argparse

class Tagger(abc.ABC):
    """ Abstact class that represent a tagger for a language """

    def _normalize(self, word):
        """ default normalization """ 
        return re.sub(r"^\W+|\W+$", '', word.lower())

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
            tags.append(len(word) > 1 and word not in self.stop_words and word in context_words)
        assert len(tags) == len(current.split(" "))
        return tags 

class FrenchTagger(Tagger):
    def __init__(self):
        self.formality_classes = {
            "t_class": {"tu", "ton", "ta", "tes", "toi", "te", "tien", "tiens", "tienne", "tiennes"},
            "v_class": {"vous", "votre", "vos"}
        }
        from spacy.lang.fr.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS


class PortugueseTagger(Tagger):
    def __init__(self):
        # TODO: this is incomplete 
        # TODO: shit I think brazilian rules are different
        self.formality_classes = {
            "t_class": {"tu", "tua", "teu", "teus", "tuas", "te"},
            "v_class": {"você", "sua", "seu", "seus", "suas", "lhe"}
        }
        from spacy.lang.pt.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class SpanishTagger(Tagger):
    def __init__(self):
        # TODO: usted/su/sus/suyo/suya works for V class and 3rd person
        self.formality_classes = {
            "t_class": {"tú", "tu", "tus", "ti", "contigo", "tuyo", "te", "tuya"},
            "v_class": {"usted", "vosotros", "vuestro", "vuestra", "vuestras", "os"}
        }
        from spacy.lang.es.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class HebrewTagger(Tagger):
    def __init__(self):
        # TODO: hebrew has t-v distinction only in extreme formality cases
        self.formality_classes = {}
        from spacy.lang.he.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class DutchTagger(Tagger):
    def __init__(self):
        # Source: https://en.wikipedia.org/wiki/T%E2%80%93V_distinction_in_the_world%27s_languages#Dutch
        self.formality_classes = {
            "t_class": {"jij", "jouw", "jou", "jullie", "je"},
            "v_class": {"u", "men", "uw"}
        }
        from spacy.lang.nl.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class RomanianTagger(Tagger):
    def __init__(self):
        self.formality_classes = {
            "t_class": {"tu", "el", "ea", "voi", "ei", "ele", "tău", "ta", "tale", "tine"},
            "v_class": {"dumneavoastră", "dumneata", "mata", "matale", "dânsul", "dânsa" \
                        "dumnealui", "dumneaei", "dumnealor"}
        }
        from spacy.lang.ro.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class TurkishTagger(Tagger):
    def __init__(self):
        self.formality_classes = {
            "t_class": {"sen", "senin"},
            "v_class": {"siz", "sizin"}
        }
        from spacy.lang.tr.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class ArabicTagger(Tagger):
    def __init__(self):
        self.formality_classes = {}
        from spacy.lang.ar.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class ItalianTagger(Tagger):
    def __init__(self):
        self.formality_classes = {
            "t_class": {"tu", "tuo", "tua", "tuoi"},
            "v_class": {"lei", "suo", "sua", "suoi"}
        }
        from spacy.lang.it.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class KoreanTagger(Tagger):
    def __init__(self):
        self.formality_classes = {}
        from spacy.lang.ko.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class JapaneseTagger(Tagger):
    def __init__(self):
        self.formality_classes = {}
        from spacy.lang.ja.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class ChineseTagger(Tagger):
    def __init__(self):
        self.formality_classes = {}
        from spacy.lang.zh.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class TaiwaneseTagger(Tagger):
    def __init__(self):
        self.formality_classes = {}
        from spacy.lang.zh.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS

class RussianTagger(Tagger):
    def __init__(self):
        self.formality_classes = {
            "t_class": {"ты", "тебя", "тебе", "тобой", "твой", "твоя", "твои", "тебе"},
            "v_class": {"вы", "вас", "вам", "вами", "ваш", "ваши"}
        }
        from spacy.lang.ru.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True, help="file to be translated")
    parser.add_argument("--target-file", required=True, help="file to be translated")
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument("--align-file", default=None, help="file with word alignments")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    taggers = {"fr": FrenchTagger(), "pt_br": PortugueseTagger(), "es": SpanishTagger(), \
                "he": HebrewTagger(), "nl": DutchTagger(), "ro": RomanianTagger(), \
                "tr": TurkishTagger(), "ar": ArabicTagger(), "it": ItalianTagger(), \
                "ko": KoreanTagger(), "ru": RussianTagger(), "ja": JapaneseTagger(), \
                "zh_cn": ChineseTagger(), "zh_tw": TaiwaneseTagger()}

    with open(args.source_file, "r") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.target_file, "r") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [idx for idx in docids_f]

    tagger = taggers.get(args.target_lang)
    if tagger is None:
        raise ValueError("unknown language")

    prev_docid = None
    with open(args.output, "w") as output_file:
        for _, target, docid in zip(srcs, tgts, docids):
            if prev_docid is None or docid != prev_docid:
                prev_docid = docid
                target_context = []

            current_tgt_ctx = " ".join(target_context[len(target_context) - args.target_context_size:])
            lexical_tags = tagger.lexical_cohesion(target, current_tgt_ctx)
            formality_tags = tagger.formality_tags(target, current_tgt_ctx)
            tags = []
            for i in range(len(lexical_tags)):
                if formality_tags[i]:
                    tags.append("formality")
                elif lexical_tags[i]:
                    tags.append("lexical")
                else:
                    tags.append("other")

            print(" ".join(tags), file=output_file)

            target_context.append(target)

if __name__ == "__main__":
    main()