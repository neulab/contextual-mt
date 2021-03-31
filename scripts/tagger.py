import re
import abc
import argparse
import spacy

en_tagger = spacy.load("en_core_web_sm")

class Tagger(abc.ABC):
    """ Abstact class that represent a tagger for a language """

    def __init__(self):
        self.tagger = None
        self.formality_classes = {}
        self.src_neutral_pronouns = ["it", "they", "you", "I"]
        self.tgt_gendered_pronouns = None

    def _normalize(self, word):
        """ default normalization """
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def formality_tags(self, cur_src, ctx_src, cur_tgt, ctx_tgt, cur_align, ctx_align):
        ctx_formality = None
        context = " ".join(ctx_tgt)
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
            return [False for _ in cur_tgt.split(" ")]

        # TODO: shouldn't we penalize words that are in the wrong formality class?
        tags = []
        for word in cur_tgt.split(" "):
            word = self._normalize(word)
            tags.append(word in self.formality_classes[ctx_formality])

        assert len(tags) == len(cur_tgt.split(" "))

        try: 
            tags2 = self.verb_formality(cur_src, ctx_src, cur_tgt, ctx_tgt, cur_align, ctx_align)
            assert len(tags2) == len(cur_tgt.split(" "))
            return [a or b for a,b in zip(tags, tags2)]
        except:
            return tags

    def lexical_cohesion(self, current, context):
        context = " ".join(context)
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
        context = " ".join(context)
        # if there is no tagger, use multilingual tagger
        if self.tagger is None:
            self.tagger = spacy.load("xx_ent_wiki_sm")

        cur_doc = self.tagger(current)
        ctx_doc = self.tagger(context)
        prev_tenses = []
        for tok in ctx_doc:
            if tok.pos_ == "VERB":
                vform = tok.morph.get("VerbForm")
                if vform is not None:
                    prev_tenses.append(vform)
        cur_tenses = dict()
        for tok in cur_doc:
            if tok.pos_ == "VERB":
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
        self.tgt_gendered_pronouns = ["él", "ella", "ellos", "ellas"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy.load("es_core_news_sm")

    def verb_formality(self, cur_src, ctx_src, cur_tgt, ctx_tgt, cur_align, ctx_align):
        ctx_formality = None
        tags = [False for _ in cur_tgt.split(" ")]
        
        for i, c_align in enumerate(ctx_align):
            src_doc = en_tagger(ctx_src[i])
            tgt_doc = self.tagger(ctx_tgt[i])
            align = {self._normalize(ctx_src[i].split(" ")[s]) : self._normalize(ctx_tgt[i].split(" ")[t]) for s,t in c_align.items()}

            you_verbs = []
            for tok in src_doc:
                if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                    you_verbs.append(self._normalize(tok.head.text))
            you_verbs = list(map(lambda x: self._normalize(align.get(x, "<NONE>")), you_verbs))

            for tok in tgt_doc:
                if self._normalize(tok.text) in you_verbs:
                    if '2' in tok.morph.get("Person"):
                        ctx_formality = "t_class" if ctx_formality is None else "ambiguous"
                    elif '3' in tok.morph.get("Person"):
                        ctx_formality = "v_class" if ctx_formality is None else "ambiguous"
                if ctx_formality == "ambiguous":
                    break

        # in case of undefined formality just return everything false
        if ctx_formality is None or ctx_formality == "ambiguous":
            return tags

        src_doc = en_tagger(cur_src)
        tgt_doc = self.tagger(cur_tgt)
        align = {self._normalize(cur_src.split(" ")[s]) : self._normalize(cur_tgt.split(" ")[t]) for s,t in cur_align.items()}
        you_verbs = []
        for tok in src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))
        
        cur_tgt = cur_tgt.split(" ")
        for tok in tgt_doc:
            if self._normalize(tok.text) in you_verbs:
                if '2' in tok.morph.get("Person") and ctx_formality == "t_class": 
                    try:
                        tags[cur_tgt.index(tok.text)] = True 
                    except:
                        pass 
                if '3' in tok.morph.get("Person") and ctx_formality == "v_class": 
                    try:
                        tags[cur_tgt.index(tok.text)] = True 
                    except:
                        pass 
        return tags


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
        self.tagger = spacy.load("nl_core_news_sm")


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
        self.tgt_gendered_pronouns = ["el", "ei", "ea", "ele"]
        self.stop_words = STOP_WORDS
        self.tagger = spacy.load("ro_core_news_sm")


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

        self.tgt_gendered_pronouns = ["هم", "هن", "أنتم", "أنتن", "انتَ", "انتِ", "هو", "هي"]


class ItalianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"tu", "tuo", "tua", "tuoi"},
            "v_class": {"lei", "suo", "sua", "suoi"},
        }
        from spacy.lang.it.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tgt_gendered_pronouns = ["esso", "essa"]
        self.tagger = spacy.load("it_core_news_sm")

    def verb_formality(self, cur_src, ctx_src, cur_tgt, ctx_tgt, cur_align, ctx_align):
        ctx_formality = None
        tags = [False for _ in cur_tgt.split(" ")]
        
        for i, c_align in enumerate(ctx_align):
            src_doc = en_tagger(ctx_src[i])
            tgt_doc = self.tagger(ctx_tgt[i])
            align = {self._normalize(ctx_src[i].split(" ")[s]) : self._normalize(ctx_tgt[i].split(" ")[t]) for s,t in c_align.items()}

            you_verbs = []
            for tok in src_doc:
                if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                    you_verbs.append(self._normalize(tok.head.text))
            you_verbs = list(map(lambda x: self._normalize(align.get(x, "<NONE>")), you_verbs))

            for tok in tgt_doc:
                if self._normalize(tok.text) in you_verbs:
                    if '2' in tok.morph.get("Person"):
                        ctx_formality = "t_class" if ctx_formality is None else "ambiguous"
                    elif '3' in tok.morph.get("Person"):
                        ctx_formality = "v_class" if ctx_formality is None else "ambiguous"
                if ctx_formality == "ambiguous":
                    break

        # in case of undefined formality just return everything false
        if ctx_formality is None or ctx_formality == "ambiguous":
            return tags

        src_doc = en_tagger(cur_src)
        tgt_doc = self.tagger(cur_tgt)
        align = {self._normalize(cur_src.split(" ")[s]) : self._normalize(cur_tgt.split(" ")[t]) for s,t in cur_align.items()}
        you_verbs = []
        for tok in src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))
        
        cur_tgt = cur_tgt.split(" ")
        for tok in tgt_doc:
            if self._normalize(tok.text) in you_verbs:
                if '2' in tok.morph.get("Person") and ctx_formality == "t_class": 
                    try:
                        tags[cur_tgt.index(tok.text)] = True 
                    except:
                        pass 
                if '3' in tok.morph.get("Person") and ctx_formality == "v_class": 
                    try:
                        tags[cur_tgt.index(tok.text)] = True 
                    except:
                        pass 
        return tags

class KoreanTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"저", "tuo", "tua", "tuoi"},
            "v_class": {"lei", "suo", "sua", "suoi"},
        }
        
        from spacy.lang.ko.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS


class JapaneseTagger(Tagger):
    def __init__(self):
        super().__init__()
        # Formality verb forms from https://www.aclweb.org/anthology/D19-5203.pdf adapted to stanza tokens
        self.formality_classes = {
            "t_class": {"だ", "だっ", "じゃ", "だろう", "だ", "だけど", "だっ"},
            "v_class": {"ござい", "ます", "いらっしゃれ", "いらっしゃい", "ご覧", "伺い", "伺っ", "存知"},
        }
        
        from spacy.lang.ja.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.tgt_gendered_pronouns = ["私", "僕", "俺"]
        self.tagger = spacy.load("ja_core_news_sm")


class ChineseTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.formality_classes = {
            "t_class": {"你"},
            "v_class": {"您"},
        }
        self.tagger = spacy.load("zh_core_web_sm")


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
        self.tagger = spacy.load("ru_core_web_sm")


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
                source_context = []
                target_context = []
                align_context = []

            # current_src_ctx = " ".join(
            #     source_context[len(source_context) - args.source_context_size :]
            # )
            # current_tgt_ctx = " ".join(
            #     target_context[len(target_context) - args.target_context_size :]
            # )
            # current_align_ctx = " ".join(
            #     align_context[len(align_context) - max(args.source_context_size, args.target_context_size) :]
            # )
            current_src_ctx = source_context[len(source_context) - args.source_context_size :]
            current_tgt_ctx = target_context[len(target_context) - args.target_context_size :]
            current_align_ctx = align_context[len(align_context) - max(args.source_context_size, args.target_context_size) :]

            lexical_tags = tagger.lexical_cohesion(target, current_tgt_ctx)
            formality_tags = tagger.formality_tags(source, current_src_ctx, target, current_tgt_ctx, align, current_align_ctx)
            tense_cohesion_tags = tagger.tense_cohesion(target, current_tgt_ctx)
            pronouns_tags = tagger.pronouns(source, target, align)
            ellipsis_tags = tagger.ellipsis(source, target, align)
            tags = []
            for i in range(len(lexical_tags)):
                tag = ["all"]
                if pronouns_tags[i]:
                    tag.append("pronouns")
                if formality_tags[i]:
                    tag.append("formality")
                if tense_cohesion_tags[i]:
                    tag.append("verb_tense")
                if ellipsis_tags[i]:
                    tag.append("ellipsis")
                if lexical_tags[i]:
                    tag.append("lexical")
                if len(tag) == 0:
                    tag.append("other")
                tags.append("+".join(tag))

            print(" ".join(tags), file=output_file)

            source_context.append(source)
            target_context.append(target)


if __name__ == "__main__":
    main()
