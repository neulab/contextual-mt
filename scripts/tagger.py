import re
import abc
import argparse
import spacy
import spacy_stanza
from collections import defaultdict

#en_tagger = spacy.load("en_core_web_sm")
en_tagger = spacy_stanza.load_pipeline("en", processors="tokenize,pos,lemma,depparse")

class Tagger(abc.ABC):
    """ Abstact class that represent a tagger for a language """

    def __init__(self):
        self.tagger = spacy.load("xx_ent_wiki_sm")
        self.formality_classes = {}
        #self.src_neutral_pronouns = ["it", "they", "you", "I"]
        #self.tgt_gendered_pronouns = None
        self.ambiguous_pronouns = None
        self.ambiguous_verbform = []

    def _normalize(self, word):
        """ default normalization """
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def formality_tags(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align):
        tags = []
        for word in cur_tgt.split(" "):
            word = self._normalize(word)
            tags.append(word in self.formality_classes.values())
        assert len(tags) == len(cur_tgt.split(" "))

        try: 
            tags2 = self.verb_formality(cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align)
            assert len(tags2) == len(cur_tgt.split(" "))
            return [a or b for a,b in zip(tags, tags2)]
        except:
            return tags

    # def lexical_cohesion(self, current, context):
    #     context = " ".join(context)
    #     tags = []
    #     context_words = map(self._normalize, context.split(" "))
    #     for word in current.split(" "):
    #         word = self._normalize(word)
    #         if len(word.split("'")) > 1:
    #             word = word.split("'")[1]
    #         tags.append(
    #             len(word) > 1 and word not in self.stop_words and word in context_words
    #         )
    #     assert len(tags) == len(current.split(" "))
    #     return tags

    def lexical_cohesion(self, src_doc, tgt_doc, align, cohesion_words):
        src_lemmas = [t if not tok.is_stop and not tok.is_punct else None for tok in src_doc for t in tok.lemma_.split(" ")]
        tgt_lemmas = [t if not tok.is_stop and not tok.is_punct else None for tok in tgt_doc for t in tok.lemma_.split(" ")]
        tags = [False] * len(tgt_lemmas)

        for s, t in align.items():
            src_lemma = src_lemmas[s]
            tgt_lemma = tgt_lemmas[t]
            if src_lemma is not None and tgt_lemma is not None:
                cohesion_words[src_lemma][tgt_lemma] += 1
                if cohesion_words[src_lemma][tgt_lemma] > 3:
                    tags[t] = True
        return tags, cohesion_words

    def verb_form(self, cur_doc):
        tags = []
        for tok in cur_doc:
            if tok.pos_ == "VERB" and (len([a for a in tok.morph.get("Tense") if a in self.ambiguous_verbform]) > 0 or len(self.ambiguous_verbform) == 0):
                for _ in tok.text.split(" "):
                    tags.append(True)
            else:
                for _ in tok.text.split(" "):
                    tags.append(False)
        return tags

    def pronouns(self, src, ref, align):
        src = src.split(" ")
        ref = ref.split(" ")
        tags = [False] * len(ref)
        #if self.src_neutral_pronouns is None or self.tgt_gendered_pronouns is None:
        if self.ambiguous_pronouns is None:
            return tags
        for s, r in align.items():
            # if self._normalize(src[s]) in self.src_neutral_pronouns:
            #     if self._normalize(ref[r]) in self.tgt_gendered_pronouns:
            if self._normalize(ref[r]) in self.ambiguous_pronouns.get(self._normalize(src[s]), []):
                tags[r] = True
        return tags

    def ellipsis(self, ref, align):
        ref = [tok for tok in ref for _ in tok.text.split(" ")]
        tags = [False] * len(ref)
        for i, tok in enumerate(ref):
            if i not in align.values():
                word = self._normalize(tok.text)
                if word not in self.stop_words and (tok.pos_ in ["NOUN", "VERB", "PROPN"]):
                    tags[i] = True
        return tags

    def pos_morph(self, current, doc):
        tags = []
        for tok in doc:
            tag = tok.pos_
            if tag == "PRON":
                morph_tags = tok.morph.get('Person') + tok.morph.get('Number')
                m_tag = ".".join(morph_tags)
                if len(m_tag) > 0:
                    tag += "+" + "PRON." + m_tag
            elif tag == "VERB":
                m_tag = tok.morph.get("Tense")
                if len(m_tag) > 0:
                    tag += "+" + "VERB." + ".".join(m_tag)
            for _ in tok.text.split(" "):
                tags.append(tag)

        assert len(tags) == len(current.split(" "))
        return tags

    def polysemous(self, src_doc, target, align, polysemous_words):
        src_lemmas = [t if not tok.is_stop and not tok.is_punct else None for tok in src_doc for t in tok.lemma_.split(" ")]
        tags = [False] * len(target.split(" "))

        for s,t in align.items():
            if src_lemmas[s] in polysemous_words:
                tags[t] = True

        return tags



class EnglishTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.en.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS
        self.tagger = en_tagger

        self.pronoun_types = {
            "1sg": {"I", "me", "my", "mine", "myself"},
            "2": {"you", "your", "yours", "yourself", "yourselves"},
            "3sgm": {"he", "him", "his", "himself"},
            "3sgf": {"she", "her", "hers", "herself"},
            "3sgn": {"it", "its", "itself", "themself"},
            "1pl": {"we", "us", "our", "ours", "ourselves"},
            "3pl": {"they", "them", "their", "theirs", "themselves"},
        }

    def formality_tags(self, cur_src, ctx_src, cur_tgt=None, ctx_tgt=None, cur_align=None, ctx_align=None):
        #TODO?
        return [False for _ in cur_src.split(" ")]
    
    def pronouns(self, src, ref=None, align=None):
        src = src.split(" ")
        tags = []
        for word in src:
            word = self._normalize(word)
            for pro_type, pro_words in self.pronoun_types.items():
                if word in pro_words:
                    tags.append(pro_type)
                else:
                    tags.append("no_tag")
        return tags

    def ellipsis(self, src, ref=None, align=None):
        #TODO?
        return [False for _ in cur_src.split(" ")]


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
        #self.tgt_gendered_pronouns = ["il", "ils", "elle", "elles"]
        self.ambiguous_pronouns = {
            "it": ["il", "elle"],
            "they": ["ils", "elles"],
            "you": ["tu", "vous"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        from spacy.lang.fr.stop_words import STOP_WORDS
        self.stop_words = STOP_WORDS
        #self.tagger = spacy.load("fr_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("fr", processors="tokenize,pos,lemma,depparse")

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
        #self.tgt_gendered_pronouns = ["ele", "ela", "eles", "elas"]
        self.ambiguous_pronouns = {
            "it": ["ele", "ela"],
            "they": ["eles", "elas"],
        }
        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        #self.tagger = spacy.load("pt_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("pt", processors="tokenize,pos,lemma,depparse")


class GermanTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"du"},
            "v_class": {"sie"}, # formal 2nd person Sie is usually capitalized
        }
        from spacy.lang.de.stop_words import STOP_WORDS
        #self.tgt_gendered_pronouns = ["er", "sie", "es"]
        self.ambiguous_pronouns = {
            "it": ["er", "sie", "es"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline("de", processors="tokenize,pos,lemma,depparse")

class SpanishTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: usted/su/sus/suyo/suya works for V class and 3rd person
        self.formality_classes = {
            "t_class": {"tú", "tu", "tus", "ti", "contigo", "tuyo", "te", "tuya"},
            "v_class": {"usted", "vosotros", "vuestro", "vuestra", "vuestras", "os"},
        }
        from spacy.lang.es.stop_words import STOP_WORDS
        #self.tgt_gendered_pronouns = ["él", "ella", "ellos", "ellas"]
        self.ambiguous_pronouns = {
            "it": ["él", "ella"],
            "they": ["ellos", "ellas"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]


        self.stop_words = STOP_WORDS
        #self.tagger = spacy.load("es_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("es", processors="tokenize,pos,lemma,depparse")

    def verb_formality(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align):
        
        cur_src = cur_src.split(" ")
        cur_tgt = cur_tgt.split(" ")
        tags = [False] * len(cur_tgt)

        align = {self._normalize(cur_src[s]) : self._normalize(cur_tgt[t]) for s,t in cur_align.items()}
        you_verbs = []
        for tok in cur_src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))
        
        for i,tok in enumerate(cur_tgt_doc):
            if self._normalize(tok.text) in you_verbs:
                if '2' in tok.morph.get("Person") or '3' in tok.morph.get("Person"): 
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
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]
        self.tagger = spacy_stanza.load_pipeline("he", processors="tokenize,pos,lemma,depparse")


class DutchTagger(Tagger):
    def __init__(self):
        super().__init__()
        # Source: https://en.wikipedia.org/wiki/T%E2%80%93V_distinction_in_the_world%27s_languages#Dutch
        self.formality_classes = {
            "t_class": {"jij", "jouw", "jou", "jullie", "je"},
            "v_class": {"u", "men", "uw"},
        }
        from spacy.lang.nl.stop_words import STOP_WORDS
        self.ambiguous_verbform = ["Past"]
        self.stop_words = STOP_WORDS
        #self.tagger = spacy.load("nl_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("nl", processors="tokenize,pos,lemma,depparse")


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
        #self.tgt_gendered_pronouns = ["el", "ei", "ea", "ele"]
        self.ambiguous_pronouns = {
            "it": ["el", "ea"],
            "they": ["ei", "ele"],
        }
        self.ambiguous_verbform = ["Past", "Imp", "Fut"]
        self.stop_words = STOP_WORDS
        #self.tagger = spacy.load("ro_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("ro", processors="tokenize,pos,lemma,depparse")


class TurkishTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"sen", "senin"},
            "v_class": {"siz", "sizin"},
        }
        from spacy.lang.tr.stop_words import STOP_WORDS
        self.ambiguous_verbform = ["Pqp"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline("tr", processors="tokenize,pos,lemma,depparse")


class ArabicTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.ar.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        #self.tgt_gendered_pronouns = ["هم", "هن", "أنتم", "أنتن", "انتَ", "انتِ", "هو", "هي"]
        self.ambiguous_pronouns = {
            "you": ["انت", "انتَ", "انتِ", "انتى", "أنتم", "أنتن", "انتو", "أنتما", "أنتما"], 
            "it": ["هو", "هي"],
            "they": ["هم", "هن", "هما"],
        }
        self.tagger = spacy_stanza.load_pipeline("ar", processors="tokenize,pos,lemma,depparse")


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
        self.ambiguous_pronouns = {
            "it": ["esso", "essa"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        #self.tagger = spacy.load("it_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("it", processors="tokenize,pos,lemma,depparse")

    def verb_formality(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align):
        
        cur_src = cur_src.split(" ")
        cur_tgt = cur_tgt.split(" ")
        tags = [False] * len(cur_tgt)

        align = {self._normalize(cur_src[s]) : self._normalize(cur_tgt[t]) for s,t in cur_align.items()}
        you_verbs = []
        for tok in cur_src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))
        
        for i,tok in enumerate(cur_tgt_doc):
            if self._normalize(tok.text) in you_verbs:
                if '2' in tok.morph.get("Person") or '3' in tok.morph.get("Person"): 
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
        self.tagger = spacy_stanza.load_pipeline("ko", processors="tokenize,pos,lemma,depparse")


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

        #self.tgt_gendered_pronouns = ["私", "僕", "俺"]
        self.ambiguous_pronouns = {
            "i": ["私", "僕", "俺"],
        }
        #self.tagger = spacy.load("ja_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline("ja", processors="tokenize,pos,lemma,depparse")


class ChineseTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.formality_classes = {
            "t_class": {"你"},
            "v_class": {"您"},
        }
        #self.tagger = spacy.load("zh_core_web_sm")
        self.tagger = spacy_stanza.load_pipeline("zh", processors="tokenize,pos,lemma,depparse")


class TaiwaneseTagger(Tagger):
    def __init__(self):
        super().__init__()
        
        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline("zh", processors="tokenize,pos,lemma,depparse")


class RussianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"ты", "тебя", "тебе", "тобой", "твой", "твоя", "твои", "тебе"},
            "v_class": {"вы", "вас", "вам", "вами", "ваш", "ваши"},
        }
        from spacy.lang.ru.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pres", "Past", "Fut"]
        #self.tagger = spacy.load("ru_core_web_sm")
        self.tagger = spacy_stanza.load_pipeline("ru", processors="tokenize,pos,lemma,depparse")


def build_tagger(lang) -> Tagger:
    taggers = {
        "en": EnglishTagger,
        "fr": FrenchTagger,
        "pt_br": PortugueseTagger,
        "es": SpanishTagger,
        "de": GermanTagger,
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
    parser.add_argument("--polysemous-file", required=True, help="file with polysemous words")
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
    with open(args.source_file.replace("tok", "detok"), "r") as src_f:
        detok_srcs = [line.strip() for line in src_f]
    with open(args.target_file.replace("tok", "detok"), "r") as tgt_f:
        detok_tgts = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r") as docids_f:
        docids = [idx for idx in docids_f]
    with open(args.alignments_file, "r") as file:
        alignments = file.readlines()
    with open(args.polysemous_file, "r") as file:
        polysemous_words = [line.strip() for line in file]

    alignments = list(map(lambda x: dict(list(map(lambda y: list(map(int,y.split("-"))), x.strip().split(" ")))), alignments))

    tagger = build_tagger(args.target_lang)

    src_docs = en_tagger.pipe(detok_srcs)
    tgt_docs = tagger.tagger.pipe(detok_tgts)

    prev_docid = None
    with open(args.output, "w") as output_file:
        for source, target, cur_src_doc, cur_tgt_doc, docid, align in zip(srcs, tgts, src_docs, tgt_docs, docids, alignments):
            if prev_docid is None or docid != prev_docid:
                prev_docid = docid
                source_context = []
                target_context = []
                align_context = []
                cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))

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
            
            #assert len(cur_src_doc) == len(source.split(" "))
            #assert len(cur_tgt_doc) == len(target.split(" "))

            lexical_tags, cohesion_words = tagger.lexical_cohesion(cur_src_doc, cur_tgt_doc, align, cohesion_words)
            formality_tags = tagger.formality_tags(source, cur_src_doc, target, cur_tgt_doc, align)
            verb_tags = tagger.verb_form(cur_tgt_doc)
            pronouns_tags = tagger.pronouns(source, target, align)
            ellipsis_tags = tagger.ellipsis(cur_tgt_doc, align)
            posmorph_tags = tagger.pos_morph(target, cur_tgt_doc)
            polysemous_tags = tagger.polysemous(cur_src_doc, target, align, polysemous_words)
            #ner_tags = tagger.ner(detok_tgt_doc)
            tags = []
            for i in range(len(lexical_tags)):
                tag = ["all"]
                if pronouns_tags[i]:
                    tag.append("pronouns")
                if formality_tags[i]:
                    tag.append("formality")
                if verb_tags[i]:
                    tag.append("verb_tense")
                if ellipsis_tags[i]:
                    tag.append("ellipsis")
                if lexical_tags[i]:
                    tag.append("lexical")
                if polysemous_tags[i]:
                    tag.append("polysemous")
                if len(tag) == 1:
                    tag.append("no_tag")
                tag.append(posmorph_tags[i])
                tags.append("+".join(tag))

            print(" ".join(tags), file=output_file)

            source_context.append(source)
            target_context.append(target)


if __name__ == "__main__":
    main()
