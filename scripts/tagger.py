import re
import abc
import argparse
import spacy
import spacy_stanza
from collections import defaultdict
from allennlp.predictors.predictor import Predictor

en_tagger = spacy_stanza.load_pipeline("en", processors="tokenize,pos,lemma,depparse")

class Tagger(abc.ABC):
    """Abstact class that represent a tagger for a language"""

    def __init__(self):
        self.tagger = spacy.load("xx_ent_wiki_sm")
        self.formality_classes = {}
        self.ambiguous_pronouns = None
        self.ambiguous_verbform = []

    def _normalize(self, word):
        """default normalization"""
        return re.sub(r"^\W+|\W+$", "", word.lower())

    def formality_tags(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags):
        # TODO: inter-sentential especification needs to be added
        # this would go by checking if the formality already appeared in the context
        # by for example, passing a set of seen formalities in the previsous sentences
        # similar to what happens in lexical cohesion
        # NOTE: every language specific verb formality checker will have to do this aswell
        tags = []
        #formality_words = [v for vs in self.formality_classes.values() for v in vs]
        formality_classes = {word : formality for formality, words in self.formality_classes.items() for word in words}
        formality_words = list(formality_classes.keys())
        for word in cur_tgt.split(" "):
            word = self._normalize(word)
            if word in formality_words:
                if formality_classes[word] in prev_formality_tags:
                    tags.append(True)
                else:
                    tags.append(False)
                    prev_formality_tags.add(formality_classes[word])
            else:
                tags.append(False)
        assert len(tags) == len(cur_tgt.split(" "))

        try:
            tags2, prev_formality_tags = self.verb_formality(
                cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align
            )
            assert len(tags2) == len(cur_tgt.split(" "))
            return [a or b for a, b in zip(tags, tags2)], prev_formality_tags
        except:  # noqa: E722
            return tags, prev_formality_tags

    def lexical_cohesion(self, src_doc, tgt_doc, align, cohesion_words):
        src_lemmas = [
            t if not tok.is_stop and not tok.is_punct else None
            for tok in src_doc
            for t in tok.text.split(" ")
        ]
        tgt_lemmas = [
            t if not tok.is_stop and not tok.is_punct else None
            for tok in tgt_doc
            for t in tok.text.split(" ")
        ]
        tags = [False] * len(tgt_lemmas)

        tmp_cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))
        for s, t in align.items():
            src_lemma = src_lemmas[s]
            tgt_lemma = tgt_lemmas[t]
            if src_lemma is not None and tgt_lemma is not None:
                if cohesion_words[src_lemma][tgt_lemma] > 1:
                    tags[t] = True
                tmp_cohesion_words[src_lemma][tgt_lemma] += 1
        
        for src_lemma in tmp_cohesion_words.keys():
            for tgt_lemma in tmp_cohesion_words[src_lemma].keys():
                cohesion_words[src_lemma][tgt_lemma] += tmp_cohesion_words[src_lemma][tgt_lemma]

        return tags, cohesion_words

    def verb_form(self, cur_doc, verb_forms):
        # TODO: inter-sentential especification needs to be added
        # this would go by checking if a specific verb_form already appeared in the context
        # by for example, passing a set of seen verb_forms in the previsous sentences
        # similar to what happens in lexical cohesion
        # NOTE: every language specific verb formality checker will have to do this aswell
        tags = []
        for tok in cur_doc:
            tag = False
            if tok.pos_ == "VERB": 
                amb_verb_forms = [a for a in tok.morph.get("Tense") if a in self.ambiguous_verbform]
                for form in set(amb_verb_forms):
                    if form in verb_forms:
                        tag = True # Set tag to true if ambiguous form appeared before
                    else:
                        verb_forms.add(form) # Add ambiguous form to memory
            for _ in tok.text.split(" "):
                tags.append(tag)
        return tags, verb_forms

    def pronouns(self, src_doc, tgt_doc, align, has_ante):
        # TODO: inter-sentential especification needs to be added
        # this would go by adding a coreference resolution that would
        # check if the coreferent is part of the context rather than the current sentence
        src = [
            tok.text if not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        src_pos = [
            tok.pos_ if not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        tgt = [
            tok.text if not tok.is_punct else None
            for tok in tgt_doc
            for _ in tok.text.split(" ")
        ]
        tgt_pos = [
            tok.pos_ if not tok.is_punct else None
            for tok in tgt_doc
            for _ in tok.text.split(" ")
        ]
        tags = [False] * len(tgt)
        # if self.src_neutral_pronouns is None or self.tgt_gendered_pronouns is None:
        if self.ambiguous_pronouns is None:
            return tags
        for s, r in align.items():
            # if self._normalize(src[s]) in self.src_neutral_pronouns:
            #     if self._normalize(ref[r]) in self.tgt_gendered_pronouns:
            if s > len(src):
                print(f"IndexError{s}: {src}")
            if r > len(tgt):
                print(f"IndexError{r}: {tgt}")
            if (
                not has_ante[s]
                and src_pos[s] == "PRON"
                and tgt_pos[r] == "PRON"
                and self._normalize(tgt[r])
                in self.ambiguous_pronouns.get(self._normalize(src[s]), [])
            ):
                tags[r] = True
        return tags

    def ellipsis(self, src, ref, align, verbs, nouns, ellipsis_sent):
        # src_pos = "+".join([tok.pos_ for tok in src for _ in tok.text.split(" ")])
        # src_text = [
        #     self._normalize(tok.text) for tok in src for _ in tok.text.split(" ")
        # ]
        ref = [tok for tok in ref for _ in tok.text.split(" ")]
        tags = [False] * len(ref)

        for i, tok in enumerate(ref):
            if (
                not tok.is_stop
                and tok.pos_ == "VERB"
                and len(self._normalize(tok.text)) > 1
            ):  # VP ellipsis
                # if i not in align.values() and (("AUX" in src_pos and "AUX+VERB" not in src_pos) or "to" in src_text) and tok.lemma_ in verbs:
                if i not in align.values() and tok.lemma_ in verbs and ellipsis_sent:
                    tags[i] = True
                verbs.add(tok.lemma_)
            if (
                not tok.is_stop
                and tok.pos_ in ["PRON", "PROPN", "NOUN"]
                and len(self._normalize(tok.text)) > 1
            ):  # NP classifier ellipsis
                # if i not in align.values() and ref[i-1].pos_ == "NUM" and "NUM" in src_pos:
                if i not in align.values() and tok.lemma_ in nouns and ellipsis_sent:
                    tags[i] = True
                nouns.add(tok.lemma_)
        return tags, verbs, nouns

    def pos_morph(self, current, doc):
        tags = []
        for tok in doc:
            tag = tok.pos_
            if tag == "PRON":
                morph_tags = tok.morph.get("Person") + tok.morph.get("Number")
                for i in range(len(morph_tags)):
                    m_tag = ".".join(morph_tags[: i + 1])
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
        src_lemmas = [
            tok.lemma_ if not tok.is_stop and not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        src_poss = [
            tok.pos_ if not tok.is_stop and not tok.is_punct else None
            for tok in src_doc
            for _ in tok.text.split(" ")
        ]
        tags = [False] * len(target.split(" "))

        for s, t in align.items():
            if f"{src_lemmas[s]}.{src_poss[s]}" in polysemous_words:
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

    def formality_tags(
        self,
        cur_src,
        ctx_src,
        cur_tgt=None,
        ctx_tgt=None,
        cur_align=None,
        ctx_align=None,
    ):
        # TODO?
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
        # TODO?
        # return [False for _ in cur_src.split(" ")]
        return None


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
        # self.tgt_gendered_pronouns = ["il", "ils", "elle", "elles"]
        self.ambiguous_pronouns = {
            "it": ["il", "elle", "lui"],
            "they": ["ils", "elles"],
            "them": ["ils", "elles"],
            "you": ["tu", "vous", "on"],
            "we": ["nous", "on"],
            "this": ["celle", "ceci"],
            "that": ["celle", "celui"],
            "these": ["celles", "ceux"],
            "those": ["celles", "ceux"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Past"]

        from spacy.lang.fr.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        # self.tagger = spacy.load("fr_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "fr", processors="tokenize,pos,lemma,depparse"
        )


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

        # self.tgt_gendered_pronouns = ["ele", "ela", "eles", "elas"]
        self.ambiguous_pronouns = {
            "this": ["este", "esta", "esse", "essa"],
            "that": ["este", "esta", "esse", "essa"],
            "these": ["estes", "estas", "esses", "essas"],
            "those": ["estes", "estas", "esses", "essas"],
            "it": ["ele", "ela", "o", "a"],
            "they": ["eles", "elas"],
            "them": ["eles", "elas", "os", "as"],
        }
        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pqp"]

        # self.tagger = spacy.load("pt_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "pt", processors="tokenize,pos,lemma,depparse"
        )


class GermanTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"du"},
            "v_class": {"sie"},  # formal 2nd person Sie is usually capitalized
        }
        from spacy.lang.de.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["er", "sie", "es"]
        self.ambiguous_pronouns = {
            "it": ["er", "sie", "es"],
        }
        # self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "de", processors="tokenize,pos,lemma,depparse"
        )


class SpanishTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: usted/su/sus/suyo/suya works for V class and 3rd person
        self.formality_classes = {
            "t_class": {"tú", "tu", "tus", "ti", "contigo", "tuyo", "te", "tuya"},
            "v_class": {"usted", "vosotros", "vuestro", "vuestra", "vuestras", "os"},
        }
        from spacy.lang.es.stop_words import STOP_WORDS

        # self.tgt_gendered_pronouns = ["él", "ella", "ellos", "ellas"]
        self.ambiguous_pronouns = {
            "it": ["él", "ella"],
            "they": ["ellos", "ellas"],
            "them": ["ellos", "ellas"],
            "this": ["ésta", "éste", "esto"],
            "that": ["esa", "ese"],
            "these": ["estos", "estas"],
            "those": ["aquellos", "aquellas", "ésos", "ésas"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        self.stop_words = STOP_WORDS
        # self.tagger = spacy.load("es_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "es", processors="tokenize,pos,lemma,depparse"
        )

    def verb_formality(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags):

        cur_src = cur_src.split(" ")
        cur_tgt = cur_tgt.split(" ")
        tags = [False] * len(cur_tgt)

        align = {
            self._normalize(cur_src[s]): self._normalize(cur_tgt[t])
            for s, t in cur_align.items()
        }
        you_verbs = []
        for tok in cur_src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))

        for i, tok in enumerate(cur_tgt_doc):
            if self._normalize(tok.text) in you_verbs:
                person = tok.morph.get("Person")
                if "2" in person:
                    if "2" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("2")
                elif "3" in person:
                    if "3" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("3")
        return tags, prev_formality_tags


class HebrewTagger(Tagger):
    def __init__(self):
        super().__init__()
        # TODO: hebrew has t-v distinction only in extreme formality cases

        from spacy.lang.he.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]
        self.tagger = spacy_stanza.load_pipeline(
            "he", processors="tokenize,pos,lemma,depparse"
        )


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
        # self.tagger = spacy.load("nl_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "nl", processors="tokenize,pos,lemma,depparse"
        )


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

        # self.tgt_gendered_pronouns = ["el", "ei", "ea", "ele"]
        self.ambiguous_pronouns = {
            "it": ["el", "ea"],
            "they": ["ei", "ele"],
            "them": ["ei", "ele"],
        }
        self.ambiguous_verbform = ["Past", "Imp", "Fut"]
        self.stop_words = STOP_WORDS
        # self.tagger = spacy.load("ro_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "ro", processors="tokenize,pos,lemma,depparse"
        )


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
        self.tagger = spacy_stanza.load_pipeline(
            "tr", processors="tokenize,pos,lemma,depparse"
        )


class ArabicTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.ar.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_pronouns = {
            "you": [
                "انت",
                "انتَ",
                "انتِ",
                "انتى",
                "أنتم",
                "أنتن",
                "انتو",
                "أنتما",
                "أنتما",
            ],
            "it": ["هو", "هي"],
            "they": ["هم", "هن", "هما"],
            "them": ["هم", "هن", "هما"],
        }
        self.tagger = spacy_stanza.load_pipeline(
            "ar", processors="tokenize,pos,lemma,depparse"
        )


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
            "them": ["ellos", "ellas"],
            "this": ["questa", "questo"],
            "that": ["quella", "quello"],
            "these": ["queste", "questi"],
            "those": ["quelle", "quelli"],
        }
        self.ambiguous_verbform = ["Pqp", "Imp", "Fut"]

        # self.tagger = spacy.load("it_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "it", processors="tokenize,pos,lemma,depparse"
        )

    def verb_formality(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags):

        cur_src = cur_src.split(" ")
        cur_tgt = cur_tgt.split(" ")
        tags = [False] * len(cur_tgt)

        align = {
            self._normalize(cur_src[s]): self._normalize(cur_tgt[t])
            for s, t in cur_align.items()
        }
        you_verbs = []
        for tok in cur_src_doc:
            if self._normalize(tok.text) == "you" and tok.dep_ == "nsubj":
                you_verbs.append(self._normalize(tok.head.text))
        you_verbs = list(map(lambda x: align.get(x), you_verbs))

        for i, tok in enumerate(cur_tgt_doc):
            if self._normalize(tok.text) in you_verbs:
                person = tok.morph.get("Person")
                if "2" in person:
                    if "2" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("2")
                elif "3" in person:
                    if "3" in prev_formality_tags:
                        try:
                            tags[cur_tgt.index(tok.text)] = True
                        except IndexError:
                            pass
                    else:
                        prev_formality_tags.add("3")
        return tags


class KoreanTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"제가", "저희", "나"},
            "v_class": {
                "댁에",
                "성함",
                "분",
                "생신",
                "식사",
                "연세",
                "병환",
                "약주",
                "자제분",
                "뵙다",
                "저",
            },
        }

        from spacy.lang.ko.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "ko", processors="tokenize,pos,lemma,depparse"
        )

    def verb_formality(self, cur_src, cur_src_doc, cur_tgt, cur_tgt_doc, cur_align, prev_formality_tags):
        tags = []
        for tok in cur_tgt_doc:
            honorific = False
            if tok.pos_ == "VERB":
                for suffix in [
                    "어",
                    "아",
                    "여",
                    "요",
                    "ㅂ니다",
                    "습니다",
                    "었어",
                    "았어",
                    "였어",
                    "습니다",
                    "겠어",
                    "습니다",
                ]:
                    if tok.text.endswith(suffix):
                        honorific = True
                        break
            for _ in tok.text.split(" "):
                if honorific:
                    if "honorific" in prev_formality_tags: # TODO for Korean specific
                        tags.append(True)
                    else:
                        tags.append(False)
                        prev_formality_tags.add("honorific")
                else:
                    tags.append(False)

        return tags, prev_formality_tags


class JapaneseTagger(Tagger):
    def __init__(self):
        super().__init__()
        # Formality verb forms from https://www.aclweb.org/anthology/D19-5203.pdf adapted to stanza tokens
        self.formality_classes = {
            "t_class": {"だ", "だっ", "じゃ", "だろう", "だ", "だけど", "だっ"},
            "v_class": {
                "ござい",
                "ます",
                "いらっしゃれ",
                "いらっしゃい",
                "ご覧",
                "伺い",
                "伺っ",
                "存知",
                "です",
                "まし",
            },
        }

        from spacy.lang.ja.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        # self.tgt_gendered_pronouns = ["私", "僕", "俺"]
        self.ambiguous_pronouns = {
            "i": ["私", "僕", "俺"],
        }
        # self.tagger = spacy.load("ja_core_news_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "ja", processors="tokenize,pos,lemma,depparse"
        )


class ChineseTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS

        self.formality_classes = {
            "t_class": {"你"},
            "v_class": {"您"},
        }
        # self.tagger = spacy.load("zh_core_web_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )


class TaiwaneseTagger(Tagger):
    def __init__(self):
        super().__init__()

        from spacy.lang.zh.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.tagger = spacy_stanza.load_pipeline(
            "zh", processors="tokenize,pos,lemma,depparse"
        )


class RussianTagger(Tagger):
    def __init__(self):
        super().__init__()
        self.formality_classes = {
            "t_class": {"ты", "тебя", "тебе", "тобой", "твой", "твоя", "твои", "тебе"},
            "v_class": {"вы", "вас", "вам", "вами", "ваш", "ваши"},
        }
        from spacy.lang.ru.stop_words import STOP_WORDS

        self.stop_words = STOP_WORDS
        self.ambiguous_verbform = ["Past"]
        # self.tagger = spacy.load("ru_core_web_sm")
        self.tagger = spacy_stanza.load_pipeline(
            "ru", processors="tokenize,pos,lemma,depparse"
        )


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
    parser.add_argument("--src-tok-file", required=True, help="")
    parser.add_argument("--tgt-tok-file", required=True, help="")
    parser.add_argument("--src-detok-file", required=True, help="")
    parser.add_argument("--tgt-detok-file", required=True, help="")
    parser.add_argument(
        "--ellipsis-file",
        default="/projects/tir4/users/kayoy/contextual-mt/data-with-de/ellipsis-nofrag.en",
        help="file with source ellipsis bools",
    )
    parser.add_argument(
        "--ellipsis-filt-file",
        default="/projects/tir4/users/kayoy/contextual-mt/data-with-de/ellipsis.manual.en",
        help="file with source ellipsis bools",
    )
    parser.add_argument("--docids-file", required=True, help="file with document ids")
    parser.add_argument(
        "--alignments-file", required=True, help="file with word alignments"
    )
    # parser.add_argument("--polysemous-file", required=True, help="file with polysemous words")
    parser.add_argument("--source-lang", default=None)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--source-context-size", type=int, default=None)
    parser.add_argument("--target-context-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.src_tok_file, "r", encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f]
    with open(args.tgt_tok_file, "r", encoding="utf-8") as tgt_f:
        tgts = [line.strip() for line in tgt_f]
    with open(args.ellipsis_file, "r", encoding="utf-8") as file:
        ellipsis_sents = [line.split("|||")[0].strip() == "True" for line in file]
    with open(args.ellipsis_filt_file, "r", encoding="utf-8") as file:
        ellipsis_sents_filt = [line.split("|||")[0].strip() == "True" for line in file]
    with open(args.src_detok_file, "r", encoding="utf-8") as src_f:
        detok_srcs = [line.strip() for line in src_f]
    with open(args.tgt_detok_file, "r", encoding="utf-8") as tgt_f:
        detok_tgts = [line.strip() for line in tgt_f]
    with open(args.docids_file, "r", encoding="utf-8") as docids_f:
        docids = [idx for idx in docids_f]
    with open(args.alignments_file, "r", encoding="utf-8") as file:
        alignments = file.readlines()

    alignments = list(
        map(
            lambda x: dict(
                list(map(lambda y: list(map(int, y.split("-"))), x.strip().split(" ")))
            ),
            alignments,
        )
    )

    tagger = build_tagger(args.target_lang)
    en_coref = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

    src_docs = en_tagger.pipe(detok_srcs)
    tgt_docs = tagger.tagger.pipe(detok_tgts)

    prev_docid = None
    with open(args.output, "w", encoding="utf-8") as output_file:
        for (
            source,
            target,
            ellipsis_sent,
            ellipsis_sent_filt,
            cur_src_doc,
            cur_tgt_doc,
            docid,
            align,
        ) in zip(
            srcs,
            tgts,
            ellipsis_sents,
            ellipsis_sents_filt,
            src_docs,
            tgt_docs,
            docids,
            alignments,
        ):
            if prev_docid is None or docid != prev_docid:
                prev_docid = docid
                source_context = []
                target_context = []
                # align_context = []
                cohesion_words = defaultdict(lambda: defaultdict(lambda: 0))
                prev_formality_tags = set()
                verb_forms = set()
                verbs = set()
                nouns = set()
                verbs_filt = set()
                nouns_filt = set()

            # TODO: this might be used for V2?
            # current_src_ctx = source_context[
            #     len(source_context) - args.source_context_size :
            # ]
            # current_tgt_ctx = target_context[
            #     len(target_context) - args.target_context_size :
            # ]
            # current_align_ctx = align_context[
            #     len(align_context)
            #     - max(args.source_context_size, args.target_context_size) :
            # ]

            
            coref = en_coref.predict(document=source)
            assert len(source.split(" ")) == len(coref["document"])
            has_ante = [False for _ in range(len(coref["document"]))]
            try:
                for cluster in coref["clusters"]:
                    for mention in cluster[1:]:
                        for i in range(mention[0], mention[1] + 1):
                            has_ante[i] = True
            except:
                pass

            lexical_tags, cohesion_words = tagger.lexical_cohesion(
                cur_src_doc, cur_tgt_doc, align, cohesion_words
            )
            formality_tags, prev_formality_tags = tagger.formality_tags(
                source, cur_src_doc, target, cur_tgt_doc, align, prev_formality_tags
            )
            verb_tags, verb_forms = tagger.verb_form(cur_tgt_doc, verb_forms)
            pronouns_tags = tagger.pronouns(cur_src_doc, cur_tgt_doc, align, has_ante)
            ellipsis_tags, verbs, nouns = tagger.ellipsis(
                cur_src_doc, cur_tgt_doc, align, verbs, nouns, ellipsis_sent
            )
            ellipsis_tags_filt, verbs_filt, nouns_filt = tagger.ellipsis(
                cur_src_doc,
                cur_tgt_doc,
                align,
                verbs_filt,
                nouns_filt,
                ellipsis_sent_filt,
            )
            posmorph_tags = tagger.pos_morph(target, cur_tgt_doc)
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
                if ellipsis_tags_filt[i]:
                    tag.append("ellipsis_filt")
                if lexical_tags[i]:
                    tag.append("lexical")
                if len(tag) == 1:
                    tag.append("no_tag")
                tag.append(posmorph_tags[i])
                tags.append("+".join(tag))

            print(" ".join(tags), file=output_file)

            source_context.append(source)
            target_context.append(target)


if __name__ == "__main__":
    main()
