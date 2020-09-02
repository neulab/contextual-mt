import json
import os
import tempfile

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import encoders, indexed_dataset

from dialogue_mt.data import DialogueLangPairDataset


@register_task("dialogue_translation_task")
class DialogueTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--source-context-size",
            default=0,
            type=int,
            help="number of previous source sentences/messages to include in the context",
        )
        parser.add_argument(
            "--target-context-size",
            default=0,
            type=int,
            help="number of previous target sentences/messages to include in the context",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.bpe = encoders.build_bpe(args)

    def build_dataset(self, split, epoch=1, combine=False, **kwargs):
        def binarize(s, vocab):
            """ binarizes a sentence by applying bpe and tokenization """
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = vocab.encode_line(s, append_eos=False).long()
            return tokens

        data_path = os.path.join(self.args.data, f"{split}.json")
        with open(data_path, "r") as f:
            chat_dict = json.load(f)

        src_tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        tgt_tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        src_ds = indexed_dataset.make_builder(src_tmp_file, self.args.dataset_imp)
        tgt_ds = indexed_dataset.make_builder(tgt_tmp_file, self.args.dataset_imp)

        # TODO: ids should be a fairseq dataset aswell
        ids = []
        for chat in chat_dict.values():
            for turn in chat:
                src = turn["source"]
                tgt = turn["target"]
                idx = turn["utteranceID"]
                src_ds.add_item(binarize(src, self.src_dict))
                tgt_ds.add_item(binarize(tgt, self.tgt_dict))
                ids.append(idx.long())

        # TODO: need to add speaker tags
        self.datasets[split] = DialogueLangPairDataset(
            src_ds,
            src_ds.sizes,
            self.src_dict,
            tgt_ds,
            tgt_ds.sizes,
            self.tgt_dict,
            ids,
            src_ctx_size=self.args.source_context_size,
            tgt_ctx_size=self.args.target_context_size,
            shuffle=True,
        )
