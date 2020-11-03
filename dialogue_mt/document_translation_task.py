from argparse import Namespace

from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import indexed_dataset, data_utils, encoders

import os
import json

from dialogue_mt import ContextualDataset, ContextualSequenceGenerator


@register_task("document_translation")
class DocumentTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        # context options
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
        parser.add_argument(
            "--break-tag",
            default="<brk>",
            type=str,
            help="is set, separates context sentences by the break tag",
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args), seq_gen_cls=ContextualSequenceGenerator
            )
        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(
                data_path, "{}.{}-{}.{}".format(split, src, tgt, lang)
            )
            return indexed_dataset.dataset_exists(filename, impl=self.args.dataset_impl)

        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        if split_exists(split, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
        elif split_exists(split, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
        else:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, data_path)
            )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, self.src_dict, self.args.dataset_impl
        )
        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, self.tgt_dict, self.args.dataset_impl
        )
        with open(prefix + "docids", "r") as f:
            doc_ids = [int(idx) for idx in f]

        self.datasets[split] = ContextualDataset(
            src_dataset,
            src_dataset.sizes,
            self.src_dict,
            tgt_dataset,
            tgt_dataset.sizes,
            self.tgt_dict,
            doc_ids,
            self.args.source_context_size,
            self.args.target_context_size,
            break_tag=self.args.break_tag,
            shuffle=True,
        )
