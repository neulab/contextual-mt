from argparse import Namespace

from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import indexed_dataset, data_utils, encoders, MultiCorpusSampledDataset

import os
import json

from dialogue_mt import ContextualDataset, ContextualSequenceGenerator, HighlightedDataset


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
        parser.add_argument(
            "--pos-drop-probs",
            nargs="+",
            type=str,
            help="",
        )
        parser.add_argument(
            "--regularize-heads",
            default=None,
            type=int,
            help="if set, regularize the i-th attention head using highlighted data"
        )
        parser.add_argument(
            "--highlight-sample",
            default=0.5,
            type=float,
            help="probability to sample highlighted data during training",
        )
        parser.add_argument(
            "--highlight-on-tag",
            default="<hon>",
            type=str
        )
        parser.add_argument(
            "--highlight-off-tag",
            default="<hoff>",
            type=str
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

        pos_tags = None
        if split == "train" and os.path.exists(f"{prefix}pos.{src}"):
            with open(f"{prefix}pos.{src}", "r") as f:
                pos_tags = [line.strip().split(" ") for line in f]
        pos_drop_probs = None
        if self.args.pos_drop_probs is not None:
            pos_drop_probs = {
                p.split(":")[0]: float(p.split(":")[1]) for p in self.args.pos_drop_probs
            }

        main_data = ContextualDataset(
                src_dataset,
                src_dataset.sizes,
                self.src_dict,
                tgt_dataset,
                tgt_dataset.sizes,
                self.tgt_dict,
                doc_ids,
                self.args.source_context_size,
                self.args.target_context_size,
                src_pos_tags=pos_tags,
                pos_drop_probs=pos_drop_probs,
                break_tag=self.args.break_tag,
                shuffle=True,
            )

        if (self.args.regularize_heads is not None) and (split == "train"):
            # Load highlighted data
            if split_exists('highlighted', src, tgt, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format('highlighted', src, tgt))
            elif split_exists('highlighted', tgt, src, src, data_path):
                prefix = os.path.join(data_path, "{}.{}-{}.".format('highlighted', tgt, src))
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format('highlighted', data_path)
                )

            h_src_dataset = data_utils.load_indexed_dataset(
                prefix + src, self.src_dict, self.args.dataset_impl
            )
            h_tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, self.tgt_dict, self.args.dataset_impl
            )
            h_src_ctx_dataset = data_utils.load_indexed_dataset(
                prefix + 'ctx.' + src, self.src_dict, self.args.dataset_impl
            )
            h_tgt_ctx_dataset = data_utils.load_indexed_dataset(
                prefix + 'ctx.' + tgt, self.tgt_dict, self.args.dataset_impl
            )

            highlighted_data = HighlightedDataset(
                h_src_dataset,
                h_src_dataset.sizes,
                self.src_dict,
                h_tgt_dataset,
                h_tgt_dataset.sizes,
                self.tgt_dict,
                h_src_ctx_dataset,
                h_src_ctx_dataset.sizes,
                h_tgt_ctx_dataset,
                h_tgt_ctx_dataset.sizes,
                break_tag=self.args.break_tag,
                hon_tag=self.args.highlight_on_tag,
                hoff_tag=self.args.highlight_off_tag,
                shuffle=True,
            )

            def sampler(x, p=self.args.highlight_sample):
                if random.random() > p:
                    return x[1]
                else:
                    return x[0]

            self.datasets[split] = MultiCorpusSampledDataset(OrderedDict({"highlighted": highlighted_data, "main": main_data}), sampler)

            
        else:
            self.datasets[split] = main_data
