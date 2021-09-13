from fairseq import utils
from fairseq.tasks import register_task
from fairseq.data import indexed_dataset, data_utils, MultiCorpusSampledDataset
from collections import OrderedDict
import random

import os

from contextual_mt import ContextualDataset, HighlightedDataset, DocumentTranslationTask


@register_task("attention_regularization")
class AttentionRegularizationTask(DocumentTranslationTask):
    @staticmethod
    def add_args(parser):
        DocumentTranslationTask.add_args(parser)
        # attn reg options
        parser.add_argument(
            "--regularize-heads",
            default=None,
            type=int,
            help="if set, regularize the i-th attention head using highlighted data",
        )
        parser.add_argument(
            "--regularize-attention",
            default=["self", "cross", "enc"],
            nargs="+",
            type=str,
            help="specify which attention to regularize",
        )
        parser.add_argument(
            "--enc-alignment-layer",
            default=[0],
            nargs="+",
            type=int,
            help="specify which encoder layer to regularize",
        )
        parser.add_argument(
            "--dec-alignment-layer",
            default=[5],
            nargs="+",
            type=int,
            help="specify which decoder layer to regularize",
        )
        parser.add_argument(
            "--self-alignment-layer",
            default=None,
            nargs="+",
            type=int,
            help="specify which decoder layer to regularize",
        )
        parser.add_argument(
            "--cross-alignment-layer",
            default=None,
            nargs="+",
            type=int,
            help="specify which decoder layer to regularize",
        )
        parser.add_argument(
            "--highlight-sample",
            default=None,
            type=float,
            help="probability to sample highlighted data during training",
        )
        parser.add_argument(
            "--kl-lambda",
            default=1,
            type=float,
            help="lambda weight term for attention KL div loss",
        )
        parser.add_argument("--highlight-on-tag", default="<hon>", type=str)
        parser.add_argument("--highlight-off-tag", default="<hoff>", type=str)
        parser.add_argument("--src-word-on-tag", default="<p>", type=str)
        parser.add_argument("--src-word-off-tag", default="</p>", type=str)

    def load_dataset(self, split, epoch=1, combine=False, shuffle=True, **kwargs):
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
        with open(prefix + "docids", "r", encoding="utf-8") as f:
            doc_ids = [idx for idx in f]  # allow string

        pos_tags = None
        if split == "train" and os.path.exists(f"{prefix}pos.{src}"):
            with open(f"{prefix}pos.{src}", "r", encoding="utf-8") as f:
                pos_tags = [line.strip().split(" ") for line in f]
        pos_drop_probs = None
        if self.args.pos_drop_probs is not None:
            pos_drop_probs = {
                p.split(":")[0]: float(p.split(":")[1])
                for p in self.args.pos_drop_probs
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
            shuffle=shuffle,
        )

        if (self.args.regularize_heads is not None) and (split == "train"):
            # Load highlighted data
            split_path = f"highlighted.{split}"
            if split_exists(split_path, src, tgt, src, data_path):
                prefix = os.path.join(
                    data_path, "{}.{}-{}.".format(split_path, src, tgt)
                )
            elif split_exists(split_path, tgt, src, src, data_path):
                prefix = os.path.join(
                    data_path, "{}.{}-{}.".format(split_path, tgt, src)
                )
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split_path, data_path)
                )

            h_src_dataset = data_utils.load_indexed_dataset(
                prefix + src, self.src_dict, self.args.dataset_impl
            )

            h_tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, self.tgt_dict, self.args.dataset_impl
            )

            split_path = f"highlighted.{split}.context"

            if split_exists(split_path, src, tgt, src, data_path):
                prefix = os.path.join(
                    data_path, "{}.{}-{}.".format(split_path, src, tgt)
                )
            elif split_exists("highlighted.context", tgt, src, src, data_path):
                prefix = os.path.join(
                    data_path, "{}.{}-{}.".format(split_path, tgt, src)
                )
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split_path, data_path)
                )
            h_src_ctx_dataset = data_utils.load_indexed_dataset(
                prefix + src, self.src_dict, self.args.dataset_impl
            )
            h_tgt_ctx_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, self.tgt_dict, self.args.dataset_impl
            )

            contra = True if split == "test" else False

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
                p_tag=self.args.src_word_on_tag,
                p2_tag=self.args.src_word_off_tag,
                shuffle=shuffle,
                contrastive=contra,
            )

            def sampler(x, p=self.args.highlight_sample):
                if random.random() > p:
                    return x[1]
                else:
                    return x[0]

            self.datasets[split] = MultiCorpusSampledDataset(
                OrderedDict({"highlighted": highlighted_data, "main": main_data}),
                sampler,
            )

        else:
            self.datasets[split] = main_data

    def load_highlighted(self, epoch=1, combine=False, **kwargs):
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
        paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        # Load highlighted data
        if split_exists("highlighted", src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format("highlighted", src, tgt)
            )
        elif split_exists("highlighted", tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format("highlighted", tgt, src)
            )
        else:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format("highlighted", data_path)
            )

        h_src_dataset = data_utils.load_indexed_dataset(
            prefix + src, self.src_dict, self.args.dataset_impl
        )
        h_tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, self.tgt_dict, self.args.dataset_impl
        )

        if split_exists("highlighted.context", src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format("highlighted.context", src, tgt)
            )
        elif split_exists("highlighted.context", tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, "{}.{}-{}.".format("highlighted.context", tgt, src)
            )
        else:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format("highlighted.context", data_path)
            )
        h_src_ctx_dataset = data_utils.load_indexed_dataset(
            prefix + src, self.src_dict, self.args.dataset_impl
        )
        h_tgt_ctx_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, self.tgt_dict, self.args.dataset_impl
        )

        self.datasets["highlighted"] = HighlightedDataset(
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
            p_tag=self.args.src_word_on_tag,
            p2_tag=self.args.src_word_off_tag,
            shuffle=True,
        )
