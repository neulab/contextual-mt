from argparse import Namespace
import json
import os
import logging

import torch

from fairseq import options, utils
from fairseq.tasks import register_task, FairseqTask
from fairseq.tasks.translation import TranslationTask
from fairseq.data import encoders, indexed_dataset, data_utils

from dialogue_mt import DialogueLangPairDataset
from .contrastive_lang_pair_dataset import ContrastiveDataset


logger = logging.getLogger(__name__)


@register_task("dialogue_translation")
class DialogueTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                        will be iterated upon during epochs in round-robin manner; \
                        however, valid and test data are always in the first directory to \
                        avoid the need for repeating them in all directories",
        )
#        parser.add_argument("--contra", default=None, type=str)
        parser.add_argument(
            "--left-pad-source",
            default="True",
            type=str,
            metavar="BOOL",
            help="pad the source on the left",
        )
        parser.add_argument(
            "--left-pad-target",
            default="False",
            type=str,
            metavar="BOOL",
            help="pad the target on the left",
        )
        parser.add_argument(
            "--max-source-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--upsample-primary",
            default=1,
            type=int,
            help="amount to upsample primary dataset",
        )
        parser.add_argument(
            "--truncate-source",
            action="store_true",
            default=False,
            help="truncate source to max-source-positions",
        )

        # options for reporting BLEU during validation
        parser.add_argument(
            "--eval-bleu", action="store_true", help="evaluation with BLEU scores"
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            action="store_true",
            default=False,
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing BLEU",
        )
        parser.add_argument(
            "--eval-bleu-args",
            type=str,
            metavar="JSON",
            help="generation args for BLUE scoring, "
            'e.g., \'{"beam": 4, "lenpen": 0.6}\'',
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            action="store_true",
            help="print sample generations during validation",
        )

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
            "--context-method",
            default="tag",
            type=str,
            help="how context is integrated, choose from 'tag', 'flag', 'encode'",
        )

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary, dictionary)
        self.bpe = encoders.build_bpe(args)
        self.tokenizer = encoders.build_tokenizer(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        # load dictionaries
        dictionary = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
        dictionary.add_symbol("<brk>", overwrite=False)
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Loads a dataset
        Args:
            split (str): the split to load (train/valid/test/contra)
        """

        def binarize(s, speaker=None):
            """ binarizes a sentence by applying bpe and tokenization and adding a speaker tag """
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.src_dict.encode_line(
                s, append_eos=False, add_if_not_exist=False
            ).long()
            if speaker is not None:
<<<<<<< HEAD
                spk_tensor = torch.Tensor([self.src_dict.index(speaker)]).long()
=======
                spk_tensor = torch.Tensor([self.src_dict.index(speaker)])
>>>>>>> 318651caeb1f1a1dd1455424cbb9a706c2205ad6
                tokens = torch.cat([spk_tensor, tokens])
            return tokens

        src_bin_file = os.path.join(self.args.data, f"{split}.src-tgt.src.bin")
        tgt_bin_file = os.path.join(self.args.data, f"{split}.src-tgt.tgt.bin")

        data_path = os.path.join(self.args.data, f"{split}.json")
        with open(data_path, "r") as f:
            chat_dict = json.load(f)

        src_ds = indexed_dataset.make_builder(src_bin_file, self.args.dataset_impl)
        tgt_ds = indexed_dataset.make_builder(tgt_bin_file, self.args.dataset_impl)

        # TODO: ids should be a fairseq dataset aswell
        ids = []
        for chat in chat_dict.values():
            for turn in chat:
                src = turn["source"]
                tgt = turn["target"]
                idx = turn["utteranceID"]
                speaker = turn["speaker"]
                src_ds.add_item(binarize(src, speaker=speaker))
                tgt_ds.add_item(binarize(tgt))
                ids.append(idx)

        src_idx_file = os.path.join(self.args.data, f"{split}.src-tgt.src.idx")
        tgt_idx_file = os.path.join(self.args.data, f"{split}.src-tgt.tgt.idx")
        src_ds.finalize(src_idx_file)
        tgt_ds.finalize(tgt_idx_file)

        src_ds = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, f"{split}.src-tgt.src"),
            self.src_dict,
            self.args.dataset_impl,
        )
        tgt_ds = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, f"{split}.src-tgt.tgt"),
            self.tgt_dict,
            self.args.dataset_impl,
        )

        # TODO: need to add speaker tags
        self.datasets[split] = DialogueLangPairDataset(
            src_ds,
            src_ds.sizes,
            self.src_dict,
            tgt_ds,
            tgt_ds.sizes,
            self.tgt_dict,
            ids,
            ctx_method=self.args.context_method,
            src_ctx_size=self.args.source_context_size,
            tgt_ctx_size=self.args.target_context_size,
            shuffle=True,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
        )
    def build_model(self, args):
        model = FairseqTask.build_model(self, args)
        if getattr(args, "eval_bleu", False):
            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        assert (
            prefix_tokens is None
        ), "dialogue translation task currently doesn't support prefix tokens"

        context = None
        if self.args.target_context_size > 0 and sample["target"] is not None:
            context, target, idxs = [], [], []
            for full_tgt in sample["target"]:
                # for each sample in the batch, strip the padding
                stripped_tgt = utils.strip_pad(full_tgt, pad=self.tgt_dict.pad())
                # find the latest <brk> token and split the target on it
                brk_indices = torch.nonzero(
                    stripped_tgt == self.tgt_dict.index("<brk>")
                )
                brk_index = brk_indices[-1][0] if len(brk_indices) > 0 else -1
                ctx, tgt = stripped_tgt[: brk_index + 1], stripped_tgt[brk_index + 1 :]
                # and redo the batch, saving the indexes used for splitting for later
                target.append(tgt)
                context.append(ctx)
                idxs.append(brk_index)

            # tensorize splitted target/context again and set new target
            context = data_utils.collate_tokens(
                context, self.tgt_dict.pad(), self.tgt_dict.eos()
            )
            target = data_utils.collate_tokens(
                target, self.tgt_dict.pad(), self.tgt_dict.eos()
            )
            sample["target"] = target
        else:
            context = prefix_tokens

        with torch.no_grad():
            batched_output = generator.generate(
                models,
                sample,
                prefix_tokens=context,
                constraints=constraints,
            )

        if self.args.target_context_size > 0:
            for brk_id, output in zip(idxs, batched_output):
                for beam in output:
                    # for each beam in each sample, remove the context and associated atributes
                    beam["tokens"] = beam["tokens"][brk_id + 1 :]
                    beam["attention"] = beam["attention"][brk_id + 1 :]
                    beam["positional_scores"] = beam["positional_scores"][brk_id + 1 :]

        return batched_output

    def load_contra(self, **kwargs):
        """Loads contrastive dataset
        """

        def binarize(s, speaker=None):
            """ binarizes a sentence by applying bpe and tokenization and adding a speaker tag """
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.src_dict.encode_line(
                s, append_eos=False, add_if_not_exist=False
            ).long()
            if speaker is not None:
                spk_tensor = torch.Tensor([self.src_dict.index(speaker)]).long()
                tokens = torch.cat([spk_tensor, tokens])
            return tokens

        with open(self.args.contra + ".context.src", 'r', encoding='utf-8') as file:
            src_cxt = file.read().splitlines() 
        with open(self.args.contra + ".context.trg", 'r', encoding='utf-8') as file:
            tgt_cxt = file.read().splitlines() 
        with open(self.args.contra + ".current.src", 'r', encoding='utf-8') as file:
            src = file.read().splitlines() 
        with open(self.args.contra + ".current.trg", 'r', encoding='utf-8') as file:
            tgt = file.read().splitlines() 

        src_bin_file = os.path.join(self.args.data, "contra.src.bin")
        tgt_bin_file = os.path.join(self.args.data, "contra.tgt.bin")
        src_ds = indexed_dataset.make_builder(src_bin_file, self.args.dataset_impl)
        tgt_ds = indexed_dataset.make_builder(tgt_bin_file, self.args.dataset_impl)
        c_src_bin_file = os.path.join(self.args.data, "contra.c_src.bin")
        c_tgt_bin_file = os.path.join(self.args.data, "contra.c_tgt.bin")
        c_src_ds = indexed_dataset.make_builder(c_src_bin_file, self.args.dataset_impl)
        c_tgt_ds = indexed_dataset.make_builder(c_tgt_bin_file, self.args.dataset_impl)

        for s, t, c_s, c_t in zip(src, tgt, src_cxt, tgt_cxt):
            src_ds.add_item(binarize(s, "<en>"))
            tgt_ds.add_item(binarize(t))
            c_src_ds.add_item(binarize(c_s, "<en>"))
            c_tgt_ds.add_item(binarize(c_t))

        src_idx_file = os.path.join(self.args.data, "contra.src.idx")
        tgt_idx_file = os.path.join(self.args.data, "contra.tgt.idx")
        src_ds.finalize(src_idx_file)
        tgt_ds.finalize(tgt_idx_file)
        c_src_idx_file = os.path.join(self.args.data, "contra.c_src.idx")
        c_tgt_idx_file = os.path.join(self.args.data, "contra.c_tgt.idx")
        c_src_ds.finalize(c_src_idx_file)
        c_tgt_ds.finalize(c_tgt_idx_file)

        src_ds = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, "contra.src"),
            self.src_dict,
            self.args.dataset_impl,
        )
        tgt_ds = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, "contra.tgt"),
            self.tgt_dict,
            self.args.dataset_impl,
        )
        c_src_ds = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, "contra.c_src"),
            self.src_dict,
            self.args.dataset_impl,
        )
        c_tgt_ds = data_utils.load_indexed_dataset(
            os.path.join(self.args.data, "contra.c_tgt"),
            self.tgt_dict,
            self.args.dataset_impl,
        )

        # TODO: need to add speaker tags
        self.datasets["contra"] = ContrastiveDataset(
            src_ds,
            src_ds.sizes,
            self.src_dict,
            tgt_ds,
            tgt_ds.sizes,
            self.tgt_dict,
            c_src_ds,
            c_src_ds.sizes,
            c_tgt_ds,
            c_tgt_ds.sizes,
            ctx_method=self.args.context_method,
            src_ctx_size=self.args.source_context_size,
            tgt_ctx_size=self.args.target_context_size,
            shuffle=True,
        )

    def contra_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            net_input = sample["net_input"]

            if 'src_tokens' in net_input:
                src_tokens = net_input['src_tokens']
                # length of the source text being the character length except EndOfSentence and pad
                src_lengths = (src_tokens.ne(self.src_dict.eos()) & src_tokens.ne(self.src_dict.pad())).long().sum(dim=1)
            elif 'source' in net_input:
                src_tokens = net_input['source']
                src_lengths = (
                    net_input['padding_mask'].size(-1) - net_input['padding_mask'].sum(-1)
                    if net_input['padding_mask'] is not None
                    else torch.tensor(src_tokens.size(-1)).to(src_tokens)
                )
            else:
                raise Exception('expected src_tokens or source in net input')

            tgt_tokens = sample['target']
            # length of the source text being the character length except EndOfSentence and pad
            tgt_lengths = (tgt_tokens.ne(self.tgt_dict.eos()) & tgt_tokens.ne(self.tgt_dict.pad())).long().sum(dim=1)

            max_len = tgt_lengths.max().item()

            encoder_outs = models.forward_encoder(net_input)
            tokens = (
                torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
            )  # +2 for eos and pad
            tokens[:, 0] = self.eos if bos_token is None else bos_token
            attn: Optional[Tensor] = None

            for step in range(max_len + 1):  # one extra step for EOS marker
                lprobs, avg_attn_scores = models.forward_decoder(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                    self.temperature,
                )
                print("lprobs ", lprobs.size)
                print("target ", tgt_tokens.size)
                raise ValueError

