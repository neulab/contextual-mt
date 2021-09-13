import torch
import numpy as np

from collections import defaultdict

from fairseq.data import data_utils, FairseqDataset


def collate(samples, pad_id, eos_id, sort_by_src=False):
    """creates a batch out of a list of samples to be feed to a contextual_mt model"""
    id = torch.LongTensor([s["id"] for s in samples])
    # encode source and source context
    src_tokens = data_utils.collate_tokens(
        [s["source"] for s in samples],
        pad_id,
        eos_id,
    )
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_id).long().sum() for s in samples]
    )
    src_ctx_tokens = data_utils.collate_tokens(
        [s["src_context"] for s in samples],
        pad_id,
        eos_id,
    )
    src_ctx_lengths = torch.LongTensor(
        [s["src_context"].ne(pad_id).long().sum() for s in samples]
    )
    # encode target and target context
    tgt_ctx_tokens = data_utils.collate_tokens(
        [s["tgt_context"] for s in samples], pad_id, eos_id, move_eos_to_beginning=True
    )
    tgt_ctx_lengths = torch.LongTensor(
        [s["tgt_context"].ne(pad_id).long().sum() for s in samples]
    )
    if sort_by_src:
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        src_ctx_tokens = src_ctx_tokens.index_select(0, sort_order)
        src_ctx_lengths = src_ctx_lengths.index_select(0, sort_order)
        tgt_ctx_tokens = tgt_ctx_tokens.index_select(0, sort_order)
        tgt_ctx_lengths = tgt_ctx_lengths.index_select(0, sort_order)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "src_ctx_tokens": src_ctx_tokens,
            "src_ctx_lengths": src_ctx_lengths,
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "tgt_ctx_tokens": tgt_ctx_tokens,
            "tgt_ctx_lengths": tgt_ctx_lengths,
        },
    }

    if samples[0].get("target", None) is not None:
        tgt_tokens = data_utils.collate_tokens(
            [s["target"] for s in samples],
            pad_id,
            eos_id,
        )
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_id).long().sum() for s in samples]
        )

        prev_output_tokens = data_utils.collate_tokens(
            [s["target"] for s in samples],
            pad_id,
            eos_id,
            move_eos_to_beginning=True,
        )
        # encode target and target context
        tgt_ctx_tokens_out = data_utils.collate_tokens(
            [s["tgt_context"] for s in samples], pad_id, eos_id
        )

        if sort_by_src:
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
            tgt_tokens = tgt_tokens.index_select(0, sort_order)
            tgt_ctx_tokens_out = tgt_ctx_tokens_out.index_select(0, sort_order)

        ntokens = tgt_lengths.sum().item()
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens
        batch["target"] = tgt_tokens
        batch["context_target"] = tgt_ctx_tokens_out
    else:
        ntokens = src_lengths.sum().item()

    if samples[0].get("src_sample_probs", None) is not None:
        src_sample_probs = data_utils.collate_tokens(
            [s["src_sample_probs"] for s in samples],
            pad_idx=0.0,
        )
        if sort_by_src:
            src_sample_probs.index_select(0, sort_order)

        batch["net_input"]["src_sample_probs"] = src_sample_probs

    batch["ntokens"] = ntokens
    return batch


class ContextualDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets with a contextual structure

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset): target dataset to wrap
        tgt_sizes (List[int]): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        contextual_ids (List[int]): List of indices associating every sample
            to a "document"
        source_context_size (int): the number o sentences to pass in the source
            context
        target_context_size (int): the number of sentences to pass in the target
            context
        pos_drop_probs: NOT USED
        src_pos_targ: NOT USED
        sampled_context_size (bool): if set, context sizes will be sampled in
            a between 0 and `src/tgt_ctx_size` (default: False)
        break_tag: token used to separate context sentences
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt,
        tgt_sizes,
        tgt_dict,
        contextual_ids,
        src_ctx_size=0,
        tgt_ctx_size=0,
        pos_drop_probs=None,
        src_pos_tags=None,
        break_tag=None,
        sample_context_size=False,
        shuffle=True,
    ):
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        assert len(src) == len(
            tgt
        ), "Source and target must contain the same number of examples"

        self.src = src
        self.tgt = tgt
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_ctx_size = src_ctx_size
        self.tgt_ctx_size = tgt_ctx_size
        self.contextual_ids = np.array(contextual_ids)
        self.break_tag = break_tag
        self.sample_context_size = sample_context_size
        self.shuffle = shuffle

        # recompute sizes  based on context size and special tokens
        full_src_sizes, full_tgt_sizes = [], []
        for i, size in enumerate(src_sizes):
            for j in range(1, self.src_ctx_size + 1):
                if self.contextual_ids[i - j] != self.contextual_ids[i]:
                    break
                size += src_sizes[i - j] + 1
            full_src_sizes.append(size + 1)
        # FIXME: if target context is part of input, this needs to be rethinked
        for i, size in enumerate(tgt_sizes):
            for j in range(1, self.tgt_ctx_size + 1):
                if self.contextual_ids[i - j] != self.contextual_ids[i]:
                    break
                size += tgt_sizes[i - j] + 1
            full_tgt_sizes.append(size + 1)

        self.src_sizes = np.array(full_src_sizes)
        self.tgt_sizes = np.array(full_tgt_sizes)

        # NOTE: not used in the paper
        if pos_drop_probs is not None:
            self.pos_drop_probs = defaultdict(lambda: 0.0)
            for pos, p in pos_drop_probs.items():
                self.pos_drop_probs[pos] = p
        else:
            self.pos_drop_probs = None
        self.src_pos_tags = src_pos_tags

    def __getitem__(self, index):
        # remove included eos token
        src_item = self.src[index][:-1]
        tgt_item = self.tgt[index][:-1]
        src_ctx_item = torch.tensor([]).long()
        tgt_ctx_item = torch.tensor([]).long()
        src_break_id = torch.tensor([self.src_dict.index(self.break_tag)])
        tgt_break_id = torch.tensor([self.tgt_dict.index(self.break_tag)])
        if self.src_ctx_size > 0:
            if self.sample_context_size:
                src_context_size = np.random.randint(0, self.src_ctx_size + 1)
            else:
                src_context_size = self.src_ctx_size

            for i in range(1, src_context_size + 1):
                # break if previous sample is from a different context (doc/chat)
                if self.contextual_ids[index - i] != self.contextual_ids[index]:
                    break
                # add break tag if passed
                if len(src_ctx_item) > 0 and self.break_tag is not None:
                    src_ctx_item = torch.cat([src_break_id, src_ctx_item])

                src_ctx_item = torch.cat([self.src[index - i][:-1], src_ctx_item])

        if self.tgt_ctx_size > 0:
            if self.sample_context_size:
                tgt_context_size = np.random.randint(0, self.tgt_ctx_size + 1)
            else:
                tgt_context_size = self.tgt_ctx_size

            for i in range(1, tgt_context_size + 1):
                if self.contextual_ids[index - i] != self.contextual_ids[index]:
                    break
                if len(tgt_ctx_item) > 0 and self.break_tag is not None:
                    tgt_ctx_item = torch.cat([tgt_break_id, tgt_ctx_item])

                tgt_ctx_item = torch.cat([self.tgt[index - i][:-1], tgt_ctx_item])

        src_eos_id = torch.Tensor([self.src_dict.eos()]).long()
        tgt_eos_id = torch.Tensor([self.tgt_dict.eos()]).long()
        src_ctx_item = torch.cat([src_ctx_item, src_eos_id])
        tgt_ctx_item = torch.cat([tgt_ctx_item, tgt_eos_id])
        src_item = torch.cat([src_item, src_eos_id])
        tgt_item = torch.cat([tgt_item, tgt_eos_id])
        sample = {
            "id": index,
            "src_context": src_ctx_item,
            "source": src_item,
            "tgt_context": tgt_ctx_item,
            "target": tgt_item,
        }

        if self.src_pos_tags is not None and self.pos_drop_probs is not None:
            probs = []
            for pos in self.src_pos_tags[index]:
                probs.append(self.pos_drop_probs[pos])
            sample["src_sample_probs"] = torch.tensor(probs)

        return sample

    def collater(self, samples):
        return collate(
            samples,
            self.src_dict.pad(),
            self.src_dict.eos(),
            sort_by_src=False,
        )

    def __len__(self):
        return len(self.src)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # FIXME: do something about the sample_context_size. currently it always
        # assumes the maximum context size, so this might lead to underusage of gpu
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        # sort by target length, then source length
        indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.
        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)
        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        # FIXME: add context sizes
        if max_sizes is None:
            return indices, []
        if type(max_sizes) in (int, float):
            max_src_size, max_tgt_size = max_sizes, max_sizes
        else:
            max_src_size, max_tgt_size = max_sizes

        ignored = indices[
            (self.src_sizes[indices] > max_src_size)
            | (self.tgt_sizes[indices] > max_tgt_size)
        ]
        if len(ignored) > 0:
            indices = indices[
                (self.src_sizes[indices] <= max_src_size)
                & (self.tgt_sizes[indices] <= max_tgt_size)
            ]
        return indices, ignored.tolist()
