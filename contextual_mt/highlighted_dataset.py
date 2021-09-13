import torch
import numpy as np

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

    src_ctx_highlights = data_utils.collate_tokens(
        [s["h_src_context"] for s in samples],
        0,
        0,
    )

    tgt_ctx_highlights = data_utils.collate_tokens(
        [s["h_tgt_context"] for s in samples],
        0,
        0,
    )

    source_highlights = data_utils.collate_tokens(
        [s["h_source"] for s in samples],
        0,
        0,
    )

    target_highlights = data_utils.collate_tokens(
        [s["h_target"] for s in samples],
        0,
        0,
    )

    src_words = data_utils.collate_tokens(
        [s["src_words_idx"] for s in samples],
        0,
        0,
    )

    tgt_words = data_utils.collate_tokens(
        [s["tgt_words_idx"] for s in samples],
        0,
        0,
    )

    batch["highlights"] = {
        "src_ctx_highlights": src_ctx_highlights,
        "tgt_ctx_highlights": tgt_ctx_highlights,
        "source_highlights": source_highlights,
        "target_highlights": target_highlights,
        "src_words": src_words,
        "tgt_words": tgt_words,
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

    batch["ntokens"] = ntokens
    return batch


def extract_highlights(data, hon_id, hoff_id, p_id=None, p_id2=None):

    tokens_data = []
    highlighted_data = []
    source_words_idx = []

    for tokens in data:
        highlighted = []
        high = 0
        source_word = 0
        to_delete = []
        words_idx = []
        for i, token in enumerate(tokens):
            if token == hon_id:
                high = 1
                to_delete.append(i)
            elif token == hoff_id:
                high = 0
                to_delete.append(i)
            elif token == p_id:
                source_word = 1
                to_delete.append(i)
            elif token == p_id2:
                source_word = 0
                to_delete.append(i)
            else:
                highlighted.append(high)
                words_idx.append(source_word)

        for i in to_delete[::-1]:
            tokens = torch.cat([tokens[:i], tokens[i + 1 :]])
        # For eos tag
        highlighted.append(0)
        words_idx.append(0)

        highlighted = torch.Tensor(highlighted)
        words_idx = torch.Tensor(words_idx)

        tokens_data.append(tokens)
        highlighted_data.append(highlighted)

        if p_id is not None:
            source_words_idx.append(words_idx)

    if p_id is not None:
        return tokens_data, highlighted_data, source_words_idx
    return tokens_data, highlighted_data


def extract_contrastive(data, hon_id, hoff_id, p_id=None, p_id2=None):

    tokens_data = []
    source_words_idx = []
    highlighted_data = []
    high = 0

    for tokens in data:
        highlighted = []
        to_delete = []
        words_idx = []
        for i, token in enumerate(tokens):
            if token == hon_id:
                high = 1
                to_delete.append(i)
            elif token == hoff_id:
                high = 0
                to_delete.append(i)
            elif token == p_id:
                source_word = 1
                to_delete.append(i)
            elif token == p_id2:
                source_word = 0
                to_delete.append(i)
            else:
                highlighted.append(high)
                words_idx.append(source_word)

        for i in to_delete[::-1]:
            tokens = torch.cat([tokens[:i], tokens[i + 1 :]])
        # For eos tag
        highlighted.append(0)
        words_idx.append(0)

        highlighted = torch.Tensor(highlighted)
        words_idx = torch.Tensor(words_idx)

        tokens_data.append(tokens)
        highlighted_data.append(highlighted)

        if p_id is not None:
            source_words_idx.append(words_idx)

    if p_id is not None:
        return tokens_data, highlighted_data, source_words_idx
    return tokens_data, highlighted_data


class HighlightedDataset(FairseqDataset):
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt,
        tgt_sizes,
        tgt_dict,
        ctx_src,
        ctx_src_sizes,
        ctx_tgt,
        ctx_tgt_sizes,
        break_tag=None,
        hon_tag=None,
        hoff_tag=None,
        p_tag=None,
        p2_tag=None,
        shuffle=True,
        contrastive=False,
    ):
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        assert len(src) == len(
            tgt
        ), "Source and target must contain the same number of examples"

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.break_tag = break_tag
        self.hon_tag = hon_tag
        self.hoff_tag = hoff_tag
        self.shuffle = shuffle
        self.contrastive = contrastive

        src_hon_id = self.src_dict.index(self.hon_tag)
        src_hoff_id = self.src_dict.index(self.hoff_tag)
        tgt_hon_id = self.tgt_dict.index(self.hon_tag)
        tgt_hoff_id = self.tgt_dict.index(self.hoff_tag)

        self.src, self.h_src, self.src_words_idx = extract_highlights(
            src,
            src_hon_id,
            src_hoff_id,
            self.src_dict.index(p_tag),
            self.src_dict.index(p2_tag),
        )
        self.tgt, self.h_tgt, self.tgt_words_idx = extract_highlights(
            tgt,
            tgt_hon_id,
            tgt_hoff_id,
            self.tgt_dict.index(p_tag),
            self.tgt_dict.index(p2_tag),
        )
        self.c_src, self.h_c_src = extract_highlights(ctx_src, src_hon_id, src_hoff_id)
        self.c_tgt, self.h_c_tgt = extract_highlights(ctx_tgt, tgt_hon_id, tgt_hoff_id)

        # recompute sizes  based on context size and special tokens
        full_src_sizes, full_tgt_sizes = [], []
        for i, size in enumerate(src_sizes):
            size += ctx_src_sizes[i] + 1
            full_src_sizes.append(size + 1)
        for i, size in enumerate(tgt_sizes):
            size += ctx_tgt_sizes[i] + 1
            full_tgt_sizes.append(size + 1)

        self.src_sizes = np.array(full_src_sizes)
        self.tgt_sizes = np.array(full_tgt_sizes)

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]
        src_ctx_item = self.c_src[index]
        tgt_ctx_item = self.c_tgt[index]
        h_src_item = self.h_src[index]
        h_tgt_item = self.h_tgt[index]
        h_src_ctx_item = self.h_c_src[index]
        h_tgt_ctx_item = self.h_c_tgt[index]
        src_words = self.src_words_idx[index]
        tgt_words = self.tgt_words_idx[index]

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
            "h_src_context": h_src_ctx_item,
            "h_source": h_src_item,
            "h_tgt_context": h_tgt_ctx_item,
            "h_target": h_tgt_item,
            "src_words_idx": src_words,
            "tgt_words_idx": tgt_words,
        }

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
        # FIXME: incoporate context size here
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # FIXME: incoporate context size here
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    @property
    def sizes(self):
        return self.src_sizes

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
