import torch
from fairseq.data import data_utils, LanguagePairDataset


class DialogueDataset(LanguagePairDataset):
    def __init__(
        self,
        srcs,
        src_sizes,
        tgts,
        tgt_sizes,
        dictionary,
        ids,
        src_ctx_size=0,
        tgt_ctx_size=0,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        concat_source_context=True,
        concat_target_context=True,
    ):
        self.ids = ids
        self.dictionary = dictionary
        self.src_ctx_size = src_ctx_size
        self.tgt_ctx_size = tgt_ctx_size
        self.concat_source_context = concat_source_context
        self.concat_target_context = concat_target_context

        # recompute sizes  based on context size and special tokens
        full_src_sizes, full_tgt_sizes = [], []
        for i, size in enumerate(src_sizes):
            for j in range(1, min(self.src_ctx_size, self.ids[i]) + 1):
                size += src_sizes[i - j] + 1
            full_src_sizes.append(size + 1)
        # FIXME: if target context is part of input, this needs to be rethinked
        for i, size in enumerate(tgt_sizes):
            for j in range(1, min(self.tgt_ctx_size, self.ids[i]) + 1):
                size += tgt_sizes[i - j] + 1
            full_tgt_sizes.append(size + 1)

        super().__init__(
            srcs,
            torch.tensor(full_src_sizes),
            dictionary,
            tgts,
            torch.tensor(full_tgt_sizes),
            dictionary,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            shuffle=shuffle,
        )

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]
        src_ctx_item = torch.tensor([]).long()
        tgt_ctx_item = torch.tensor([]).long()
        brk_id = torch.Tensor([self.dictionary.index("<brk>")]).long()
        if self.src_ctx_size > 0:
            for i in range(1, min(self.src_ctx_size, self.ids[index]) + 1):
                # preprend <brk> if ctx already has something
                if len(src_ctx_item) > 0:
                    src_ctx_item = torch.cat([brk_id, src_ctx_item])
                src_ctx_item = torch.cat([self.src[index - i], src_ctx_item])
        if self.tgt_ctx_size > 0:
            for i in range(1, min(self.tgt_ctx_size, self.ids[index]) + 1):
                # preprend <brk> if ctx already has something
                if len(tgt_ctx_item) > 0:
                    tgt_ctx_item = torch.cat([brk_id, tgt_ctx_item])
                tgt_ctx_item = torch.cat([self.tgt[index - i], tgt_ctx_item])

        eos_id = torch.Tensor([self.dictionary.eos()]).long()
        src_item = torch.cat([src_item, eos_id])
        tgt_item = torch.cat([tgt_item, eos_id])

        sample = {"id": index}
        if self.concat_source_context:
            if len(src_ctx_item) > 0:
                src_item = torch.cat([src_ctx_item, brk_id, src_item])
            sample["source"] = src_item
        else:
            sample["src_context"] = torch.cat([src_ctx_item, eos_id])
            sample["source"] = src_item
        if self.concat_target_context:
            if len(tgt_ctx_item) > 0:
                tgt_item = torch.cat([tgt_ctx_item, brk_id, tgt_item])
            sample["target"] = tgt_item
        else:
            sample["tgt_context"] = torch.cat([tgt_ctx_item, eos_id])
            sample["target"] = tgt_item

        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        if samples[0].get("src_context", None) is not None:
            src_context = data_utils.collate_tokens(
                [s["src_context"] for s in samples],
                self.dictionary.pad(),
                self.dictionary.eos(),
                self.left_pad_source,
            )
            src_ctx_lengths = torch.LongTensor(
                [
                    s["src_context"].ne(self.dictionary.pad()).long().sum()
                    for s in samples
                ]
            )

            batch["net_input"].update(
                {"src_context": src_context, "src_ctx_lengths": src_ctx_lengths}
            )
        if samples[0].get("tgt_context", None) is not None:
            tgt_context = data_utils.collate_tokens(
                [s["tgt_context"] for s in samples],
                self.dictionary.pad(),
                self.dictionary.eos(),
                self.left_pad_source,
            )
            tgt_ctx_lengths = torch.LongTensor(
                [
                    s["tgt_context"].ne(self.dictionary.pad()).long().sum()
                    for s in samples
                ]
            )
            batch["net_input"].update(
                {"tgt_context": tgt_context, "tgt_ctx_lengths": tgt_ctx_lengths}
            )
        return batch
