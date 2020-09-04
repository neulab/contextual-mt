import torch
from fairseq.data import LanguagePairDataset


class DialogueLangPairDataset(LanguagePairDataset):
    def __init__(
        self,
        srcs,
        src_sizes,
        src_dict,
        tgts,
        tgt_sizes,
        tgt_dict,
        ids,
        src_ctx_size=0,
        tgt_ctx_size=0,
        shuffle=True,
    ):
        self.ids = ids
        self.src_ctx_size = src_ctx_size
        self.tgt_ctx_size = tgt_ctx_size

        # recompute sizes  based on context size and special tokens
        full_src_sizes, full_tgt_sizes = [], []
        for i, size in enumerate(src_sizes):
            for j in range(1, min(self.src_ctx_size, self.ids[i]) + 1):
                size += src_sizes[i - j] + 1
            full_src_sizes.append(size + 1)
        for i, size in enumerate(tgt_sizes):
            for j in range(1, min(self.tgt_ctx_size, self.ids[i]) + 1):
                size += tgt_sizes[i - j] + 1
            full_tgt_sizes.append(size + 1)

        super().__init__(
            srcs,
            torch.tensor(full_src_sizes),
            src_dict,
            tgts,
            torch.tensor(full_tgt_sizes),
            tgt_dict,
            shuffle=shuffle,
        )

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]
        if self.src_ctx_size > 0:
            brk_id = torch.Tensor([self.src_dict.index("<brk>")]).long()
            for i in range(1, min(self.src_ctx_size, self.ids[index]) + 1):
                src_item = torch.cat([self.src[index - i], brk_id, src_item])
        if self.tgt_ctx_size > 0:
            brk_id = torch.Tensor([self.src_dict.index("<brk>")]).long()
            for i in range(1, min(self.tgt_ctx_size, self.ids[index]) + 1):
                tgt_item = torch.cat([self.tgt[index - i], brk_id, tgt_item])

        eos_id = torch.Tensor([self.src_dict.eos()]).long()
        src_item = torch.cat([src_item, eos_id])
        tgt_item = torch.cat([tgt_item, eos_id])
        return {"id": index, "source": src_item, "target": tgt_item}
