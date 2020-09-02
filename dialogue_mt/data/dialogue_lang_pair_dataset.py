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
        super().__init__(
            srcs, src_sizes, src_dict, tgts, tgt_sizes, tgt_dict, shuffle=shuffle
        )
        self.ids = ids
        self.src_ctx_size = src_ctx_size
        self.tgt_ctx_size = tgt_ctx_size

    def __getitem__(self, index):
        src_item = self.src[index]
        tgt_item = self.tgt[index]
        if self.src_ctx_size > 0:
            brk_id = torch.Tensor([self.src_dict.brk()]).long()
            for i in range(1, min(self.src_ctx_size, self.ids[index]) + 1):
                src_item = torch.cat([self.src[index - i], brk_id, src_item])
        if self.tgt_ctx_size > 0:
            brk_id = torch.Tensor([self.tgt_dict.brk()]).long()
            for i in range(1, min(self.tgt_ctx_size, self.ids[index]) + 1):
                tgt_item = torch.cat([self.tgt[index - i], brk_id, tgt_item])

        return {"id": index, "source": src_item, "target": tgt_item}
