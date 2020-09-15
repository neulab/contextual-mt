import torch
from fairseq.data import LanguagePairDataset
from .collaters import Seq2SeqCollater


class ContrastiveDataset(LanguagePairDataset):
    def __init__(
        self,
        srcs,
        src_sizes,
        src_dict,
        tgts,
        tgt_sizes,
        tgt_dict,
        c_srcs,
        c_src_sizes,
        c_tgts,
        c_tgt_sizes,
        ctx_method="tag",
        src_ctx_size=0,
        tgt_ctx_size=0,
        shuffle=True,
    ):
        self.src_ctx_size = src_ctx_size
        self.tgt_ctx_size = tgt_ctx_size
        self.ctx_method = ctx_method
        self.c_srcs = c_srcs
        self.c_tgts = c_tgts
        # recompute sizes  based on context size and special tokens
        full_src_sizes, full_tgt_sizes = [], []
        for i, size in enumerate(src_sizes):
            for j in range(min(self.src_ctx_size, i) + 1):
                size += c_src_sizes[i - j] + 1
            full_src_sizes.append(size + 1)
        for i, size in enumerate(tgt_sizes):
            for j in range(min(self.tgt_ctx_size, i) + 1):
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

    def collater(self, samples):
        collate_fn = Seq2SeqCollater(pad_index=self.src_dict.pad(), eos_index=self.src_dict.eos(), ctx_method=self.ctx_method)
        samples = collate_fn.collate(samples)
        return samples

    def __getitem__(self, index):
        bos_id = torch.Tensor([self.src_dict.bos()]).long()
        eos_id = torch.Tensor([self.src_dict.eos()]).long()
        src_item = torch.cat([self.src[index], eos_id])
        tgt_item = torch.cat([self.tgt[index], eos_id])
        if self.ctx_method == "flag":
            embed = torch.zeros(src_item.size()[0])
        src_ctx_item = torch.Tensor().long()
        brk_id = torch.Tensor([self.src_dict.index("<brk>")]).long()
        for i in range(min(self.src_ctx_size, index) + 1):
            src_ctx_item = torch.cat([self.c_srcs[index - i], src_ctx_item])
        if self.ctx_method == "flag":
            embed = torch.cat((torch.ones(src_ctx_item.size()[0] + 1), embed)).long()
            src_item = torch.cat([src_ctx_item, brk_id, src_item])
        elif self.ctx_method == "encode":
            src_ctx_item = torch.cat([bos_id, src_ctx_item, eos_id])
        else:
            src_item = torch.cat([src_ctx_item, brk_id, src_item])
        tgt_ctx_item = torch.Tensor().long()
        brk_id = torch.Tensor([self.src_dict.index("<brk>")]).long()
        for i in range(min(self.tgt_ctx_size, index) + 1):
            tgt_ctx_item = torch.cat([self.c_tgts[index - i], tgt_ctx_item])
        tgt_item = torch.cat([tgt_ctx_item, brk_id, tgt_item])
          
        if self.ctx_method == "encode":
            return {"id": index, "source": src_item, "context": src_ctx_item, "target": tgt_item}
        elif self.ctx_method == "flag":
            return {"id": index, "source": src_item, "embed": embed, "target": tgt_item}
        else:
            return {"id": index, "source": src_item, "target": tgt_item}
