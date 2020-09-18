import argparse
import os
from typing import List
from fire import Fire

import torch

from fairseq.data import Dictionary


def load_dict(langs: List[str], path: str) -> Dictionary:
    d = Dictionary.load(path)
    for l in langs:
        d.add_symbol(f"[{l}]")
    d.add_symbol("<mask>")
    return d


def main(pretrain_dict_file, pretrain_model_file, ft_dict, langs, output) -> None:
    #langs = langs.split(",")
    pre_dict = load_dict(langs, pretrain_dict_file)
    ft_dict = load_dict(langs, ft_dict)
    data = torch.load(pretrain_model_file)
    model = data["model"]

    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))

    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        model[name] = ft_tensor

    torch.save(data, os.path.join(output, "model.pt"))

if __name__ == "__main__":
    Fire(main)