# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
    This module contains collection of classes which implement
    collate functionalities for various tasks.

    Collaters should know what data to expect for each sample
    and they should pack / collate them into batches
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

import torch
from fairseq.data import data_utils


class Seq2SeqCollater(object):
    """
        Implements collate function mainly for seq2seq tasks
        This expects each sample to contain feature (src_tokens) and
        targets.
        This collator is also used for aligned training task.
    """

    def __init__(
        self,
        feature_index=0,
        label_index=1,
        pad_index=1,
        eos_index=2,
        move_eos_to_beginning=False,
        ctx_method="tag",
    ):
        self.feature_index = feature_index
        self.label_index = label_index
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.move_eos_to_beginning = move_eos_to_beginning
        self.ctx_method = ctx_method

    def collate(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s["id"] for s in samples])
        if self.ctx_method == "flag":
            source = data_utils.collate_tokens([s["source"]for s in samples], self.pad_index, eos_idx=self.eos_index)
            embed = data_utils.collate_tokens([s["embed"] for s in samples], self.pad_index)
        elif self.ctx_method == "encode":
            size = max(max(s["source"].size(0) for s in samples), max(s["context"].size(0) for s in samples))
            source = data_utils.collate_tokens([s["source"] for s in samples], self.pad_index, eos_idx=self.eos_index, pad_to_length=size)
            context = data_utils.collate_tokens([s["context"] for s in samples], self.pad_index, pad_to_length=size)
        else:
            source = data_utils.collate_tokens([s["source"] for s in samples], self.pad_index, eos_idx=self.eos_index)

        target = data_utils.collate_tokens([s["target"] for s in samples], self.pad_index, eos_idx=self.eos_index)
        
        prev_output_tokens = data_utils.collate_tokens(
                [s["target"] for s in samples],
                self.pad_index,
                self.eos_index,
                left_pad=False,
                move_eos_to_beginning=True,
            )

        batch = {
            "id": id,
            "ntokens": sum(len(s["target"]) for s in samples),
            "net_input": {"src_tokens": source, "src_lengths": torch.LongTensor([s.size(0) for s in source]), "prev_output_tokens":prev_output_tokens},
            "target": target,
            "nsentences": len(samples),
        }
        if self.ctx_method == "flag":
            batch["net_input"]["brk_embed"] = embed
        elif self.ctx_method == "encode":
            batch["net_input"]["cxt_tokens"] = context
        return batch
