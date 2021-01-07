from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
    ARCH_MODEL_REGISTRY,
)
from fairseq.models.fairseq_encoder import FairseqEncoder, EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models.transformer import (
    base_architecture as transformer_base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

import math


@register_model("contextual_transformer")
class ContextualTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--context-loss', default=False, action='store_true',
                            help='if set, trains to predict target context tokens')
        parser.add_argument('--source-dropout', default=0.0, type=float,
                            help='if set to value>0, randomly drops source tokens')
        parser.add_argument('--source-dropout-type', 
                            choices=("sample", "predefined_sample", "whole", "suffix"), 
                            default='sample',
                            help='')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ContextualTransformerEncoder(
            args, src_dict, embed_tokens,
            source_dropout_prob=getattr(args,"source_dropout", 0.0),
            source_dropout_type=getattr(args, "source_dropout_type", "sample"))

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ContextualTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            context_loss=getattr(args, "context_loss", False)
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        src_ctx_tokens=None,
        src_ctx_lengths=None,
        tgt_ctx_tokens=None,
        tgt_ctx_lengths=None,
        src_sample_probs=None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_ctx_tokens=src_ctx_tokens,
            src_ctx_lengths=src_ctx_lengths,
            src_sample_probs=src_sample_probs,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            context_tokens=tgt_ctx_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        if self.training and self.decoder.context_loss:
            target = torch.cat([sample["context_target"], sample["target"]], axis=1)
        else:
            target = sample["target"]
        return target


class ContextualTransformerEncoder(TransformerEncoder):
    def __init__(
        self, 
        args, dictionary, embed_tokens, 
        source_dropout_type="sample",
        source_dropout_prob=0.
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.source_dropout_type = source_dropout_type
        self.source_dropout_prob = source_dropout_prob
        # TODO: add this a variable token
        self.mask_id = dictionary.index("<mask>")

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_ctx_tokens,
        src_ctx_lengths,
        src_sample_probs=None,
        return_all_hiddens: bool = False,
    ):

        # if source dropout enabled, randomly drop tokens from input
        if self.training and self.source_dropout_type != None:
            if self.source_dropout_type == "sample":
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                probs = torch.ones_like(src_tokens) * self.source_dropout_prob
                mask = torch.logical_and(torch.bernoulli(probs), torch.logical_not(padding_mask))
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.source_dropout_type == "predefined_sample":
                assert src_sample_probs is not None, "need sample probabilities as a given"
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                mask = torch.logical_and(torch.bernoulli(src_sample_probs), torch.logical_not(padding_mask))
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.source_dropout_type == "whole":
                # make tensor with a single token (mask token)
                mask_samples = torch.zeros_like(src_tokens).to(src_tokens)
                mask_samples[mask_samples==0] = self.padding_idx
                mask_samples[:, 0] = self.mask_id
                # replace samples by this tensor based on bernoulli
                probs = torch.ones((src_tokens.size(0),)) * self.source_dropout_prob
                mask = torch.bernoulli(probs).to(src_tokens)
                mask = torch.unsqueeze(mask, -1).repeat(1, src_tokens.size(1))
                src_tokens = torch.where(mask==0, src_tokens, mask_samples)
            elif self.source_dropout_type == "suffix":
                # mask random sample
                pass
            else:
                raise ValueError(f"unknown type of source dropout {self.source_dropout_type}")

        # Encode source tokens
        # as simple context encoding, we just concatenate context to input
        # TODO: add option for separate encoder
        # how to do it so that input can still attend to context 
        input_tokens = torch.cat([src_ctx_tokens, src_tokens], axis=1)
        padding_mask = input_tokens.eq(self.padding_idx)

        x, encoder_embedding = self.forward_embedding(input_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x_encoder_states = [] if return_all_hiddens else None
        for layer in self.layers:
            x = layer(x, padding_mask)
            if return_all_hiddens:
                assert x_encoder_states is not None
                x_encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=x_encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


class ContextualTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, context_loss=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.context_loss = context_loss

        last_layer = TransformerDecoderLayerReturnSelfAttention(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            last_layer = checkpoint_wrapper(last_layer)
        self.layers[-1] = last_layer

    def forward(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            context_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            context_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # concat context_tokens to input
        # FIXME: this is really simple
        input_tokens = torch.cat([context_tokens, prev_output_tokens], axis=1)
        context_end_id = context_tokens.size(1)

        # embed positions
        if self.embed_positions is not None:
            positions = self.embed_positions(
                input_tokens, incremental_state=incremental_state
            )
        else:
            positions = None

        if incremental_state is not None and len(incremental_state) > 0:
            input_tokens = input_tokens[:, -1:]
            context_end_id = 0
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(input_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or input_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = input_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if (
                incremental_state is None or len(incremental_state) == 0
            ) and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            if attn.dim() == 4:
                attn = attn.mean(dim=0)

        # remove context
        if (not self.training) or not self.context_loss:
            x = x[context_end_id:]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


class TransformerDecoderLayerReturnSelfAttention(TransformerDecoderLayerReturnSelfAttention):

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, _ = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None



@register_model_architecture("contextual_transformer", "contextual_transformer")
def contextual_transformer_base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("contextual_transformer", "contextual_transformer_iwslt")
def contextual_transformer_iwslt_architecture(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("contextual_transformer", "contextual_transformer_big")
def contextual_transformer_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)
