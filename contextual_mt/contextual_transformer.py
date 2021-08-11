from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.modules import LayerDropModuleList, TransformerEncoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from fairseq.models import (
    register_model,
    register_model_architecture,
)
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


@register_model("contextual_transformer")
class ContextualTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--context-loss",
            default=False,
            action="store_true",
            help="if set, trains to predict target context tokens",
        )
        parser.add_argument(
            "--coword-dropout",
            default=0.0,
            type=float,
            help="if set to value>0, randomly drops source tokens",
        )
        parser.add_argument(
            "--coword-dropout-type",
            choices=("sample", "predefined_sample", "whole", "suffix"),
            default="sample",
            help="type of coword dropout to use. NOTE: only sample is used"
            "used in the paper",
        )
        parser.add_argument(
            "--multi-encoder",
            default=False,
            action="store_true",
            help="wether to use multi-encoder in the source side",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ContextualTransformerEncoder(
            args,
            src_dict,
            embed_tokens,
            multi_encoder=getattr(args, "multi_encoder", False),
            coword_dropout_prob=getattr(args, "coword_dropout", 0.0),
            coword_dropout_type=getattr(args, "coword_dropout_type", "sample"),
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ContextualTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            multi_encoder=getattr(args, "multi_encoder", False),
            no_encoder_attn=getattr(args, "no_cross_attention", False),
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


class ContextualTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        multi_encoder=False,
        coword_dropout_type="sample",
        coword_dropout_prob=0.0,
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.coword_dropout_type = coword_dropout_type
        self.coword_dropout_prob = coword_dropout_prob
        # TODO: add this a variable token
        self.mask_id = dictionary.index("<mask>")
        self.multi_encoder = multi_encoder
        if self.multi_encoder:
            if self.encoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
            )

        self.num_layers = len(self.layers)

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
        if self.training and self.coword_dropout_type is not None:
            if self.coword_dropout_type == "sample":
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                probs = torch.ones_like(src_tokens) * self.coword_dropout_prob
                mask = torch.logical_and(
                    torch.bernoulli(probs), torch.logical_not(padding_mask)
                )
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.coword_dropout_type == "predefined_sample":
                # This is used for sampling with token specific probabilies
                # NOTE: this was not used in the paper
                assert (
                    src_sample_probs is not None
                ), "need sample probabilities as a given"
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                mask = torch.logical_and(
                    torch.bernoulli(src_sample_probs), torch.logical_not(padding_mask)
                )
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.coword_dropout_type == "whole":
                # make tensor with a single token (mask token)
                # NOTE: not used in the paper
                mask_samples = torch.zeros_like(src_tokens).to(src_tokens)
                mask_samples[mask_samples == 0] = self.padding_idx
                mask_samples[:, 0] = self.mask_id
                # replace samples by this tensor based on bernoulli
                probs = torch.ones((src_tokens.size(0),)) * self.coword_dropout_prob
                mask = torch.bernoulli(probs).to(src_tokens)
                mask = torch.unsqueeze(mask, -1).repeat(1, src_tokens.size(1))
                src_tokens = torch.where(mask == 0, src_tokens, mask_samples)
            else:
                raise ValueError(
                    f"unknown type of source dropout {self.coword_dropout_type}"
                )

        # Encode source tokens
        # as simple context encoding, we just concatenate context to input
        # TODO: add option for separate encoder
        # how to do it so that input can still attend to context
        def encode(tokens, layers):
            padding_mask = tokens.eq(self.padding_idx)
            x, encoder_embedding = self.forward_embedding(tokens)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            x_encoder_states = []
            for layer in layers:
                x = layer(x, padding_mask)
                if return_all_hiddens:
                    assert x_encoder_states is not None
                    x_encoder_states.append(x)

            if self.layer_norm is not None:
                x = self.layer_norm(x)

            return x, padding_mask, encoder_embedding, x_encoder_states

        if self.multi_encoder:
            ctx_x, ctx_padding_mask, ctx_enc_embeddings, ctx_x_enc_states = encode(
                src_ctx_tokens, self.context_layers
            )
            x, padding_mask, encoder_embedding, x_encoder_states = encode(
                src_tokens, self.layers
            )

            x = torch.cat([ctx_x, x], axis=0)
            padding_mask = torch.cat([ctx_padding_mask, padding_mask], axis=1)
            encoder_embedding = torch.cat(
                [ctx_enc_embeddings, encoder_embedding], axis=1
            )
            x_encoder_states = [
                torch.cat([ctx_states, states], axis=0)
                for ctx_states, states in zip(ctx_x_enc_states, x_encoder_states)
            ]

        else:
            x, padding_mask, encoder_embedding, x_encoder_states = encode(
                torch.cat([src_ctx_tokens, src_tokens], axis=1), self.layers
            )

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": x_encoder_states,  # List[T x B x C]
            "src_tokens": torch.empty(0),
            "src_lengths": torch.empty(0),
        }


class ContextualTransformerDecoder(TransformerDecoder):
    def __init__(
        self, args, dictionary, embed_tokens, multi_encoder=False, no_encoder_attn=False
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.multi_encoder = multi_encoder
        if self.multi_encoder:
            if self.decoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [self.build_encoder_layer(args) for i in range(args.decoder_layers)]
            )

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward_embedding(self, tokens, token_embedding: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
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
            context_tokens (LongTensor): context tokens (ie a prefix
                to prev_output_tokens), shape `(batch, tgt_ctx_len)`
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
        encoder_out: Optional[Dict[str, List[Tensor]]],
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
        encoder_out: Optional[Dict[str, List[Tensor]]],
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
            alignment_layer = 0  # self.num_layers - 1

        if self.multi_encoder:
            ctx_padding_mask = context_tokens.eq(self.padding_idx)
            ctx_x, _ = self.forward_embedding(context_tokens)
            # B x T x C -> T x B x C
            ctx_x = ctx_x.transpose(0, 1)
            for layer in self.context_layers:
                ctx_x = layer(ctx_x, ctx_padding_mask)

            if self.layer_norm is not None:
                ctx_x = self.layer_norm(ctx_x)

            input_tokens = prev_output_tokens
        else:
            input_tokens = torch.cat([context_tokens, prev_output_tokens], axis=1)
            context_end_id = context_tokens.size(1)

        # embed positions
        if self.embed_positions is not None:
            # concat context_tokens to input
            # FIXME: this is really simple
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

        if self.multi_encoder:
            cross_attn = (
                torch.cat([encoder_out["encoder_out"][0], ctx_x], axis=0)
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else ctx_x
            )
            cross_attn_mask = (
                torch.cat(
                    [encoder_out["encoder_padding_mask"][0], ctx_padding_mask], axis=1
                )
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else ctx_padding_mask
            )
        else:
            cross_attn = (
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None
            )
            cross_attn_mask = (
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None
            )

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
                cross_attn,
                cross_attn_mask,
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
        if not self.multi_encoder:
            x = x[context_end_id:]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("contextual_transformer", "contextual_transformer")
def contextual_transformer_base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("contextual_transformer", "contextual_transformer_iwslt")
def contextual_transformer_iwslt_architecture(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("contextual_transformer", "contextual_transformer_big")
def contextual_transformer_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)
