from typing import Optional, List, Dict, Any

import torch
from torch import Tensor

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
from fairseq.modules import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)


@register_model("attn_reg_transformer")
class AttnRegTransformerModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return AttnRegTransformerEncoder(
            args,
            src_dict,
            embed_tokens,
            coword_dropout_prob=getattr(args, "coword_dropout", 0.0),
            coword_dropout_type=getattr(args, "coword_dropout_type", "sample"),
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AttnRegTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            context_loss=getattr(args, "context_loss", False),
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
        enc_alignment_layer: Optional[int] = None,
        dec_alignment_layer: Optional[int] = None,
        cross_alignment_layer: Optional[int] = None,
        self_alignment_layer: Optional[int] = None,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        if enc_alignment_layer is None:
            enc_alignment_layer = alignment_layer
        if dec_alignment_layer is None:
            dec_alignment_layer = alignment_layer
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_ctx_tokens=src_ctx_tokens,
            src_ctx_lengths=src_ctx_lengths,
            src_sample_probs=src_sample_probs,
            alignment_layer=enc_alignment_layer,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            context_tokens=tgt_ctx_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=dec_alignment_layer,
            cross_alignment_layer=cross_alignment_layer,
            self_alignment_layer=self_alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out[1]["encoder_out"] = encoder_out
        return decoder_out

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        if self.training and self.decoder.context_loss:
            target = torch.cat([sample["context_target"], sample["target"]], axis=1)
        else:
            target = sample["target"]
        return target


class AttnRegTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        coword_dropout_type="sample",
        coword_dropout_prob=0.0,
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.coword_dropout_type = coword_dropout_type
        self.coword_dropout_prob = coword_dropout_prob
        # TODO: add this a variable token
        self.mask_id = dictionary.index("<mask>")

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayerReturnSelfAttention(args)
        return layer

    def forward(
        self,
        src_tokens,
        src_lengths,
        src_ctx_tokens,
        src_ctx_lengths,
        src_sample_probs=None,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_all_hiddens: bool = False,
    ):

        # Encode source tokens
        # as simple context encoding, we just concatenate context to input
        # TODO: add option for separate encoder
        # how to do it so that input can still attend to context
        input_tokens = torch.cat([src_ctx_tokens, src_tokens], axis=1)
        padding_mask = input_tokens.eq(self.padding_idx)

        x, encoder_embedding = self.forward_embedding(input_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attention = []

        x_encoder_states = []
        for idx, layer in enumerate(self.layers):
            x, self_attn = layer(x, padding_mask)
            if self_attn is not None:
                if idx == alignment_layer:
                    self_attention = self_attn.float().to(x)
                elif isinstance(alignment_layer, list) and idx in alignment_layer:
                    self_attention.append(self_attn.float().to(x))
            if return_all_hiddens:
                assert x_encoder_states is not None
                x_encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": x_encoder_states,  # List[T x B x C]
            "enc_self_attn": torch.stack(self_attention)
            if len(self_attention) > 0
            else torch.empty(0),
            "src_tokens": torch.empty(0),  # B x T
            "src_lengths": torch.empty(0),  # B x 1
        }


class TransformerEncoderLayerReturnSelfAttention(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, self_attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            need_head_weights=True,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

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
        return x, self_attn


class AttnRegTransformerDecoder(TransformerDecoder):
    def __init__(
        self, args, dictionary, embed_tokens, no_encoder_attn=False, context_loss=False
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.context_loss = context_loss

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayerReturnSelfAttention(args, no_encoder_attn)
        return layer

    def forward(
        self,
        prev_output_tokens,
        context_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        cross_alignment_layer: Optional[int] = None,
        self_alignment_layer: Optional[int] = None,
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
            cross_alignment_layer=cross_alignment_layer,
            self_alignment_layer=self_alignment_layer,
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
        cross_alignment_layer: Optional[int] = None,
        self_alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            context_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            cross_alignment_layer,
            self_alignment_layer,
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
        cross_alignment_layer: Optional[int] = None,
        self_alignment_layer: Optional[int] = None,
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
        if cross_alignment_layer is None:
            cross_alignment_layer = alignment_layer
        if self_alignment_layer is None:
            self_alignment_layer = alignment_layer
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

        if isinstance(cross_alignment_layer, list):
            cross_attn = []
        if isinstance(self_alignment_layer, list):
            self_attn = []

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or input_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = input_tokens.eq(self.padding_idx)

        # decoder layers
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
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=True,
                need_head_weights=True,
            )
            inner_states.append(x)

            if layer_attn is not None:
                if idx == self_alignment_layer:
                    self_attn = layer_attn[0].float().to(x)
                if idx == cross_alignment_layer:
                    cross_attn = layer_attn[1].float().to(x)
                if isinstance(self_alignment_layer, list):
                    if idx in self_alignment_layer:
                        self_attn.append(layer_attn[0].float().to(x))
                if isinstance(cross_alignment_layer, list):
                    if idx in cross_alignment_layer:
                        cross_attn.append(layer_attn[1].float().to(x))

        if self_attn is not None:
            if isinstance(self_alignment_layer, list):
                self_attn = torch.stack(self_attn)
            elif alignment_heads is not None:
                self_attn = self_attn[:alignment_heads]

        if cross_attn is not None:
            if isinstance(cross_alignment_layer, list):
                cross_attn = torch.stack(cross_attn)
            elif alignment_heads is not None:
                cross_attn = cross_attn[:alignment_heads]

        # remove context
        if (not self.training) or not self.context_loss:
            x = x[context_end_id:]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [self_attn, cross_attn], "inner_states": inner_states}


class TransformerDecoderLayerReturnSelfAttention(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def residual_connection(self, x, residual):
        return residual + x

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

        x, self_attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=True,
            need_head_weights=True,
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

            x, cross_attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=True,
                need_head_weights=True,
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
        return x, [cross_attn, self_attn], None


@register_model_architecture("attn_reg_transformer", "attn_reg_transformer")
def attn_reg_transformer_base_architecture(args):
    transformer_base_architecture(args)


@register_model_architecture("attn_reg_transformer", "attn_reg_transformer_iwslt")
def attn_reg_transformer_iwslt_architecture(args):
    transformer_iwslt_de_en(args)


@register_model_architecture("attn_reg_transformer", "attn_reg_transformer_big")
def attn_reg_transformer_big_architecture(args):
    transformer_vaswani_wmt_en_de_big(args)
