from typing import Optional

import torch

from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, TransformerEncoder
from fairseq.models.transformer import (
    base_architecture as transformer_base_architecture,
)


@register_model("contextual_transformer")
class ContextualTransformerModel(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ContextualTransformerEncoder(args, src_dict, embed_tokens)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        src_context,
        src_ctx_lengths,
        prev_output_tokens,
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
            src_context=src_context,
            src_ctx_lengths=src_ctx_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out


class ContextualTransformerEncoder(TransformerEncoder):
    def forward(
        self,
        src_tokens,
        src_lengths,
        src_context,
        src_ctx_lengths,
        return_all_hiddens: bool = False,
    ):
        # Encode source tokens
        x, encoder_embedding = self.forward_embedding(src_tokens)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x_padding_mask = src_tokens.eq(self.padding_idx)
        x_encoder_states = [] if return_all_hiddens else None
        for layer in self.layers[:-2]:
            x = layer(x, x_padding_mask)
            if return_all_hiddens:
                assert x_encoder_states is not None
                x_encoder_states.append(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Encode context tokens
        c, _ = self.forward_embedding(src_context)
        # B x T x C -> T x B x C
        c = c.transpose(0, 1)
        c_padding_mask = src_context.eq(self.padding_idx)
        for layer in self.layers[:-2]:
            c = layer(c, c_padding_mask)
        if self.layer_norm is not None:
            c = self.layer_norm(c)

        # collapse context to a single embedding (via mean)
        unsq_c_padding = torch.unsqueeze(c_padding_mask, -1).transpose(0, 1)
        c_collapsed = torch.sum(c * unsq_c_padding, dim=0)
        c_collapsed = c_collapsed / torch.sum(unsq_c_padding, dim=0)
        # and add that embedding to all x
        x = x + torch.unsqueeze(c_collapsed, dim=0)

        x = self.layers[-1](x, x_padding_mask)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=x_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=x_encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@register_model_architecture("contextual_transformer", "contextual_transformer")
def base_architecture(args):
    transformer_base_architecture(args)
