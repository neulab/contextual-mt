from typing import Optional

import torch
import torch.nn as nn

from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import FairseqEncoder, EncoderOut
from fairseq.models.transformer import TransformerModel, TransformerEncoder
from fairseq.models.transformer import (
    base_architecture as transformer_base_architecture,
)
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

import math


@register_model("contextual_transformer")
class ContextualTransformerModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--share-context-layers",
            default=False,
            action="store_true",
            help="if set, context is encoded using same transformer layers as the input",
        )
        parser.add_argument(
            "--context-encoder-layers",
            type=int,
            metavar="N",
            help="num encoder layers for context",
        )

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
    def __init__(self, args, dictionary, embed_tokens):
        FairseqEncoder.__init__(self, dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if not args.share_context_layers:
            if self.encoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [
                    self.build_encoder_layer(args)
                    for i in range(args.context_encoder_layers)
                ]
            )
        else:
            self.context_layers = self.layers[:-1]

        self.num_context_layers = len(self.context_layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

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
        for layer in self.layers[:-1]:
            x = layer(x, x_padding_mask)
            if return_all_hiddens:
                assert x_encoder_states is not None
                x_encoder_states.append(x)

        # Encode context tokens
        c, _ = self.forward_embedding(src_context)
        # B x T x C -> T x B x C
        c = c.transpose(0, 1)
        c_padding_mask = src_context.eq(self.padding_idx)
        for layer in self.context_layers:
            c = layer(c, c_padding_mask)

        # collapse context to a single embedding (via mean)
        unsq_c_mask = torch.logical_not(torch.unsqueeze(c_padding_mask, -1)).transpose(
            0, 1
        )
        num_c = torch.torch.sum(unsq_c_mask, dim=0)
        c_collapsed = torch.sum(c * unsq_c_mask, dim=0) / num_c
        # and add that embedding to all x
        x = x + torch.unsqueeze(c_collapsed, dim=0)

        x = self.layers[-1](x, x_padding_mask)
        if return_all_hiddens:
            x_encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=x_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=x_encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        state_dict = super().upgrade_state_dict_named(state_dict, name)
        for i in range(self.num_context_layers):
            # update layer norms
            self.context_layers[i].upgrade_state_dict_named(
                state_dict, "{}.context_layers.{}".format(name, i)
            )
        return state_dict


@register_model_architecture("contextual_transformer", "contextual_transformer")
def base_architecture(args):
    transformer_base_architecture(args)
    args.context_encoder_layers = getattr(args, "context_encoder_layers", 2)
    args.share_context_layers = getattr(args, "share_context_layers", False)
