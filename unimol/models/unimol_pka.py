# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .unimol import UniMolModel, ClassificationHead, NonLinearHead

logger = logging.getLogger(__name__)


@register_model("unimol_pka")
class UniMolPKAModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-charge-loss",
            type=float,
            metavar="D",
            help="mask charge loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )

    def __init__(self, args, dictionary, charge_dictionary):
        super().__init__()
        unimol_pka_architecture(args)
        self.args = args
        self.unimol = UniMolModel(self.args, dictionary, charge_dictionary)
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
        if args.masked_charge_loss > 0:
            self.charge_lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(charge_dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )
        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.charge_dictionary)

    def forward(
        self,
        input_metadata,
        classification_head_name=None,
        encoder_masked_tokens=None,
        features_only=False,
        **kwargs
    ):
        if not features_only:
            src_tokens, src_charges, src_coord, src_distance, src_edge_type, batch, charge_targets, coord_targets, dist_targets, token_targets = input_metadata
        else:
            src_tokens, src_charges, src_coord, src_distance, src_edge_type, batch = input_metadata
            charge_targets, coord_targets, dist_targets, token_targets = None, None, None, None
        padding_mask = src_tokens.eq(self.unimol.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.unimol.embed_tokens(src_tokens)

        charge_padding_mask = src_charges.eq(self.unimol.charge_padding_idx)
        if not charge_padding_mask.any():
            padding_mask = None
        charges_emb = self.unimol.embed_charges(src_charges)
        # involve charge info
        x += charges_emb

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.unimol.gbf(dist, et)
            gbf_result = self.unimol.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.unimol.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_charge_loss > 0:
                charge_logits = self.charge_lm_head(encoder_rep, encoder_masked_tokens)                
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)

        if classification_head_name is not None:
            cls_logits = self.unimol.classification_heads[classification_head_name](encoder_rep)
        if not features_only:
            return (
            cls_logits, batch,
            logits, charge_logits, encoder_distance, encoder_coord, x_norm, delta_encoder_pair_rep_norm,
            token_targets, charge_targets, coord_targets, dist_targets
        )
        else: 
            return cls_logits, batch

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.unimol.classification_heads:
            prev_num_classes = self.unimol.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.unimol.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.unimol.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@register_model_architecture("unimol_pka", "unimol_pka")
def unimol_pka_architecture(args):
    def base_architecture(args):
        args.encoder_layers = getattr(args, "encoder_layers", 15)
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
        args.dropout = getattr(args, "dropout", 0.1)
        args.emb_dropout = getattr(args, "emb_dropout", 0.1)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.activation_dropout = getattr(args, "activation_dropout", 0.0)
        args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
        args.max_seq_len = getattr(args, "max_seq_len", 512)
        args.activation_fn = getattr(args, "activation_fn", "gelu")
        args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
        args.post_ln = getattr(args, "post_ln", False)
        args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
        args.masked_charge_loss = getattr(args, "masked_charge_loss", -1.0)
        args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
        args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)    
        args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
        args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)

    base_architecture(args)
