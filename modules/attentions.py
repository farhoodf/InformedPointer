"""
These codes mainly borrowed from distilbert implementation for transformer of huggingface:
https://github.com/huggingface/transformers
"""

import math

import torch
import torch.nn.functional as F

import copy

import numpy as np
import torch
import torch.nn as nn


# from transformers.activations import gelu, gelu_new, swish

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
def _gelu_python(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        This is now written in C in torch.nn.functional
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu

gelu = gelu_new

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)
        self.output_attentions = config.output_attentions

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.qdim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()


    def forward(self, query, key, value, mask, head_mask=None):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        # q = shape(gelu(self.q_lin(query)))  # (bs, n_heads, q_length, dim_per_head)
        # k = shape(gelu(self.k_lin(key)))  # (bs, n_heads, k_length, dim_per_head)
        # v = shape(gelu(self.v_lin(value)))  # (bs, n_heads, k_length, dim_per_head)
        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if self.output_attentions:
            return (context, weights)
        else:
            return (context,)

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.ffn_dropout)
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in ["relu", "gelu"], "activation ({}) must be in ['relu', 'gelu']".format(
            config.activation
        )
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.output_attentions = config.output_attentions

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(query=x, key=x, value=x, mask=attn_mask, head_mask=head_mask)
        if self.output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output

class TransformerDecoderBlock(nn.Module):
	"""docstring for TransformerDecoderBlock"""
	def __init__(self, config):
		super(TransformerDecoderBlock, self).__init__()
		self.output_attentions = config.output_attentions

		assert config.dim % config.n_heads == 0

		self.attention = MultiHeadAttention(config)
		self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

		self.ffn = FFN(config)
		self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)


	def forward(self, query, states, attn_mask=None, head_mask=None):
		"""
		Parameters
		----------
		query: torch.tensor(bs, seq_length, dim)
		states: torch.tensor(bs, seq_length, dim)
		attn_mask: torch.tensor(bs, seq_length)

		Outputs
		-------
		sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
			The attention weights
		ffn_output: torch.tensor(bs, seq_length, dim)
			The output of the transformer block contextualization.
		"""

		sa_output = self.attention(query=query, key=states, value=states, mask=attn_mask, head_mask=head_mask)
		if self.output_attentions:
			sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
		else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
			assert type(sa_output) == tuple
			sa_output = sa_output[0]

		# sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

		# Feed Forward Network
		ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
		ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

		output = (ffn_output,)
		if self.output_attentions:
			output = (sa_weights,) + output
		return output

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        layer = TransformerEncoderBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, attn_mask=None, head_mask=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        device = x.device
        if attn_mask is None:
            attn_mask = torch.ones(x.shape[:-1], device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layers


        all_hidden_states = ()
        all_attentions = ()

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i])
            hidden_state = layer_outputs[-1]

            if self.output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class Attention(nn.Module):
	"""docstring for Attention
	Inputs:
	query:		(bs,emb)
	states:		(bs, slen, dim)
	Output:
	weights:	(bs, slen)
	attn:		(bs, slen)
	"""
	def __init__(self, q_dim, v_dim, internal_dim=None, dropout_value = 0.2,need_attn=True):
		super(Attention, self).__init__()
		
		if internal_dim == None:
			internal_dim = v_dim
		self.dropout_value = dropout_value

		self.need_attn = need_attn

		self.proj_q = nn.Linear(q_dim, internal_dim)
		self.proj_k = nn.Linear(v_dim, internal_dim)
		self.proj_e = nn.Linear(2*internal_dim,1)

		self.drop_q = nn.Dropout(self.dropout_value)
		self.drop_k = nn.Dropout(self.dropout_value)

	def forward(self, query, states, mask=None):
		bs,slen,em = states.shape
		q = F.relu(self.proj_q(query)).unsqueeze(1).repeat(1,slen,1)
		q = self.drop_q(q)
		k = F.relu(self.proj_k(states))
		k = self.drop_k(k)

		energy = self.proj_e(torch.tanh(torch.cat((q,k),-1))).squeeze(2)
		# if not self.need_attn:
		if mask is not None:
			energy.masked_fill_(mask,1e-45)
		# mask can be applied here to put very low value for energy

		attn_weights = F.softmax(energy,dim=1)
		# if mask is not None:
		# 	attn_weights.masked_fill_(mask, 0)
		out = (attn_weights,)

		if self.need_attn:
			# v = F.relu(self.proj_v(states))
			v = states
			attn = torch.mul(attn_weights.unsqueeze(2).repeat(1,1,em),v).sum(1)
			out += (attn,)

		return out

