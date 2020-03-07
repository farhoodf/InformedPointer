import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel

from .attentions import *
from .pointers import *

def getmask_hugging(s_lengths, batch_size, s_len):
	"""attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
		Mask to avoid performing attention on padding token indices.
		Mask values selected in ``[0, 1]``:
		``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens."""
	mask=torch.zeros((batch_size,s_len+1),dtype=torch.bool,device=s_lengths.device,requires_grad=False)

	mask[torch.arange(batch_size),s_lengths] = 1
	mask = ~mask.cumsum(dim=1).bool().to(device=s_lengths.device)
	return mask[:,:-1]


class WordEncoder(nn.Module):
	"""docstring for WordEncoder"""
	def __init__(self, config):
		super(WordEncoder, self).__init__()
		self.encoder_type = config.encoder_type
		if self.encoder_type == 'bert':
			self.encoder = BertModel.from_pretrained('bert-base-uncased')
			self.forward = self.forward_berties
		elif self.encoder_type == 'distil':
			self.encoder = DistilBertModel.from_pretrained('distilbert-base-cased')
			self.forward = self.forward_berties
		elif self.encoder_type == 'rnn':
			self.encoder = RNNWordEncoder(config.rnn_config)
			self.embedding = nn.Embedding(config.n_tokens, config.word_embedding_size)
			self.drop = nn.Dropout(config.dropout)
			self.forward = self.forward_rnn

	def forward_berties(self, x, s_lengths):
		batch_size, s_len = x.shape
		mask = getmask_hugging(s_lengths, batch_size, s_len)
		word_embedded = self.encoder(x, attention_mask = mask)
		return word_embedded[0], mask

	def forward_rnn(self, x, s_lengths):
		batch_size, s_len = x.shape
		mask = getmask_hugging(s_lengths, batch_size, s_len)
		word_embedded = self.embedding(x)
		word_embedded = self.drop(word_embedded)
		word_embedded = self.encoder(word_embedded, lengths=s_lengths)
		return word_embedded, mask
	

class SentenceEncoder(nn.Module):
	"""docstring for SentenceEncoder"""
	def __init__(self, config):
		super(SentenceEncoder, self).__init__()
		self.encoder_type = config.encoder_type
		if self.encoder_type == 'rnn':
			self.encoder = RNNEncoder(config.rnn_config)
			self.forward = self.forward_birnn
		elif self.encoder_type == 'attention':
			self.encoder = MultiHeadAttention(config.multihead_conf)
			self.query = nn.Linear(1, config.multihead_conf.dim, bias=False)
			self.forward = self.forward_attention

	def forward_birnn(self,inputs, lengths=None):
		outputs_ = self.encoder(inputs, lengths)
		outputs = torch.zeros_like(outputs_[:,0,:])
		outputs[:,:self.hidden_size] = outputs_[:,0,self.hidden_size:]
		outputs[:,self.hidden_size:] = outputs_[:,-1,:self.hidden_size]
		outputs = self.dense(outputs)
		return outputs

	def forward_attention(self, word_embedded, mask):
		ones = torch.ones((word_embedded.shape[0],1,1),device=word_embedded.device)
		q = self.query(ones)
		sent_embedded = self.encoder(q, word_embedded, word_embedded, mask)[0]
		sent_embedded = sent_embedded.squeeze(1)
		return sent_embedded


class ParagEncoder(nn.Module):
	"""docstring for ParagEncoder"""
	def __init__(self, config):
		super(ParagEncoder, self).__init__()
		self.encoder_type = config.encoder_type
		self.query = nn.Linear(1, config.multihead_conf.dim, bias=False)
		self.encoder = MultiHeadAttention(config.multihead_conf)
		# if self.encoder_type == 'transformer':
		self.transformer = TransformerEncoder(config.transformer_conf)

	def forward(self, sent_embedded, mask=None):
		if mask is None:
			bs, p_len, _ = sent_embedded.shape
			mask = torch.ones(bs, p_len).to(device=sent_embedded.device)

		# if self.encoder_type == 'transformer':
		sent_embedded = self.transformer(sent_embedded, mask)[0]

		ones = torch.ones((sent_embedded.shape[0],1,1),device=sent_embedded.device)
		q = self.query(ones)
		parag_embedded = self.encoder(q, sent_embedded, sent_embedded, mask)[0]
		parag_embedded = parag_embedded.squeeze(1)
		return parag_embedded, sent_embedded


class RNNEncoder(nn.Module):
	"""docstring for RNNWordEncoder"""
	def __init__(self, config):
		super(RNNWordEncoder, self).__init__()
		

		self.rnn_type = config.rnn_type
		self.input_size = config.input_size
		self.hidden_size = config.hidden_size
		self.n_layers = config.n_layers
		self.dropout = config.dropout
		self.bidirectional = config.bidirectional
		self.batch_first = config.batch_first
		
		self.rnn = getattr(nn, self.rnn_type)(self.input_size, self.hidden_size, self.n_layers, 
			dropout=self.dropout, bidirectional=self.bidirectional, batch_first=self.batch_first)
		# self.rnn.apply(init_weights)
		

	def forward(self, inputs, hiddens=None, lengths=None):#, enforce_sorted=True):
		# lengths = None
		if lengths is not None:
			inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, 
						batch_first=self.batch_first, enforce_sorted=False)
		
		outputs, _ = self.rnn(inputs, hiddens)

		if lengths is not None:
			outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)

		return outputs

