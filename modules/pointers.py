import torch
import torch.nn as nn

from .attentions import *


class InformedPointer(nn.Module):
	"""docstring for InformedPointer

	Inputs:parag_embedded, sent_embedded, word_embedded
	parag_embedded	()
	sent_embedded	()
	word_embedded	(bs,slen,emsize)
	"""
	def __init__(self, config):
		super(InformedPointer, self).__init__()
		self.rnn_type = 'LSTM'
		self.embedding_dim = config.embedding_dim
		self.rnn_hidden_size = config.rnn_hidden_size
		self.rnn_n_layers = config.rnn_n_layers
		self.rnn_dropout_value = config.rnn_dropout_value
		self.dropout_value = config.rnnout_dropout_value
		self.informed_attention_type = config.informed_attention_type
		self.pointer_drop = config.pointer_drop



		self.rnn = getattr(nn, self.rnn_type)(self.embedding_dim, self.rnn_hidden_size, self.rnn_n_layers, 
			dropout=self.rnn_dropout_value, bidirectional=False)
		
		self.after_rnn_drop = nn.Dropout(self.dropout_value)
		self.pointer = Attention(self.rnn_hidden_size, self.embedding_dim, dropout_value=self.pointer_drop, need_attn=False)

		self.parag_to_hidden = nn.Linear(self.embedding_dim,self.rnn_hidden_size)

		if self.informed_attention_type == 'informed':
			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.just_informed
		elif self.informed_attention_type == 'addinformed':
			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.add_informed
			self.sent_score = nn.Linear(self.embedding_dim,1)
			self.info_score = nn.Linear(self.embedding_dim,1)
		elif self.informed_attention_type == 'notinformed':
			self.get_informed = self.pass_informed

		
		
	def forward(self, parag_embedded, sent_embedded, word_embedded, sent_mask, labels=None):
		batch_size, p_len, s_len, _ = word_embedded.shape
		if self.training:
			do_tf = True 
			# if torch.rand(1) < 0.3:
			# 	do_mask = False
			# else:
			# 	do_mask = True
		else:
			do_tf = False
		do_mask = True

		res = []
		hidden = self.init_hidden(batch_size,parag_embedded.device)
		hidden[0][-1] = self.parag_to_hidden(parag_embedded)
		selected = torch.zeros_like(parag_embedded).unsqueeze(0)
		par_mask = torch.zeros((batch_size, p_len), dtype=bool, device= word_embedded.device, requires_grad=False)
		for i in range(p_len):
			_, hidden = self.rnn(selected,hidden)
			hidden = (self.after_rnn_drop(hidden[0]), self.after_rnn_drop(hidden[1]))
			
			informed = self.get_informed(hidden[0][-1], word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len)
			attn_weights = self.pointer(hidden[0][-1], informed, mask=par_mask)[0]

			if do_tf:
				index = labels[:,i]
			else:
				index = attn_weights.argmax(1)

			selected = sent_embedded[torch.arange(batch_size),index] 
			selected = selected.unsqueeze(0)

			par_mask = par_mask.clone().detach()
			if do_mask:
				par_mask[torch.arange(batch_size),index] = 1


			res.append(attn_weights)

		return torch.stack(res).transpose(0,1)
	
	

	def just_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		# get informed keys
		q = query.unsqueeze(1).repeat(1,p_len,1).view(batch_size*p_len,1,self.rnn_hidden_size)
		s = word_embedded.view(batch_size*p_len,s_len,self.embedding_dim)
		# _, informed = self.informed_attn(q, s) #hidden or selected? # single head
		informed = self.informed_attn(q, s, sent_mask)[0] #hidden or selected?
		informed = informed.view(batch_size,p_len,self.embedding_dim)
		return informed

	def add_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		informed = self.just_informed(query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len)
		g = torch.sigmoid(self.info_score(informed)+self.sent_score(sent_embedded))
		informed = self.informed_layernorm(g*informed+(1-g)*sent_embedded)
		return informed
	
	def pass_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		return sent_embedded
	
	def init_hidden(self, batch_size,device):
		# num_layers * num_directions, batch, hidden_size
		if self.rnn_type =='LSTM':
			h_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			c_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			return (h_n, c_n)
