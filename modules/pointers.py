import torch
import torch.nn as nn

from .attentions import *

def get_parmask(p_lengths, batch_size, p_len):
	"""attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
		Mask to avoid performing attention on padding token indices.
		Mask values selected in ``[0, 1]``:
		``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens."""
	mask=torch.zeros((batch_size,p_len+1),dtype=torch.bool,device=p_lengths.device,requires_grad=False)

	mask[torch.arange(batch_size),p_lengths] = 1
	mask = mask.cumsum(dim=1).bool().to(device=p_lengths.device)
	return mask[:,:-1]


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
			self.info_score = nn.Linear(2*self.embedding_dim,1)
		elif self.informed_attention_type == 'catinformed':
			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.cat_informed
			self.info_score = nn.Linear(2*self.embedding_dim,self.embedding_dim)
		elif self.informed_attention_type == 'notinformed':
			self.get_informed = self.pass_informed

		
		
	def forward(self, parag_embedded, sent_embedded, word_embedded, sent_mask, p_lengths,labels=None):
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
		# par_mask = torch.zeros((batch_size, p_len), dtype=bool, device= word_embedded.device, requires_grad=False)
		par_mask = get_parmask(p_lengths, batch_size, p_len)
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
		informed = self.just_informed(query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len)
		# g = torch.sigmoid(self.info_score(informed)+self.sent_score(sent_embedded))
		g = torch.sigmoid(self.info_score(torch.cat((informed,sent_embedded),-1)))
		informed = self.informed_layernorm(g*informed+(1-g)*sent_embedded)
		# informed = self.info_score(torch.cat((informed,sent_embedded),-1))
		return informed
	
	def cat_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		informed = self.just_informed(query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len)
		# g = torch.sigmoid(self.info_score(informed)+self.sent_score(sent_embedded))
		# g = torch.sigmoid(self.info_score(torch.cat((informed,sent_embedded),-1)))
		# informed = self.informed_layernorm(g*informed+(1-g)*sent_embedded)
		informed = self.info_score(torch.cat((informed,sent_embedded),-1))
		return informed
	
	def pass_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		return sent_embedded
	
	def init_hidden(self, batch_size,device):
		# num_layers * num_directions, batch, hidden_size
		if self.rnn_type =='LSTM':
			h_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			c_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			return (h_n, c_n)

class BidirectionalPointer(nn.Module):
	"""docstring for BidirectionalPointer"""
	def __init__(self, config):
		super(BidirectionalPointer, self).__init__()

		self.embedding_dim = config.embedding_dim
		# self.dropout_value = config.rnn_dropout_value
		self.dropout_value = config.rnnout_dropout_value
		# self.informed_attention_type = config.informed_attention_type
		self.pointer_drop = config.pointer_drop
		self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
		self.forward_pointer = Attention(self.embedding_dim, self.embedding_dim, dropout_value=self.pointer_drop, need_attn=False)
		self.backward_pointer = Attention(self.embedding_dim, self.embedding_dim, dropout_value=self.pointer_drop, need_attn=False)
		
	def forward(self, parag_embedded, sent_embedded, word_embedded, sent_mask, p_lengths,labels=None):
		# parag_embedded 	: (bs,emsize)
		# sent_embedded 	: (bs,p_len,emsize)
		# word_embedded 	: (bs,p_len,s_len,emsize)
		# sent_mask		 	: (bs*p_len,s_len)
		bs, p_len,s_len,emsize = word_embedded.shape
		# q = sent_embedded.view(bs*p_len,emsize)
		# q = q.unsqueeze(1).repeat(1,p_len,1)
		q = sent_embedded.unsqueeze(1).repeat(1,p_len,1,1).view(bs*p_len,p_len,emsize)
		# q = query.unsqueeze(1).repeat(1,p_len,1).view(batch_size*p_len,1,self.rnn_hidden_size)
		s = word_embedded.view(bs*p_len,s_len,self.embedding_dim)
		# q 	: (bs*p_len, p_len, emsize)
		# s 	: (bs*p_len, s_len, emsize)

		
		informed = self.informed_attn(q,s,sent_mask)[0]
		# informed: 	(bs*p_len, p_len, emsize)
		# informed = informed.view(bs,p_len,p_len,emsize)

		#forward

		q = sent_embedded.view(bs*p_len,emsize)
		# q = sent_embedded[:,1,:]
		# s = sent_embedded.unsqueeze(1).repeat(1,p_len,1,1).view(bs*p_len,p_len,emsize)
		# s = sent_embedded
		# informed = s
		# qparag = parag_embedded.unsqueeze(1).repeat(1,p_len,1).view(bs*p_len,emsize)
		# qsent & qparag -> q
		# q = qsent
		par_mask = None
		attn_weights_forward = self.forward_pointer(q, informed, mask=par_mask)[0]
		attn_weights_backward = self.backward_pointer(q, informed, mask=par_mask)[0]
		# print(attn_weights_forward.shape)


		return attn_weights_forward.view(bs,p_len,p_len), attn_weights_backward.view(bs,p_len,p_len)

class DoublePointer(nn.Module):
	"""docstring for DoublePointer

	Inputs:parag_embedded, sent_embedded, word_embedded
	parag_embedded	()
	sent_embedded	()
	word_embedded	(bs,slen,emsize)
	"""
	def __init__(self, config):
		super(DoublePointer, self).__init__()
		self.rnn_type = 'LSTM'
		self.embedding_dim = config.embedding_dim
		self.rnn_hidden_size = config.rnn_hidden_size
		self.rnn_n_layers = config.rnn_n_layers
		self.rnn_dropout_value = config.rnn_dropout_value
		self.dropout_value = config.rnnout_dropout_value
		self.informed_attention_type = config.informed_attention_type
		self.pointer_drop = config.pointer_drop



		self.rnnfw = getattr(nn, self.rnn_type)(self.embedding_dim, self.rnn_hidden_size, self.rnn_n_layers, 
			dropout=self.rnn_dropout_value, bidirectional=False)
		self.rnnbw = getattr(nn, self.rnn_type)(self.embedding_dim, self.rnn_hidden_size, self.rnn_n_layers, 
			dropout=self.rnn_dropout_value, bidirectional=False)

		
		self.after_rnn_drop = nn.Dropout(self.dropout_value)
		self.pointerfw = Attention(self.rnn_hidden_size, self.embedding_dim, dropout_value=self.pointer_drop, need_attn=False)
		self.pointerbw = Attention(self.rnn_hidden_size, self.embedding_dim, dropout_value=self.pointer_drop, need_attn=False)

		self.parag_to_hidden = nn.Linear(self.embedding_dim,self.rnn_hidden_size)

		if self.informed_attention_type == 'informed':
			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.just_informed

			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.just_informed


		elif self.informed_attention_type == 'addinformed':
			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.add_informed
			self.info_score = nn.Linear(2*self.embedding_dim,1)

			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.add_informed
			self.info_score = nn.Linear(2*self.embedding_dim,1)


		elif self.informed_attention_type == 'catinformed':
			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.cat_informed
			self.info_score = nn.Linear(2*self.embedding_dim,self.embedding_dim)

			self.informed_attn = TransformerDecoderBlock(config.informed_attention_config)
			self.informed_layernorm = nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-12)
			self.get_informed = self.cat_informed
			self.info_score = nn.Linear(2*self.embedding_dim,self.embedding_dim)


		elif self.informed_attention_type == 'notinformed':
			self.get_informed = self.pass_informed

			self.get_informed = self.pass_informed

		
		
	def forward(self, parag_embedded, sent_embedded, word_embedded, sent_mask, p_lengths,labels=None):
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
		# par_mask = torch.zeros((batch_size, p_len), dtype=bool, device= word_embedded.device, requires_grad=False)
		par_mask = get_parmask(p_lengths, batch_size, p_len)
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
		informed = self.just_informed(query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len)
		# g = torch.sigmoid(self.info_score(informed)+self.sent_score(sent_embedded))
		g = torch.sigmoid(self.info_score(torch.cat((informed,sent_embedded),-1)))
		informed = self.informed_layernorm(g*informed+(1-g)*sent_embedded)
		# informed = self.info_score(torch.cat((informed,sent_embedded),-1))
		return informed
	
	def cat_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		informed = self.just_informed(query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len)
		# g = torch.sigmoid(self.info_score(informed)+self.sent_score(sent_embedded))
		# g = torch.sigmoid(self.info_score(torch.cat((informed,sent_embedded),-1)))
		# informed = self.informed_layernorm(g*informed+(1-g)*sent_embedded)
		informed = self.info_score(torch.cat((informed,sent_embedded),-1))
		return informed
	
	def pass_informed(self, query, word_embedded, sent_embedded ,sent_mask, batch_size, p_len, s_len):
		return sent_embedded
	
	def init_hidden(self, batch_size,device):
		# num_layers * num_directions, batch, hidden_size
		if self.rnn_type =='LSTM':
			h_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			c_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			return (h_n, c_n)