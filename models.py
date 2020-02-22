import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel
import torch.nn.functional as F

class FromBert(nn.Module):
	"""docstring for FromBert"""
	def __init__(self):
		super(FromBert, self).__init__()
		self.dropout_value = 0.3

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.pointer = InformedPointer()
		self.word_drop = nn.Dropout(self.dropout_value)
		self.sent_drop = nn.Dropout(self.dropout_value)
	def forward(self, x, s_lengths, p_lengths, labels=None):
		batch_size, p_len, s_len = x.size()

		#changing view
		x_ = x.view(p_len*batch_size, s_len)
		s_lengths_ = s_lengths.view(p_len*batch_size)
		sent_mask = self.getmask(s_lengths_, batch_size*p_len, s_len)
		# sent_mask = None

		# word_embedded, sent_embedded = self.bert(x_, attention_mask = sent_mask)
		# sent_embedded is the hidden state corresponding to the first token and probably useless
		# word_embedded: (bs*p_len, s_len, 768)
		# sent_embedded: (bs*p_len, 768)
		
		word_embedded, _ = self.bert(x_, attention_mask = sent_mask)
		word_embedded = self.word_drop(word_embedded)
		word_embedded = word_embedded.view(batch_size, p_len, s_len, -1)
		
		# sent_embedded = sent_embedded.view(batch_size, p_len, -1)
		sent_embedded = self.sentence_embedding(word_embedded)
		
		# sent_embedded = self.sent_drop(sent_embedded)
		#sent_embedded: (bs, p_len, em)
		# print('word_embedded',word_embedded.shape)
		# print('sent_embedded',sent_embedded.shape)
		
		# maybe transformer to get relative embedding
		
		parag_embedded = self.get_parag_embedding(sent_embedded)
		# print('parag_embedded',parag_embedded.shape)
		# parag_embedded: (bs, em)

		out = self.pointer(parag_embedded, sent_embedded, word_embedded, sent_mask, labels)

		return out
	def get_parag_embedding(self, sent_embedded):
		return sent_embedded.sum(1)

	def sentence_embedding(self, word_embedded):
		return word_embedded.sum(dim=2)
	
	def getmask(self,s_lengths, batch_size, s_len):
		"""attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
			Mask to avoid performing attention on padding token indices.
			Mask values selected in ``[0, 1]``:
			``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens."""
		mask=torch.zeros((batch_size,s_len+1),dtype=torch.bool,device=s_lengths.device,requires_grad=False)
		# print('mask',mask.shape)
		# print(s_lengths)
		mask[torch.arange(batch_size),s_lengths] = 1
		mask = ~mask.cumsum(dim=1).bool().to(device=s_lengths.device)
		return mask[:,:-1]

class InformedPointer(nn.Module):
	"""docstring for InformedPointer

	Inputs:parag_embedded, sent_embedded, word_embedded
	parag_embedded	()
	sent_embedded	()
	word_embedded	(bs,slen,emsize)
	"""
	def __init__(self, embedding_size = 768):
		super(InformedPointer, self).__init__()
		self.rnn_type = 'LSTM'
		self.embedding_size = 768
		self.n_layers = 1
		self.dropout_value = 0.3


		self.rnn = getattr(nn, self.rnn_type)(self.embedding_size, self.embedding_size, self.n_layers, 
			dropout=self.dropout_value, bidirectional=False)
		self.informed_values = Attention(embedding_size, embedding_size)
		self.pointer = Attention(embedding_size,embedding_size,need_attn=False)
	

	def forward(self, parag_embedded, sent_embedded, word_embedded, sent_mask, labels):
		batch_size, p_len, s_len, _ = word_embedded.shape
		if self.training:
			do_tf = True 
		else:
			do_tf = False
		
		res = []
		hidden = self.init_hidden(batch_size,parag_embedded.device)
		hidden[0][-1] = parag_embedded
		selected = torch.zeros_like(hidden[0][-1:])
		par_mask = torch.zeros((batch_size, p_len), dtype=bool, device= word_embedded.device, requires_grad=False)
		for i in range(p_len):
			_, hidden = self.rnn(selected,hidden)
			
			# get informred keys
			q = hidden[0][-1].unsqueeze(1).repeat(1,p_len,1).view(batch_size*p_len,self.embedding_size)
			s = word_embedded.view(batch_size*p_len,s_len,self.embedding_size)

			_, informred = self.informed_values(q, s) #hidden or selected?
			informred = informred.view(batch_size,p_len,self.embedding_size)


		# 	# point to the next index
			attn_weights = self.pointer(hidden[0][-1], informred, mask=par_mask)[0]
			# attn_weights = self.pointer(hidden[0][-1], sent_embedded, mask=par_mask)[0]



			if do_tf:
				index = labels[:,i]
			else:
				index = attn_weights.argmax(1)

			selected = sent_embedded[torch.arange(batch_size),index].unsqueeze(0)

			par_mask = par_mask.clone().detach()
			par_mask[torch.arange(batch_size),index] = 1


			res.append(attn_weights)

		return torch.stack(res).transpose(0,1)
	
	def init_hidden(self, batch_size,device):
		# num_layers * num_directions, batch, hidden_size
		if self.rnn_type =='LSTM':
			h_n = torch.zeros(self.n_layers, batch_size, self.embedding_size,device=device)
			c_n = torch.zeros(self.n_layers, batch_size, self.embedding_size,device=device)
			return (h_n, c_n)


class Attention(nn.Module):
	"""docstring for Attention
	Inputs:
	query:		(bs,emsize)
	states:		(bs, slen, dim)
	Output:
	weights:	(bs, slen)
	attn:		(bs,slen)
	"""
	def __init__(self, q_dim, v_dim, internal_dim=None, need_attn=True):
		super(Attention, self).__init__()
		
		if internal_dim == None:
			internal_dim = v_dim
		self.need_attn = need_attn

		self.proj_q = nn.Linear(q_dim, internal_dim)
		self.proj_k = nn.Linear(v_dim, internal_dim)
		self.proj_e = nn.Linear(2*internal_dim,1)



	def forward(self, query, states, mask=None):
		bs,slen,em = states.shape
		q = F.relu(self.proj_q(query)).unsqueeze(1).repeat(1,slen,1)
		k = F.relu(self.proj_k(states))
		energy = self.proj_e(torch.tanh(torch.cat((q,k),-1))).squeeze(2)
		# if not self.need_attn:
		# 	print('energy',energy.shape)
		if mask is not None:
			energy.masked_fill_(mask,1e-45)
		# mask can be applied here to put very low value for energy

		attn_weights = F.softmax(energy,dim=1)

		out = (attn_weights,)

		if self.need_attn:
			# v = F.relu(self.proj_v(states))
			v = states
			attn = torch.mul(attn_weights.unsqueeze(2).repeat(1,1,em),v).sum(1)
			out += (attn,)

		return out

	# def forward(self, query, states):
	# 	bs,p_len,s_len,em = states.shape
	# 	# q = F.relu(self.proj_q(query))
	# 	# k = F.relu(self.proj_k(states))
	# 	q = self.proj_q(query)
	# 	# q = F.relu(self.proj_q(query)).unsqueeze(1).repeat(1,slen,1)
	# 	k = self.proj_k(states)

	# 	print('q',q.shape)
	# 	print('k',k.shape)
	# 	print('q',q)
	# 	q = q.unsqueeze(1).unsqueeze(1).repeat(1,p_len,s_len,1)
	# 	print(q.shape)
	# 	print(k)

	# 	energy = self.proj_e(torch.tanh(torch.cat((q,k),-1)))
	# 	attn_weights = F.softmax(energy,dim=1)
	# 	print('attn_weights',attn_weights.shape)

	# 	out = (attn_weights.squeeze(-1),)

	# 	if self.need_attn:
	# 		# v = F.relu(self.proj_v(states))
	# 		v = states
	# 		attn = torch.mul(attn_weights.repeat(1,1,1,em),v).sum(-1)
	# 		out += (attn,)

	# 	return out