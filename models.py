import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from allennlp.modules.elmo import Elmo, batch_to_ids

from distilTransformer import TransformerEncoder, MultiHeadSelfAttention, FFN

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json" 
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"


def getmask_hugging(s_lengths, batch_size, s_len):
	"""attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
		Mask to avoid performing attention on padding token indices.
		Mask values selected in ``[0, 1]``:
		``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens."""
	mask=torch.zeros((batch_size,s_len+1),dtype=torch.bool,device=s_lengths.device,requires_grad=False)

	mask[torch.arange(batch_size),s_lengths] = 1
	mask = ~mask.cumsum(dim=1).bool().to(device=s_lengths.device)
	return mask[:,:-1]

class FromELMO(nn.Module):
	"""docstring for FromELMO"""
	def __init__(self):
		super(FromELMO, self).__init__()
		self.dropout_value = 0.3

		# self.word_encoder = BertModel.from_pretrained('bert-base-uncased')
		self.word_encoder = Elmo(options_file, weight_file, 1, dropout=self.dropout_value)
		self.sentence_encoder = SentenceEncoder(256)
		self.parag_encoder = ParagEncoder()
		self.pointer = InformedPointer(embedding_size=256)
		
		self.word_drop = nn.Dropout(self.dropout_value)
		self.sent_drop = nn.Dropout(self.dropout_value)
	
	def forward(self, x, s_lengths, p_lengths, labels=None):
		batch_size, p_len, s_len,_ = x.size()

		#changing view
		# x_ = x.view(p_len*batch_size, s_len)
		# s_lengths_ = s_lengths.view(p_len*batch_size)
		# sent_mask = getmask_hugging(s_lengths_, batch_size*p_len, s_len)
		
		elmo_out = self.word_encoder(x)
		word_embedded = elmo_out['elmo_representations'][0].view(p_len*batch_size, s_len,-1)

		sent_mask = elmo_out['mask'].bool()
		word_embedded = self.word_drop(word_embedded)

		# sent_embedded = sent_embedded.view(batch_size, p_len, -1)
		sent_embedded = self.sentence_encoder(word_embedded, mask=sent_mask.view(p_len*batch_size, s_len))
		sent_embedded = self.sent_drop(sent_embedded)


		word_embedded = word_embedded.view(batch_size, p_len, s_len, -1)
		sent_embedded = sent_embedded.view(batch_size, p_len, -1)
		
		parag_embedded = self.parag_encoder(sent_embedded)


		out = self.pointer(parag_embedded, sent_embedded, word_embedded, sent_mask, labels)

		return out

class FromBert(nn.Module):
	"""docstring for FromBert"""
	def __init__(self, config):
		super(FromBert, self).__init__()
		self.dropout_value = config.dropout_value

		self.word_encoder = WordEncoder(config.word_encoder_type)

		# self.sentence_encoder = SentenceEncoder(768)
		# self.sentence_encoder = RNNSentenceEncoder(768,768)
		self.sentence_encoder = MultiHeadSentenceEncoder(config.sentence_encoder_conf)

		# self.parag_encoder = ParagEncoder()
		self.parag_encoder = TransformerParagEncoder(config.parag_encoder_conf)
		self.pointer = InformedPointer(config.informed_pointer_conf)
		
		self.word_drop = nn.Dropout(self.dropout_value)
		self.sent_drop = nn.Dropout(self.dropout_value)
	
	def forward(self, x, s_lengths, p_lengths, labels=None):
		batch_size, p_len, s_len = x.size()

		#changing view
		x_ = x.view(p_len*batch_size, s_len)
		s_lengths_ = s_lengths.view(p_len*batch_size)
		# sent_mask = getmask_hugging(s_lengths_, batch_size*p_len, s_len)
		# sent_mask = None

		# word_embedded: (bs*p_len, s_len, 768)
		# sent_embedded: (bs*p_len, 768)
		
		word_embedded, sent_mask = self.word_encoder(x_, s_lengths_)
		word_embedded = self.word_drop(word_embedded)

		# sent_embedded = sent_embedded.view(batch_size, p_len, -1)
		# sent_embedded = self.sentence_encoder(word_embedded, mask=sent_mask) # single head
		# sent_embedded = self.sentence_encoder(word_embedded, lengths=s_lengths_) # rnn
		sent_embedded = self.sentence_encoder(word_embedded, mask=sent_mask) # multihead
		sent_embedded = self.sent_drop(sent_embedded)


		word_embedded = word_embedded.view(batch_size, p_len, s_len, -1)
		sent_embedded = sent_embedded.view(batch_size, p_len, -1)
		
		
		
		#sent_embedded: (bs, p_len, em)

		
		
		# parag_embedded = self.parag_encoder(sent_embedded)
		parag_embedded, sent_embedded = self.parag_encoder(sent_embedded)
		# parag_embedded: (bs, em)

		out = self.pointer(parag_embedded, sent_embedded, word_embedded, sent_mask, labels)

		return out




class WordEncoder(nn.Module):
	"""docstring for WordEncoder"""
	def __init__(self, encoder_type='bert'):
		super(WordEncoder, self).__init__()
		self.encoder_type=encoder_type
		if self.encoder_type == 'bert':
			self.encoder = BertModel.from_pretrained('bert-base-uncased')

	def forward(self, x, s_lengths):
		batch_size, s_len = x.shape
		mask = getmask_hugging(s_lengths, batch_size, s_len)
		word_embedded, _ = self.encoder(x, attention_mask = mask)

		return word_embedded, mask



		

class SingleHeadSentenceEncoder(nn.Module):
	"""docstring for SentenceEncoder"""
	def __init__(self, embedding_size ):
		super(SentenceEncoder, self).__init__()
		
		self.attn = Attention(embedding_size, embedding_size, internal_dim=None, need_attn=True)
		self.query = nn.Linear(1,embedding_size, bias=False)

	
	def forward(self, word_embedded, mask):
		ones = torch.ones((word_embedded.shape[0],1),device=word_embedded.device)
		q = self.query(ones)
		_, sent_embedded = self.attn(q, word_embedded, mask=~mask)
		return sent_embedded

class RNNSentenceEncoder(nn.Module):
	"""docstring for RNNSentenceEncoder"""
	def __init__(self, input_size, hidden_size, n_layers=1, rnn_type='LSTM', bidirectional=True, dropout=0.3):
		super(RNNSentenceEncoder, self).__init__()
		self.rnn_type = rnn_type
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dropout = dropout
		self.bidirectional = bidirectional
		self.batch_first = True
		
		self.rnn = getattr(nn, self.rnn_type)(self.input_size, self.hidden_size, self.n_layers, 
			dropout=self.dropout, bidirectional=self.bidirectional, batch_first=self.batch_first)
		self.dense = nn.Linear(2*self.hidden_size,self.hidden_size)
		# self.rnn.apply(init_weights)
		

	def forward(self, inputs, hiddens=None, lengths=None):#, enforce_sorted=True):
		# lengths = None
		if lengths is not None:
			inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, 
						batch_first=self.batch_first, enforce_sorted=False)
		
		outputs, _ = self.rnn(inputs, hiddens)

		if lengths is not None:
			outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)

		# if self.bidirectional:
		# 	if self.batch_first
		o = torch.zeros_like(outputs[:,0,:])
		o[:,:self.hidden_size] = outputs[:,0,self.hidden_size:]
		o[:,self.hidden_size:] = outputs[:,-1,:self.hidden_size]
				
			# else:
			# 	o = torch.zeros_like(outputs[0,:,:])
			# 	o[:,:self.hidden_size] = outputs[0,:,self.hidden_size:]
			# 	o[:,self.hidden_size:] = outputs[-1,:,:self.hidden_size]
		outputs = self.dense(o)

		return outputs

class MultiHeadSentenceEncoder(nn.Module):
	"""docstring for MultiHeadSentenceEncoder"""
	def __init__(self, config):
		super(MultiHeadSentenceEncoder, self).__init__()
		self.config = config
		self.encoder = MultiHeadAttention(self.config.multihead_conf)
		self.query = nn.Linear(1, self.config.multihead_conf.dim, bias=False)

	def forward(self, word_embedded, mask):
		ones = torch.ones((word_embedded.shape[0],1,1),device=word_embedded.device)
		q = self.query(ones)
		sent_embedded = self.encoder(q, word_embedded, mask)[0]
		sent_embedded = sent_embedded.squeeze(1)
		return sent_embedded
		
class TransformerParagEncoder(nn.Module):
	"""docstring for TransformerParagEncoder"""
	def __init__(self, config):
		super(TransformerParagEncoder, self).__init__()

		
		self.query = nn.Linear(1, config.multihead_conf.dim, bias=False)
		self.transformer = TransformerEncoder(config.transformer)
		self.encoder = MultiHeadAttention(config.multihead_conf)

	def forward(self, sent_embedded, mask=None):
		if mask is None:
			bs, p_len, _ = sent_embedded.shape
			mask = torch.ones(bs, p_len).to(device=sent_embedded.device)

		sent_embedded = self.transformer(sent_embedded, mask)[0]

		ones = torch.ones((sent_embedded.shape[0],1,1),device=sent_embedded.device)
		q = self.query(ones)
		parag_embedded = self.encoder(q, sent_embedded, mask)[0]
		parag_embedded = parag_embedded.squeeze(1)
		return parag_embedded, sent_embedded

class AvgParagEncoder(nn.Module):
	"""docstring for ParagEncoder"""
	def __init__(self):
		super(ParagEncoder, self).__init__()
		
	def forward(self, sent_embedded):
		return sent_embedded.sum(1)



	
	
class InformedPointer(nn.Module):
	"""docstring for InformedPointer

	Inputs:parag_embedded, sent_embedded, word_embedded
	parag_embedded	()
	sent_embedded	()
	word_embedded	(bs,slen,emsize)
	"""
	def __init__(self, config):
		super(InformedPointer, self).__init__()
		embedding_size = 768
		self.rnn_type = 'LSTM'
		self.embedding_dim = config.pointer_dim
		self.rnn_hidden_size = config.rnn_hidden_size
		self.rnn_n_layers = config.rnn_n_layers
		self.rnn_dropout_value = config.rnn_dropout_value
		self.dropout_value = config.dropout_value



		self.rnn = getattr(nn, self.rnn_type)(self.embedding_dim, self.rnn_hidden_size, self.rnn_n_layers, 
			dropout=self.rnn_dropout_value, bidirectional=False)
		
		self.after_rnn_drop = nn.Dropout(self.dropout_value)

		# self.informed_attn = Attention(embedding_dim, embedding_dim)
		# self.informed_query = Attention(embedding_dim, embedding_dim)
		self.informed_attn = MultiHeadAttention(config.informed_attention_config)

		self.pointer = Attention(self.embedding_dim, self.embedding_dim,need_attn=False)
		
	def forward(self, parag_embedded, sent_embedded, word_embedded, sent_mask, labels=None):
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
			hidden = (self.after_rnn_drop(hidden[0]), self.after_rnn_drop(hidden[1]))
			
			# get informred keys
			q = hidden[0][-1].unsqueeze(1).repeat(1,p_len,1).view(batch_size*p_len,1,self.rnn_hidden_size)
			s = word_embedded.view(batch_size*p_len,s_len,self.embedding_dim)
			# _, informred = self.informed_attn(q, s) #hidden or selected? # single head
			informred = self.informed_attn(q, s, sent_mask)[0] #hidden or selected?
			informred = informred.view(batch_size,p_len,self.embedding_dim)
			# informred = informred + sent_embedded

			# informred = sent_embedded
		# 	# point to the next index
			attn_weights = self.pointer(hidden[0][-1], informred, mask=par_mask)[0]
			# attn_weights = self.pointer(hidden[0][-1], sent_embedded, mask=par_mask)[0]

			

			if do_tf:
				index = labels[:,i]
			else:
				index = attn_weights.argmax(1)

			s = word_embedded[torch.arange(batch_size),index]
			q = hidden[0][-1]

			# _, informed_query = self.informed_query(q,s)
			selected = sent_embedded[torch.arange(batch_size),index] #+ informed_query
			selected = selected.unsqueeze(0)

			par_mask = par_mask.clone().detach()
			par_mask[torch.arange(batch_size),index] = 1


			res.append(attn_weights)

		return torch.stack(res).transpose(0,1)
	
	def init_hidden(self, batch_size,device):
		# num_layers * num_directions, batch, hidden_size
		if self.rnn_type =='LSTM':
			h_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			c_n = torch.zeros(self.rnn_n_layers, batch_size, self.rnn_hidden_size,device=device)
			return (h_n, c_n)

class InformedTransformer(nn.Module):
	"""docstring for InformedTransformer"""
	def __init__(self, arg):
		super(InformedTransformer, self).__init__()
		self.arg = arg
		

class MultiHeadAttention(nn.Module):
	"""docstring for MultiHeadAttention"""
	def __init__(self, config):
		super(MultiHeadAttention, self).__init__()
		self.output_attentions = config.output_attentions

		assert config.dim % config.n_heads == 0

		self.attention = MultiHeadSelfAttention(config)
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
		self.dropout_value = 0.2

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

	# def forward(self, query, states):
	# 	bs,p_len,s_len,em = states.shape
	# 	# q = F.relu(self.proj_q(query))
	# 	# k = F.relu(self.proj_k(states))
	# 	q = self.proj_q(query)
	# 	# q = F.relu(self.proj_q(query)).unsqueeze(1).repeat(1,slen,1)
	# 	k = self.proj_k(states)

	# 	q = q.unsqueeze(1).unsqueeze(1).repeat(1,p_len,s_len,1)

	# 	energy = self.proj_e(torch.tanh(torch.cat((q,k),-1)))
	# 	attn_weights = F.softmax(energy,dim=1)

	# 	out = (attn_weights.squeeze(-1),)

	# 	if self.need_attn:
	# 		# v = F.relu(self.proj_v(states))
	# 		v = states
	# 		attn = torch.mul(attn_weights.repeat(1,1,1,em),v).sum(-1)
	# 		out += (attn,)

	# 	return out