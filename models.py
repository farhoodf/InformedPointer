import torch
import torch.nn as nn

from modules import WordEncoder, SentenceEncoder, ParagEncoder, InformedPointer, BidirectionalPointer

def xavier_init_weights(m):
	if type(m) == nn.Linear:
		nn.init.xavier_uniform_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0.01)
class FromBert(nn.Module):
	"""docstring for FromBert"""
	def __init__(self, config, bidirectional=False):
		super(FromBert, self).__init__()
		self.dropout_value = config.dropout_value

		self.word_encoder = WordEncoder(config.word_encoder_config)

		self.dense = nn.Linear(config.bert_dim,config.dim)

		self.sentence_encoder = SentenceEncoder(config.sentence_encoder_conf)

		self.parag_encoder = ParagEncoder(config.parag_encoder_conf)
		if not bidirectional:
			self.pointer = InformedPointer(config.pointer_conf)
		else:
			self.pointer = BidirectionalPointer(config.pointer_conf)
		
		self.word_drop = nn.Dropout(self.dropout_value)
		self.sent_drop = nn.Dropout(self.dropout_value)
		# self._init_weights()
	
	def forward(self, x, s_lengths, p_lengths, labels=None):
		batch_size, p_len, s_len = x.size()

		#changing view
		x_ = x.view(p_len*batch_size, s_len)
		s_lengths_ = s_lengths.view(p_len*batch_size)

		# word_embedded: (bs*p_len, s_len, 768)
		# sent_embedded: (bs*p_len, 768)
		
		word_embedded, sent_mask = self.word_encoder(x_, s_lengths_)
		word_embedded = self.dense(word_embedded)
		word_embedded = self.word_drop(word_embedded)

		sent_embedded = self.sentence_encoder(word_embedded, mask=sent_mask)
		# sent_embedded, weights = self.sentence_encoder(word_embedded, mask=sent_mask)
		sent_embedded = self.sent_drop(sent_embedded)


		word_embedded = word_embedded.view(batch_size, p_len, s_len, -1)
		sent_embedded = sent_embedded.view(batch_size, p_len, -1)
		#sent_embedded: (bs, p_len, em)

		parag_embedded, sent_embedded = self.parag_encoder(sent_embedded)
		# parag_embedded: (bs, em)

		out = self.pointer(parag_embedded, sent_embedded, word_embedded, sent_mask, p_lengths, labels)

		return out
		# return out, weights

	def _init_weights(self,):
		self.dense.apply(xavier_init_weights)
		self.sentence_encoder.apply(xavier_init_weights)
		self.parag_encoder.apply(xavier_init_weights)
		self.pointer.apply(xavier_init_weights)
