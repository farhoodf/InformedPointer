
token_config={
	'unknown' : '__unk__',
	'pad_token': '__pad__',
	'first_sent':'__start__',
	'last_sent':'__end__'
}


class WordEncoderConfig(object):
	"""docstring for WordEncoderConfig"""
	def __init__(self, 
		encoder_type='bert', #{bert,distil,rnn}
		dropout=0.3, #only for rnn
		rnn_config=None,
		n_tokens=400004,
		word_embedding_size=100
		):
		super(WordEncoderConfig, self).__init__()
		self.encoder_type = encoder_type
		self.dropout = dropout
		self.rnn_config = rnn_config
		self.n_tokens = n_tokens
		self.word_embedding_size = word_embedding_size
		
class SentenceEncoderConfig(object):
	"""docstring for SentenceEncoderConfig"""
	def __init__(self, 
			encoder_type='attention', #{attention, rnn}
			rnn_config = None,
			multihead_conf = None
		):
		super(SentenceEncoderConfig, self).__init__()
		self.encoder_type = encoder_type
		self.rnn_config = rnn_config
		self.multihead_conf = multihead_conf
		
class ParagEncoderConfig(object):
		"""docstring for ParagEncoderConfig"""
		def __init__(self,
				encoder_type='transformer', #{transformer}
				multihead_conf=None,
				transformer_conf=None):
			super(ParagEncoderConfig, self).__init__()
			self.encoder_type = encoder_type
			self.multihead_conf = multihead_conf
			self.transformer_conf = transformer_conf

class RNNConfig(object):
	"""docstring for RNNConfig"""
	def __init__(self,
				input_size,
				hidden_size,
				n_layers=1,
				rnn_type='LSTM',
				bidirectional=True,
				dropout=0.3):
		super(RNNConfig, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		self.dropout = dropout
		self.batch_first = True
		

class TransformerConfig(object):
	"""docstring for TransformerConfig"""
	def __init__(self, 
				dim = 128,
				n_heads = 4,
				hidden_dim = 512,
				qdim = None,
				ffn_dropout = 0.3,
				attention_dropout = 0.3,
				activation = 'gelu',
				output_attentions = False,
				output_hidden_states = False,
				n_layers = None,
		):
		super(TransformerConfig, self).__init__()
		self.n_layers = n_layers
		self.n_heads = n_heads
		self.dim = dim
		self.hidden_dim = hidden_dim

		self.ffn_dropout = ffn_dropout
		self.attention_dropout = attention_dropout


		self.activation = activation
		self.output_attentions = output_attentions
		self.output_hidden_states = output_hidden_states
		if qdim is None:
			self.qdim = dim
		else:
			self.qdim = qdim

class PointerConfig(object):
	"""docstring for PointerConfig"""
	def __init__(self, 
			rnn_hidden_size = 256,
			rnn_n_layers = 1,
			rnn_dropout_value = 0,
			rnnout_dropout_value = 0.2,
			pointer_drop = 0.3,
			embedding_dim = 768,
			informed_attention_config = None,
			informed_attention_type = 'informed' #{informed,addinformed,notinformed}
			):
		super(PointerConfig, self).__init__()
		self.rnn_hidden_size = rnn_hidden_size
		self.rnn_dropout_value = rnn_dropout_value
		self.rnn_n_layers = rnn_n_layers

		self.embedding_dim = embedding_dim
		self.rnnout_dropout_value = rnnout_dropout_value
		self.pointer_drop = pointer_drop

		if informed_attention_type in {'informed','addinformed'}:
			assert informed_attention_config != None
		else:
			assert informed_attention_config == None

		self.informed_attention_config = informed_attention_config
		self.informed_attention_type = informed_attention_type

class FromBertConfig(object):
	"""docstring for FromBertConfig"""
	def __init__(self, gen_dim = 256, rnn_dim = 64):
		super(FromBertConfig, self).__init__()
		
		self.bert_dim = 768
		# self.bert_dim = 1024
		self.dim = gen_dim
		self.word_encoder_config = WordEncoderConfig(encoder_type='bert-base', #{bert-base,bert-large,distil,rnn}
													dropout=0.2, #only for rnn
													rnn_config=None,
													n_tokens=400004,
													word_embedding_size=100)
		sent_multihead_conf = TransformerConfig(
				dim = gen_dim,
				n_heads = 4,
				hidden_dim = gen_dim*2,
				ffn_dropout = 0.2,
				attention_dropout = 0.2,
				)
		self.sentence_encoder_conf = SentenceEncoderConfig(
										encoder_type='attention', #{attention, rnn}
										rnn_config = None,
										multihead_conf = sent_multihead_conf
										)
		

		parag_multihead_conf = TransformerConfig(
				dim = gen_dim,
				n_heads = 4,
				hidden_dim = gen_dim*2,
				qdim = None,
				ffn_dropout = 0.2,
				attention_dropout = 0.2,
				)
		parag_trans_conf = TransformerConfig(
				dim = gen_dim,
				n_heads = 4,
				hidden_dim = gen_dim*2,
				qdim = None,
				ffn_dropout = 0.2,
				attention_dropout = 0.2,
				n_layers = 2
				)
		self.parag_encoder_conf = ParagEncoderConfig(
										encoder_type='transformer', #{transformer}
										multihead_conf=parag_multihead_conf,
										transformer_conf=parag_trans_conf
										)

		informed_attn = TransformerConfig(
				dim = gen_dim,
				n_heads = 4,
				hidden_dim = gen_dim*2,
				qdim = rnn_dim,
				ffn_dropout = 0.2,
				attention_dropout = 0.2,
				)
		self.pointer_conf = PointerConfig(
										rnn_hidden_size = rnn_dim,
										rnn_n_layers = 1,
										rnn_dropout_value = 0,
										rnnout_dropout_value = 0.2,
										pointer_drop = 0.2,
										embedding_dim = gen_dim,
										informed_attention_config = informed_attn,
										informed_attention_type = 'informed'
										)
		self.dropout_value = 0.3


		

# class FromBert___(object):
# 	"""docstring for FromBert"""
# 	def __init__(self, dim=256, word_encoder_type = 'bert'):
# 		super(FromBert, self).__init__()
# 		self.word_encoder_type = word_encoder_type # {'bert', 'distill'}
# 		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
# 		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

# 		self.dropout_value = 0.3
# 		self.dim = dim
# 		self.sentence_encoder_conf = SentenceEncoderConfig(dim = self.dim)
# 		self.parag_encoder_conf = ParagEncoderConfig(dim = self.dim)
# 		self.informed_pointer_conf = PointerConfig(dim = self.dim)		

# class FromElmo(object):
# 	"""docstring for FromElmo"""
# 	def __init__(self, dim=512):
# 		super(FromElmo, self).__init__()
# 		self.word_encoder_type = 'bert' # {'bert', 'distill'}
# 		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
# 		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

# 		self.n_tokens = 400004
# 		self.word_embedding_size = 100
		
# 		self.dropout_value = 0.3
# 		self.dim = dim
# 		self.sentence_encoder_conf = SentenceEncoderConfig(dim = self.dim)
# 		self.parag_encoder_conf = ParagEncoderConfig(dim = self.dim)
# 		self.informed_pointer_conf = PointerConfig(dim = self.dim)

# 		if False:
# 			self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json" 
# 			self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
# 			self.elmo_size = 256
# 		else:
# 			self.options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# 			self.weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# 			self.elmo_size = 1024


# class FromGlove(object):
# 	"""docstring for FromGlove"""
# 	def __init__(self, dim=256):
# 		super(FromGlove, self).__init__()
# 		self.word_encoder_type = 'bert' # {'bert', 'distill'}
# 		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
# 		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

# 		self.n_tokens = 400004
# 		self.word_embedding_size = 100
		
# 		self.dropout_value = 0.5
# 		self.dim = dim
# 		self.sentence_encoder_conf = SentenceEncoderConfig(dim = self.dim)
# 		self.parag_encoder_conf = ParagEncoderConfig(dim = self.dim)
# 		self.informed_pointer_conf = PointerConfig(dim = self.dim)


class TrainingConfig(object):
	"""docstring for TrainingConfig"""
	def __init__(self, arg):
		super(TrainingConfig, self).__init__()
		
