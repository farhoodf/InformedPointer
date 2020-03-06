
token_config={
	'unknown' : '__unk__',
	'pad_token': '__pad__',
	'first_sent':'__start__',
	'last_sent':'__end__'
}

class TransformerConfig(object):
	"""docstring for TransformerConfig"""
	def __init__(self, dim):
		super(TransformerConfig, self).__init__()
		self.n_layers = 2
		self.n_heads = 4
		self.dim = dim
		self.hidden_dim = 256

		self.dropout = 0.3
		self.attention_dropout = 0.3


		self.activation = 'gelu'
		self.output_attentions = False
		self.output_hidden_states = False

class PointerConfig(object):
	"""docstring for PointerConfig"""
	def __init__(self, dim):
		super(PointerConfig, self).__init__()
		self.rnn_hidden_size = dim
		self.rnn_dropout_value = 0
		self.rnn_n_layers = 1

		self.pointer_dim = dim
		self.dropout_value = 0.3

		self.informed_attention_config = TransformerConfig(dim)


class SentenceEncoderConfig(object):
	"""docstring for SentenceEncoderConfig"""
	def __init__(self, dim):
		super(SentenceEncoderConfig, self).__init__()
		self.multihead_conf = TransformerConfig(dim)

class ParagEncoderConfig(object):
		"""docstring for ParagEncoderConfig"""
		def __init__(self,dim):
			super(ParagEncoderConfig, self).__init__()
			self.multihead_conf = TransformerConfig(dim)
			self.transformer = TransformerConfig(dim)
				

class FromBert(object):
	"""docstring for FromBert"""
	def __init__(self, dim=256, word_encoder_type = 'bert'):
		super(FromBert, self).__init__()
		self.word_encoder_type = word_encoder_type # {'bert', 'distill'}
		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

		self.dropout_value = 0.3
		self.dim = dim
		self.sentence_encoder_conf = SentenceEncoderConfig(dim = self.dim)
		self.parag_encoder_conf = ParagEncoderConfig(dim = self.dim)
		self.informed_pointer_conf = PointerConfig(dim = self.dim)		

class FromElmo(object):
	"""docstring for FromElmo"""
	def __init__(self, dim=512):
		super(FromElmo, self).__init__()
		self.word_encoder_type = 'bert' # {'bert', 'distill'}
		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

		self.n_tokens = 400004
		self.word_embeddeding_size = 100
		
		self.dropout_value = 0.3
		self.dim = dim
		self.sentence_encoder_conf = SentenceEncoderConfig(dim = self.dim)
		self.parag_encoder_conf = ParagEncoderConfig(dim = self.dim)
		self.informed_pointer_conf = PointerConfig(dim = self.dim)

		if False:
			self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json" 
			self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
			self.elmo_size = 256
		else:
			self.options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
			self.weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
			self.elmo_size = 1024


class FromGlove(object):
	"""docstring for FromGlove"""
	def __init__(self, dim=256):
		super(FromGlove, self).__init__()
		self.word_encoder_type = 'bert' # {'bert', 'distill'}
		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

		self.n_tokens = 400004
		self.word_embeddeding_size = 100
		
		self.dropout_value = 0.5
		self.dim = dim
		self.sentence_encoder_conf = SentenceEncoderConfig(dim = self.dim)
		self.parag_encoder_conf = ParagEncoderConfig(dim = self.dim)
		self.informed_pointer_conf = PointerConfig(dim = self.dim)


class TrainingConfig(object):
	"""docstring for TrainingConfig"""
	def __init__(self, arg):
		super(TrainingConfig, self).__init__()
		
		