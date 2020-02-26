
token_config={
	'unknown' : '__unk__',
	'pad_token': '__pad__',
	'first_sent':'start',
	'last_sent':'__end__'
}

class TransformerConfig(object):
	"""docstring for TransformerConfig"""
	def __init__(self, ):
		super(TransformerConfig, self).__init__()
		self.n_layers = 2
		self.n_heads = 4
		self.dim = 768
		self.hidden_dim = 1024

		self.dropout = 0.3
		self.attention_dropout = 0.3


		self.activation = 'gelu'
		self.output_attentions = False
		self.output_hidden_states = False

class PointerConfig(object):
	"""docstring for PointerConfig"""
	def __init__(self,):
		super(PointerConfig, self).__init__()
		self.rnn_hidden_size = 768
		self.rnn_dropout_value = 0
		self.rnn_n_layers = 1

		self.pointer_dim = 768
		self.dropout_value = 0.3

		self.informed_attention_config = TransformerConfig()


class SentenceEncoderConfig(object):
	"""docstring for SentenceEncoderConfig"""
	def __init__(self,):
		super(SentenceEncoderConfig, self).__init__()
		self.multihead_conf = TransformerConfig()

class ParagEncoderConfig(object):
		"""docstring for ParagEncoderConfig"""
		def __init__(self,):
			super(ParagEncoderConfig, self).__init__()
			self.multihead_conf = TransformerConfig()
			self.transformer = TransformerConfig()
				

class FromBert(object):
	"""docstring for FromBert"""
	def __init__(self, ):
		super(FromBert, self).__init__()
		self.word_encoder_type = 'bert' # {'bert', 'distill'}
		self.sentence_encoder_type = 'multihead' # {'singlehead', 'multihead', 'rnn'}
		self.parag_encoder_type = 'transformer' # {'transformer', 'avg'}

		self.dropout_value = 0.3

		self.sentence_encoder_conf = SentenceEncoderConfig()
		self.parag_encoder_conf = ParagEncoderConfig()
		self.informed_pointer_conf = PointerConfig()		

