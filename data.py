import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from config import token_config
import nltk
import pandas as pd
from multiprocessing import Pool
import re

from transformers import BertTokenizer, DistilBertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
from allennlp.modules.elmo import batch_to_ids


class AbstractData(Dataset):
	"""docstring for AbstractData"""
	def __init__(self, path, tokenizer=None, w_to_id=None):
		super(AbstractData, self).__init__()
		self.data = []
		self.path = path
		self.__load_data__()

		if tokenizer is not None:
			self.tokenize(tokenizer)
		if w_to_id is not None:
			self.vectorize_dic(w_to_id)
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]

	def tokenize(self, tokenizer):
		for i in range(len(self.data)):
			self.data[i]['s_len'] = []
			for j in range(self.data[i]['p_len']):

				self.data[i]['text'][j] = tokenizer(self.data[i]['text'][j])
				self.data[i]['s_len'].append(len(self.data[i]['text'][j]))
	
	def vectorize_dic(self, w_to_id):
		for i in range(len(self.data)):
			self.data[i]['vectorized'] = []
			for j in range(self.data[i]['p_len']):
				self.data[i]['vectorized'].append([])
				for k in range(self.data[i]['s_len'][j]):
					self.data[i]['vectorized'][j].append(w_to_id.get(self.data[i]['text'][j][k],w_to_id[token_config['unknown']]))
				self.data[i]['vectorized'][j] = torch.tensor(self.data[i]['vectorized'][j])
class SIND(AbstractData):
	"""docstring for SIND"""
	def __init__(self, path, tokenizer=None, w_to_id=None):
		super(SIND, self).__init__(path, tokenizer, w_to_id)

	
	def __load_data__(self):
		with open(self.path,'r') as f:
			raw = json.load(f)
		for key in raw:
			raw_sample = raw[key]
			# sample = [token_config['first_sent']]+[raw[key][str(i)]['original_text'] for i in range(len(raw_sample))]#+[[token_config['last_sent']]]
			sample = [raw[key][str(i)]['original_text'] for i in range(len(raw_sample))]#+[[token_config['last_sent']]]

			labels = list(range(len(sample)))
			# labels = list(range(len(sample)-1,-1,-1))
			self.data.append({'text':sample,'labels':labels,'p_len':len(labels)})

			# self.data = self.data[:1000]

class ROC(AbstractData):
	"""docstring for ROC"""
	def __init__(self, path, tokenizer=None, w_to_id=None):
		super(ROC, self).__init__(path, tokenizer, w_to_id)
	
	def __load_data__(self):

		dataFrame = pd.read_csv(self.path,error_bad_lines=False)
		raw = dataFrame.values
		for i in range(raw.shape[0]):
			sample = raw[i,2:]
			# sample = [[conf['first_sent']]] + sample + [[conf['last_sent']]]
			labels = list(range(len(sample)))
			# labels = list(range(len(sample)-1,-1,-1))
			self.data.append({'text':sample,'labels':labels,'p_len':len(labels)})

			# self.data = self.data[:1000]


class Synthetic(AbstractData):
	"""docstring for Synthetic"""
	def __init__(self, path):
		super(Synthetic, self).__init__()
		self.data = []
		self.path = path
		self.__load_data__()

	
	def __load_data__(self):
		with open(self.path,'r') as f:
			raw = f.read()
		raw = raw.split('\n\n')
		for sample in raw:
			sample = sample.strip().split('\n')
			if len(sample) < 2:
				continue
			sample = [[token_config['first_sent']]] + sample
			# raw_sample = raw[key]
			# sample = [[token_config['first_sent']]]+[raw[key][str(i)]['text'] for i in range(len(raw_sample))]+[[token_config['last_sent']]]
			# sample = [[token_config['first_sent']]]+[raw[key][str(i)]['original_text'] for i in range(len(raw_sample))]#+[[token_config['last_sent']]]



			# sent_len = []
			# for sent in sample:
			# 	sent_len.append(len(sent))
			labels = list(range(len(sample)))
			self.data.append({'text':sample,'labels':labels,'p_len':len(labels)})
		# if len(self.data) > 500:
		# 	self.data = self.data[:500]



def bert_batchify(batch):

	batch_size = len(batch)
	data = []
	labels = []
	s_lengths = []
	p_lens = []
	max_sent_len = []
	text = []

	for sample in batch:
		sent_ids = []
		bert_lens = []
		for sent in sample['text']:
			sent_ids.append(tokenizer.encode(sent))
			bert_lens.append(len(sent_ids[-1]))
		text.append(sample['text'])
		s_lengths.append(bert_lens)
		data.append(sent_ids)
		labels.append(sample['labels'])
		
		p_lens.append(sample['p_len'])
		max_sent_len.append(max(bert_lens))
	p_len = max(p_lens)
	s_len = max(max_sent_len)

	padded_data = torch.zeros((batch_size,p_len,s_len),dtype=torch.long)
	padded_lengths = torch.ones((batch_size,p_len),dtype=torch.long)
	padded_labels = torch.zeros((batch_size,p_len),dtype=torch.long)
	for i in range(batch_size):
		shuffled_indices = torch.randperm(p_lens[i])
		data[i] = [data[i][k] for k in shuffled_indices]
		s_lengths[i] = [s_lengths[i][k] for k in shuffled_indices]
		padded_labels[i,:p_lens[i]] = torch.argsort(shuffled_indices)#,descending=True)
		text[i] = [text[i][k] for k in shuffled_indices]

		for j in range(p_lens[i]):
			padded_data[i,j,:s_lengths[i][j]] = torch.tensor(data[i][j])
			padded_lengths[i,j] = s_lengths[i][j]


	return {'data':padded_data,'s_lengths':padded_lengths, 'labels': padded_labels,
			'p_lengths':torch.tensor(p_lens), 'text':text}

def elmo_batchify(batch):

	batch_size = len(batch)
	data = []
	labels = []
	s_lengths = []
	p_lens = []
	max_sent_len = []
	text = []
	vectorized = []

	for sample in batch:
		# d = []
		# for sent in sample['text']:
		# 	d.append(batch_to_ids(sent))
		d = batch_to_ids(sample['text'])
		text.append(sample['text'])
		s_lengths.append(sample['s_len'])
		data.append(d)
		labels.append(sample['labels'])
		vectorized.append(sample['vectorized'])
		
		p_lens.append(d.shape[0])
		max_sent_len.append(d.shape[1])
	p_len = max(p_lens)
	s_len = max(max_sent_len)

	padded_data = torch.zeros((batch_size,p_len,s_len,50),dtype=torch.long)
	padded_lengths = torch.ones((batch_size,p_len),dtype=torch.long)
	padded_labels = torch.zeros((batch_size,p_len),dtype=torch.long)
	padded_vectorized = torch.zeros((batch_size,p_len,s_len),dtype=torch.long)
	for i in range(batch_size):
		shuffled_indices = torch.randperm(p_lens[i])
		# data[i] = [data[i][k] for k in shuffled_indices]
		s_lengths[i] = [s_lengths[i][k] for k in shuffled_indices]
		padded_labels[i,:p_lens[i]] = torch.argsort(shuffled_indices)
		text[i] = [text[i][k] for k in shuffled_indices]

		for j in range(p_lens[i]):
			padded_data[i,j,:max_sent_len[i]] = data[i][shuffled_indices[j]]
			padded_vectorized[i,j,:s_lengths[i][j]] = vectorized[i][shuffled_indices[j]]
			padded_lengths[i,j] = s_lengths[i][j]


	return {'data':padded_data,'s_lengths':padded_lengths, 'labels': padded_labels,
			'p_lengths':torch.tensor(p_lens), 'text':text,'vectorized':padded_vectorized}