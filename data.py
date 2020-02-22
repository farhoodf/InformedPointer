import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from config import conf
import nltk
import pandas as pd
from multiprocessing import Pool
import re

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class SIND(Dataset):
	"""docstring for SIND"""
	def __init__(self, path):
		super(SIND, self).__init__()
		self.data = []
		self.path = path
		self.__load_data__()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
	
	def __load_data__(self):
		with open(self.path,'r') as f:
			raw = json.load(f)
		for key in raw:
			raw_sample = raw[key]
			# sample = [[conf['first_sent']]]+[raw[key][str(i)]['text'] for i in range(len(raw_sample))]+[[conf['last_sent']]]
			sample = [[conf['first_sent']]]+[raw[key][str(i)]['original_text'] for i in range(len(raw_sample))]#+[[conf['last_sent']]]



			# sent_len = []
			# for sent in sample:
			# 	sent_len.append(len(sent))
			labels = list(range(len(sample)))
			self.data.append({'text':sample,'labels':labels,'p_len':len(labels)})


class Synthetic(Dataset):
	"""docstring for Synthetic"""
	def __init__(self, path):
		super(Synthetic, self).__init__()
		self.data = []
		self.path = path
		self.__load_data__()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
	
	def __load_data__(self):
		with open(self.path,'r') as f:
			raw = f.read()
		raw = raw.split('\n\n')
		for sample in raw:
			sample = sample.strip().split('\n')
			if len(sample) < 2:
				continue
			sample = [[conf['first_sent']]] + sample
			# raw_sample = raw[key]
			# sample = [[conf['first_sent']]]+[raw[key][str(i)]['text'] for i in range(len(raw_sample))]+[[conf['last_sent']]]
			# sample = [[conf['first_sent']]]+[raw[key][str(i)]['original_text'] for i in range(len(raw_sample))]#+[[conf['last_sent']]]



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
		padded_labels[i,:p_lens[i]] = torch.argsort(shuffled_indices)
		text[i] = [text[i][k] for k in shuffled_indices]

		for j in range(p_lens[i]):
			padded_data[i,j,:s_lengths[i][j]] = torch.tensor(data[i][j])
			padded_lengths[i,j] = s_lengths[i][j]


	return {'data':padded_data,'s_lengths':padded_lengths, 'labels': padded_labels,
			'p_lengths':torch.tensor(p_lens), 'text':text}