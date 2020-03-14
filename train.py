from transformers import AdamW

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import data

import models

import time
import numpy as np


from torch.optim.lr_scheduler import ReduceLROnPlateau

import log
import config
from utils import *



torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

trainset = data.SIND('../AbstractData/SIND/sis_cleaned/train.json')
valset = data.SIND('../AbstractData/SIND/sis_cleaned/val.json')

# trainset = data.Synthetic('../AbstractData/synthetic/train_l1.txt')
# valset = data.Synthetic('../AbstractData/synthetic/val_l1.txt')


# trainset = data.ROC('../AbstractData/ROCStory/ROC_train.csv')
# valset = data.ROC('../AbstractData/ROCStory/ROC_val.csv')

# trainset = data.NIPS('../AbstractData/NIPS/fromnseg/train.lower')
# valset = data.NIPS('../AbstractData/NIPS/fromnseg/val.lower')


trainloader = DataLoader(trainset,batch_size=16,collate_fn=data.bert_batchify)
valloader = DataLoader(valset,batch_size=8,collate_fn=data.bert_batchify,shuffle=False)


conf = config.FromBertConfig()
model = models.FromBert(conf)


model = model.cuda()

params = list(model.pointer.parameters()) + list(model.sentence_encoder.parameters()) + \
		 list(model.parag_encoder.parameters()) + list(model.dense.parameters())
# params = model.parameters()


# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()


# optim = torch.optim.Adam(params, lr=0.0001,betas=(0.9, 0.99))
# optim = AdamW(params, lr=0.000001, correct_bias=True) 
# optim = torch.optim.SGD(params, lr=0.0001, momentum=0.9,weight_decay=0.0001,nesterov=True)
# optim = torch.optim.AdamW(params,lr=0.0001)

path = 'log/1'

train_loss = log.AverageMeter()
val_prm = log.AverageMeter()
val_kendal = log.AverageMeter()
val_loss = log.AverageMeter()

optim = AdamW(params, lr=0.0001, correct_bias=True) 

for i in range(1,4):
	start = time.time()
	lt = train(model, trainloader, optim, criterion, clip=1.0)
	p, k, lv = eval(model, valloader, criterion)
	end = time.time()
	
	train_loss.update(lt[1])
	val_prm.update(p)
	val_kendal.update(k)
	val_loss.update(lv)
	print('{:d},\tPMR: {:.4f},\tTau: {:.4f},\ttrain loss: {:.4f},\tval loss: {:.4f},\ttime: {:.1f}'.format(i,p,k,lt[1],lv,end-start))

	# scheduler.step(k)
	val_prm.save(path+'-val_prm.txt')
	train_loss.save(path+'-train_loss.txt')
	val_kendal.save(path+'-val_kendal.txt')


	save_model(model, path, i)


	val_kendal.draw_fig(path+'-val_kendal.png',label='Kendal')
	val_prm.draw_fig(path+'-val_prm.png',label='PMR')
	train_loss.draw_fig(path+'-train_loss.png',label='Train Loss')
	val_loss.draw_fig(path+'-val_loss.png', refresh = False,label='Val Loss')



new_params = [

	{'params':params,'correct_bias':True},
	{'params':model.word_encoder.parameters(),'correct_bias':True, 'lr':0.00001}

]

optim = AdamW(new_params, lr=0.0001 ) 
scheduler = ReduceLROnPlateau(optim, 'min', factor=0.5, patience=0, verbose=True)
# lambda1 = lambda epoch: 0.5**epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda1)

for i in range(4,31):
	start = time.time()
	lt = train(model, trainloader, optim, criterion, clip=1.0)
	p, k, lv = eval(model, valloader, criterion)
	end = time.time()
	
	train_loss.update(lt[1])
	val_prm.update(p)
	val_kendal.update(k)
	val_loss.update(lv)
	print('{:d},\tPMR: {:.4f},\tTau: {:.4f},\ttrain loss: {:.4f},\tval loss: {:.4f},\ttime: {:.1f}'.format(i,p,k,lt[1],lv,end-start))

	scheduler.step(lv)
	val_prm.save(path+'-val_prm.txt')
	train_loss.save(path+'-train_loss.txt')
	val_kendal.save(path+'-val_kendal.txt')

	save_model(model, path, i)

	val_kendal.draw_fig(path+'-val_kendal.png',label='Kendal')
	val_prm.draw_fig(path+'-val_prm.png',label='PMR')
	train_loss.draw_fig(path+'-train_loss.png',label='Train Loss')
	val_loss.draw_fig(path+'-val_loss.png', refresh = False,label='Val Loss')