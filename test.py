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

from arg import args, save_args


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(0)
np.random.seed(0)



# trainset = data.SIND('../AbstractData/SIND/sis_cleaned/train.json')
# valset = data.SIND('../AbstractData/SIND/sis_cleaned/val.json')

# trainset = data.Synthetic('../AbstractData/synthetic/train_l1.txt')
# valset = data.Synthetic('../AbstractData/synthetic/val_l1.txt')


# trainset = data.ROC('../AbstractData/ROCStory/ROC_train.csv')
# valset = data.ROC('../AbstractData/ROCStory/ROC_val.csv')

# trainset = data.NIPS('../AbstractData/NIPS/fromnseg/train.lower')
# valset = data.NIPS('../AbstractData/NIPS/fromnseg/val.lower')

# testset = data.SIND('../AbstractData/SIND/sis_cleaned/test.json')
# testset = data.ROC('../AbstractData/ROCStory/ROC_test.csv')
# testset = data.NIPS('../AbstractData/NIPS/fromnseg/test.lower')

if args.Dataset == 'sind':
	testset = data.SIND('../AbstractData/SIND/sis_cleaned/test.json')
elif args.Dataset == 'roc':
	testset = data.ROC('../AbstractData/ROCStory/ROC_test.csv')
elif args.Dataset == 'nips':
	testset = data.NIPS('../AbstractData/NIPS/fromnseg/test.lower')
elif args.Dataset == 'arxiv':
	testset = data.ArXiv('../AbstractData/ArXiv/test.txt')
elif args.Dataset == 'nfs':
	testset = data.NFS('../AbstractData/NFS/dataset/test/')

path = args.Path

testloader = DataLoader(testset,batch_size=1,collate_fn=data.batchify,shuffle=False)


# conf = config.FromBertConfig()
conf = config.FromBertConfig(
				gen_dim = args.gendim,
				rnn_dim = args.rnndim,
				multi_drop= args.multi_drop,
				pointer_drop = args.pointer_drop,
				rnn_dropout = args.rnn_dropout,
				word_encoder_drop = args.word_encoder_drop,
				n_layer_sent=args.n_layer_sent,
				n_heads = args.n_heads,
				informed_type=args.informed_type)
model = models.FromBert(conf)
model = load_model(model,path)

model = model.cuda()



start = time.time()
i=0
lt = 0
p, k, acc = test(model, testloader)
end = time.time()

print('pmr:', p,flush=True)
print('kendal:', k,flush=True)
print('accuracy:', acc,flush=True)
print('time:',end-start,flush=True)
print()
# print_res(i,p,k,acc,lt,lv,end-start)
