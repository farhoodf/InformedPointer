import torch
import numpy as np
import itertools



def new_tau(t,p):
    s_t = set([i for i in itertools.combinations(t, 2)])
    s_p = set([i for i in itertools.combinations(p, 2)])
    cn_2 = len(p) * (len(p) - 1) / 2
    pairs = len(s_p) - len(s_p.intersection(s_t))
    tau = 1 - 2 * pairs / cn_2
    return tau

def pmr(y, y_hat, lengths):
	batch_size = y.size()[0]
	corrects = 0.0
	for i in range(batch_size):
		condition = True
		for j in range(lengths[i]):
			if not (y_hat[i,j] == y[i,j]):
				condition = False
				break
		if condition:
			corrects += 1
	return corrects/batch_size

def save_model(model, path, epoch):
		mpath = path +'-'+str(epoch) +'model.pt'
		torch.save({
			'model_state_dict': model.state_dict(),
			}, mpath)


def load_model(model, path):

	model.load_state_dict(torch.load(path)['model_state_dict'])
	return model


def batch_generator(batch,device=torch.device('cuda')):
	n_batch = {}
	for k in batch:
		if k != 'text':
			n_batch[k] = batch[k].to(device=device)
	return n_batch

def reshape_labels(y,y_hat,lengths):
	r_y_hat = torch.zeros((lengths.sum(),y_hat.size()[0]),dtype=torch.float)
	r_y = torch.zeros((lengths.sum()),dtype=torch.long)
	p = 0
	for i in range(len(lengths)):
		r_y[p:lengths[i]+p] = y[:lengths[i],i]
		r_y_hat[p:lengths[i]+p,:] = y_hat[:lengths[i],i,:]
		p += lengths[i]
	return r_y, r_y_hat


def test(model, testloader):
	model.eval()
	tot_pmr = []
	tot_kendal = []
	with torch.no_grad():
		for i,batch in enumerate(testloader):
			batch = batch_generator(batch)
			out = model(batch['data'],batch['s_lengths'],batch['p_lengths'])#,batch['labels'])
			arged = out.argmax(-1)
			perf = pmr(batch['labels'],arged,batch['p_lengths'])
			tot_pmr.append(perf)
			for j in range(batch['s_lengths'].shape[0]):
				tot_kendal.append(new_tau(batch['labels'][j].detach().cpu().numpy(),arged[j].detach().cpu().numpy()))
			# tot_kendal.extend(list(kendal(,batch['labels']).detach().cpu().numpy()))
	return sum(tot_pmr)/len(tot_pmr), sum(tot_kendal)/len(tot_kendal)

def train(model, trainloader, optim, criterion, clip=None):
	model.train()
	tot_loss = []
	for i,batch in enumerate(trainloader):
		batch = batch_generator(batch)
		l = batch['s_lengths'].shape[1]
		optim.zero_grad()
		out = model(batch['data'],batch['s_lengths'],batch['p_lengths'],batch['labels'])
#         reshaped_y,reshaped_y_hat = reshape_labels(batch['labels'],out,batch['p_lengths'])
		reshaped_y = batch['labels'].view(-1)
		reshaped_y_hat = out.reshape(-1,l)
		loss = criterion(reshaped_y_hat,reshaped_y)
		loss.backward()
		if clip is not None:
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
		optim.step()
		tot_loss.append(loss.item())
	return i,sum(tot_loss)/len(tot_loss)

def eval(model, valloader, criterion):
	model.eval()
	tot_pmr = []
	tot_kendal = []
	tot_loss = []
	with torch.no_grad():
		for i,batch in enumerate(valloader):
			batch = batch_generator(batch)
			l = batch['s_lengths'].shape[1]
			out = model(batch['data'],batch['s_lengths'],batch['p_lengths'])#,batch['labels'])
			reshaped_y = batch['labels'].view(-1)
			reshaped_y_hat = out.reshape(-1,l)
			loss = criterion(reshaped_y_hat,reshaped_y)
			arged = out.argmax(-1)
			perf = pmr(batch['labels'],arged,batch['p_lengths'])
			tot_pmr.append(perf)
			tot_loss.append(loss)
			for j in range(batch['s_lengths'].shape[0]):
				tot_kendal.append(new_tau(batch['labels'][j].detach().cpu().numpy(),arged[j].detach().cpu().numpy()))
			# tot_kendal.extend(list(kendal(,batch['labels']).detach().cpu().numpy()))
	return sum(tot_pmr)/len(tot_pmr), sum(tot_kendal)/len(tot_kendal), sum(tot_loss)/len(tot_loss)