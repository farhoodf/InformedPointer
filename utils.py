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

def accuracy(y, y_hat, lengths):
	batch_size = y.size()[0]
	corrects = 0.0
	falses = 0.0 
	for i in range(batch_size):
		condition = True
		for j in range(lengths[i]):
			if (y_hat[i,j] == y[i,j]):
				corrects += 1
			else:
				falses += 1

	return corrects/(corrects+falses)

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
	r_y_hat = torch.zeros((lengths.sum(),y_hat.size()[1]),dtype=torch.float)
	r_y = torch.zeros((lengths.sum()),dtype=torch.long)
	p = 0
	for i in range(len(lengths)):
		r_y[p:lengths[i]+p] = y[i,:lengths[i]]
		r_y_hat[p:lengths[i]+p,:] = y_hat[i,:lengths[i],:]
		p += lengths[i]
	return r_y, r_y_hat

def print_res(i,p,k,acc,lt,lv,t):
	print('{:d},\tPMR: {:.4f},\tTau: {:.4f},\tAcc: {:.4f},\ttrain loss: {:.4f},\tval loss: {:.4f},\ttime: {:.1f}'.format(i,p,k,acc,lt[1],lv,t),flush=True)

def test(model, testloader):
	model.eval()
	tot_pmr = []
	tot_kendal = []
	accs = []
	with torch.no_grad():
		for i,batch in enumerate(testloader):
			batch = batch_generator(batch)
			out = model(batch['data'],batch['s_lengths'],batch['p_lengths'])#,batch['labels'])
			arged = out.argmax(-1)
			perf = pmr(batch['labels'],arged,batch['p_lengths'])
			tot_pmr.append(perf)
			acc = accuracy(batch['labels'],arged,batch['p_lengths'])
			accs.append(acc)
			for j in range(batch['s_lengths'].shape[0]):
				pl = batch['p_lengths'][j]
				tot_kendal.append(new_tau(batch['labels'][j][:pl].detach().cpu().numpy(),arged[j][:pl].detach().cpu().numpy()))
			# tot_kendal.extend(list(kendal(,batch['labels']).detach().cpu().numpy()))
	return sum(tot_pmr)/len(tot_pmr), sum(tot_kendal)/len(tot_kendal), sum(accs)/len(accs)

def train(model, trainloader, optim, criterion, clip=None):
	model.train()
	tot_loss = []
	for i,batch in enumerate(trainloader):
		batch = batch_generator(batch)
		l = batch['s_lengths'].shape[1]
		optim.zero_grad()
		out = model(batch['data'],batch['s_lengths'],batch['p_lengths'],batch['labels'])
		reshaped_y,reshaped_y_hat = reshape_labels(batch['labels'],out,batch['p_lengths'])
		# reshaped_y = batch['labels'].view(-1)
		# reshaped_y_hat = out.reshape(-1,l)
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
	tot_acc = []
	with torch.no_grad():
		for i,batch in enumerate(valloader):
			batch = batch_generator(batch)
			l = batch['s_lengths'].shape[1]
			out = model(batch['data'],batch['s_lengths'],batch['p_lengths'])#,batch['labels'])
			reshaped_y,reshaped_y_hat = reshape_labels(batch['labels'],out,batch['p_lengths'])
			# reshaped_y = batch['labels'].view(-1)
			# reshaped_y_hat = out.reshape(-1,l)
			loss = criterion(reshaped_y_hat,reshaped_y)
			arged = out.argmax(-1)
			perf = pmr(batch['labels'],arged,batch['p_lengths'])
			acc = accuracy(batch['labels'],arged,batch['p_lengths'])
			tot_pmr.append(perf)
			tot_loss.append(loss)
			tot_acc.append(acc)
			for j in range(batch['s_lengths'].shape[0]):
				pl = batch['p_lengths'][j]
				tot_kendal.append(new_tau(batch['labels'][j][:pl].detach().cpu().numpy(),arged[j][:pl].detach().cpu().numpy()))
			# tot_kendal.extend(list(kendal(,batch['labels']).detach().cpu().numpy()))
	return sum(tot_pmr)/len(tot_pmr), sum(tot_kendal)/len(tot_kendal), sum(tot_loss)/len(tot_loss), sum(tot_acc)/len(tot_acc)