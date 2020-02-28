import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class AverageMeter(object):
	"""Computes the average and stores the values"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.values = []
		self.sum = 0
		self.count = 0

	def update(self, val):
		self.values.append(val)
		self.count += 1
		self.sum += val

	def extend(self, val):
		self.values.extend(val)
		self.count += len(val)
		self.sum += sum(val)

	def average(self):
		return self.sum / self.count

	def last(self):
		if len(self.values) > 0:
			return self.values[-1]
		else:
			return None
	def save(self, path):
		with open(path,'w') as f:
			for i in range(len(self.values)):
				f.write(str(i+1).zfill(3)+' '+str(self.values[i])+'\n')
	def draw_fig(self, path, refresh=True, label=''):
		if refresh:
			plt.clf()
		plt.plot(self.values,label=label)
		plt.legend()
		plt.savefig(path)
		
