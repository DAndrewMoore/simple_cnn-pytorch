import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DL(Dataset):
	def __init__(self, input_dir, trORte):
		self.convToTensor = ToTensor()
		csv_doc = os.path.join(input_dir, trORte + '.csv')
		self.pic_dir = os.path.join(input_dir, trORte)
		self.colors = ['_red.png', '_blue.png', '_yellow.png']
		target = open(csv_doc, 'r')
		self.train_set = []
		target.readline()
		for line in target:
			self.train_set.append(line.strip().split(','))
		target.close()

	def __len__(self):
		return len(self.train_set)

	def __getitem__(self, idx):
		# Filename[0]; Classes[1].split(' ')
		row = self.train_set[idx]
		filename = row[0]
		im = Image.open(os.path.join(self.pic_dir, filename + '_green.png'))
		im = self.convToTensor(im)
		# Set class values
		cls = np.zeros(28)
		for i in list(map(int, row[1].split(' '))):
			cls[i] = 1
		cls = torch.from_numpy(cls).float()
		return im, cls
# 04c49f8c-bba1-11e8-b2b9-ac1f6b6435d0_yellow.png
