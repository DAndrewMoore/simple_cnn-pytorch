import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils import conv2dense

class DL(Dataset):
	# TODO implement channel rotation and flipping
	def __init__(self, input_dir, trORte):
		self.convToTensor = ToTensor()
		self.colors = ['_green.png', '_blue.png', '_red.png', '_yellow.png']
		csv_doc = os.path.join(input_dir, trORte + '.csv')
		self.pic_dir = os.path.join(input_dir, trORte)
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
		img_tnsr = np.zeros([4, 512, 512])
		for idx, c in enumerate(self.colors):
			tmp_im = Image.open(os.path.join(self.pic_dir, filename+c))
			img_tnsr[idx] = np.asarray(tmp_im)
		img_tnsr = torch.from_numpy(img_tnsr).float()
		# Set class values
		cls = conv2dense(row[1])
		cls = torch.from_numpy(cls).float()
		return img_tnsr, cls

class testDL(Dataset):
	def __init__(self, input_dir, trORte):
		self.convToTensor = ToTensor()
		self.pic_dir = os.path.join(input_dir, trORte)
		self.pic_list = [f for f in os.listdir(self.pic_dir) if '_green' in f]

	def __len__(self):
		return len(self.pic_list)

	def __getitem__(self, idx):
		fpath = os.path.join(self.pic_dir, self.pic_list[idx])
		im = Image.open(fpath)
		im = self.convToTensor(im)
		return im, self.pic_list[idx].replace('_green.png', '')
