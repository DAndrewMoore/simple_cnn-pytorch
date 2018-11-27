import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils import conv2dense

class DL(Dataset):
	def __init__(self, input_dir, trORte):
		self.convToTensor = ToTensor()
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
		im = Image.open(os.path.join(self.pic_dir, filename + '_green.png'))
		im = self.convToTensor(im)
		# Set class values
		cls = conv2dense(row[1])
		cls = torch.from_numpy(cls).float()
		return im, cls

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
