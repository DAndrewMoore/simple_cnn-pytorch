import torchvision
import torch.nn as nn

import pdb

class Classifier(nn.Module):
	"""VGG slightly with altered kernel_size, stride, and padding to accomodate
	a different sized input image. Original image was 254x254, current images
	are 512x512.

	ConvNet details provided by https://arxiv.org/pdf/1409.1556.pdf """

	def __init__(self):
		super(Classifier, self).__init__()
		# Convolutional layers
		def makeConvLayers(input, output):
			layers = []
			layers.append(nn.Conv2d(input, output,
				kernel_size=3, stride=1, padding=1, bias=False))
			layers.append(nn.ReLU(inplace=True))
			return layers
		# Feature construction
		featLayers = []
		featLayers += [*makeConvLayers(3, 64)]
		featLayers += [*makeConvLayers(64, 64)]
		featLayers.append(nn.MaxPool2d(kernel_size=(2,2), stride=2))
		featLayers += [*makeConvLayers(64, 128)]
		featLayers += [*makeConvLayers(128, 128)]
		featLayers.append(nn.MaxPool2d(kernel_size=(2,2), stride=2))
		featLayers += [*makeConvLayers(128, 256)]
		featLayers += [*makeConvLayers(256, 256)]
		featLayers += [*makeConvLayers(256, 256)]
		featLayers.append(nn.MaxPool2d(kernel_size=(2,2), stride=2))
		featLayers += [*makeConvLayers(256, 512)]
		featLayers += [*makeConvLayers(512, 512)]
		featLayers += [*makeConvLayers(512, 512)]
		featLayers.append(nn.MaxPool2d(kernel_size=(2,2), stride=2))
		featLayers += [*makeConvLayers(512, 512)]
		featLayers += [*makeConvLayers(512, 512)]
		featLayers += [*makeConvLayers(512, 512)]
		featLayers.append(nn.MaxPool2d(kernel_size=(2,2), stride=2))
		self.features = nn.Sequential(*featLayers)
		# Classifier Construction
		classLayers = []
		# Input linear layer is (image size) / 2^(# of MaxPool2d layers)
		classLayers.append(nn.Linear(512//2**5, 4096))
		classLayers.append(nn.Linear(4096, 4096))
		classLayers.append(nn.Linear(4096, 28))
		# The final softmax layer can't be used since
		# there can be more than a single class per instance
		# classLayers.append(nn.Softmax(dim=1))
		self.fc = nn.Sequential(*classLayers)


	def forward(self, x):
		feats = self.features(x)
		preds = self.fc(feats.view(x.size(0), -1))
		return preds
