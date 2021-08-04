import torchvision
import torch.nn as nn

import pdb

# Convolutional layer maker
def makeConvLayers(in_ch, out_ch, repeat=False, pool=True, batchnorm=False):
	convLayers = []
	convLayers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
	if batchnorm:
		convLayers.append(nn.BatchNorm2d(out_ch))
	convLayers.append(nn.LeakyReLU(inplace=True))
	if repeat:
		convLayers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
		if batchnorm:
			convLayers.append(nn.BatchNorm2d(out_ch))
		convLayers.append(nn.LeakyReLU(inplace=True))
	if pool:
		convLayers.append(nn.AvgPool2d(2, stride=2))
	return convLayers

# Fully Connected Layer maker
def makeFCNlayers(in_ch, out_ch):
	fcnLayers = []
	fcnLayers.append(nn.Linear(in_ch, out_ch, bias=False))
	fcnLayers.append(nn.LeakyReLU(inplace=True))
	return fcnLayers

class Classifier(nn.Module):
	def __init__(self, num_chan=3, num_classes=28):
		super(Classifier, self).__init__()
		# Make feature extraction layers
		features = []
		features += [*makeConvLayers(num_chan, 4)]
		features += [*makeConvLayers(4, 8)]
		features += [*makeConvLayers(8, 16)]
		features += [*makeConvLayers(16, 32)]
		features += [*makeConvLayers(32, 64, repeat=True)]
		features += [*makeConvLayers(64, 128, repeat=True)]
		features += [*makeConvLayers(128, 128, pool=False)]
		self.features = nn.Sequential(*features)
		# Make fully connected layers
		fcnLayers = []
		fcnLayers += [*makeFCNlayers(8192, 4096)]
		fcnLayers += [*makeFCNlayers(4096, 1024)]
		fcnLayers += [*makeFCNlayers(1024, 256)]
		fcnLayers.append(nn.Linear(256, num_classes, bias=False))
		self.fc = nn.Sequential(*fcnLayers)

	def forward(self, x):
		feats = self.features(x)
		feats = feats.view(x.size(0), -1)
		return self.fc(feats)

class FCN_Classifier(nn.Module):
	def __init__(self, num_chan=3, num_classes=28):
		super(FCN_Classifier, self).__init__()
		features = []
		features += [*makeConvLayers(num_chan, 4)]
		features += [*makeConvLayers(4, 8)]
		features += [*makeConvLayers(8, 32)]
		features += [*makeConvLayers(32, 64)]
		features += [*makeConvLayers(64, 128)]
		features += [*makeConvLayers(128, 256, batchnorm=True)]
		features += [*makeConvLayers(256, 512)]
		features += [*makeConvLayers(512, 128, pool=True, batchnorm=True)]
		features += [*makeConvLayers(128, 32)]
		features += [*makeConvLayers(32, num_classes, repeat=True)]
		self.features = nn.Sequential(*features)
	
	def forward(self, x):
		return self.features(x)

