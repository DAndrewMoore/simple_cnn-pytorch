import torchvision
import torch.nn as nn

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		# Convolutional layer maker
		def makeConvLayers(in_ch, out_ch):
			convLayers = []
			convLayers.append(nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=0, bias=False))
			convLayers.append(nn.ReLU(inplace=True))
			convLayers.append(nn.MaxPool2d(2, stride=2))
			return convLayers
		# Fully Connected Layer maker
		def makeFCNlayers(in_ch, out_ch):
			fcnLayers = []
			fcnLayers.append(nn.Linear(in_ch, out_ch, bias=False))
			fcnLayers.append(nn.ReLU(inplace=True))
			return fcnLayers
		# Make feature extraction layers
		features = []
		features += [*makeConvLayers(3, 64)]
		features += [*makeConvLayers(64, 128)]
		features += [*makeConvLayers(128, 128)]
		features += [*makeConvLayers(128, 256)]
		features += [*makeConvLayers(256, 256)]
		features += [*makeConvLayers(256, 512)]
		self.features = nn.Sequential(*features)
		# Make fully connected layers
		fcnLayers = []
		fcnLayers += [*makeFCNlayers(8192, 2048)]
		fcnLayers += [*makeFCNlayers(2048, 512)]
		fcnLayers.append(nn.Linear(512, 28, bias=False))
		self.fc = nn.Sequential(*fcnLayers)

	def forward(self, x):
		feats = self.features(x)
		preds = self.fc(feats.view(x.size(0), -1))
		return preds
