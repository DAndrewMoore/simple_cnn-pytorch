import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from model import Classifier
from data_loader import testDL
from utils import write_output

import pdb

########################
# Params
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_dir = 'D:\\data'
epoch_dir='epochs'
results_fname = 'test_results.csv'
num_class=28
num_chan=1
epoch=22
batch_size=16
threshold=0.33
########################

def main():
    # Load pre-trained model
    model = Classifier(num_chan=num_chan).to(device)
    model.load_state_dict(torch.load(os.path.join(epoch_dir, str(epoch)+'.model')))
    # Create dataset loader over all training set
    dataset = testDL(data_dir, 'test')
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    out_arr = []
    fname_arr = []
    data_iter = iter(dataloader)
    for idx, dat in enumerate(data_iter):
        pics = dat[0].to(device).float()
        fnames = dat[1]
        out = torch.sigmoid(model(pics)).detach().cpu().numpy()
        out[out >= threshold] = 1
        out[out <  threshold] = 0
        fname_arr += [*fnames]
        out_arr += [*out]
    write_output(fname_arr, out_arr, results_fname)

if __name__ == '__main__':
    main()
