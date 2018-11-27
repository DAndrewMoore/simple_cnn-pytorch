import os
import time
import time

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from data_loader import DL
from model import Classifier
# from vgg import Classifier
# from test_model import Classifier

import pdb

###################
# Params
num_class = 28
data_dir = 'data'
save_dir = 'epochs'
finl_sve = os.path.join(save_dir, 'comp.model')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_size = 512
num_chan = 1
l_r = 1e-4
batch_size = 16
num_epochs = 500
num_cpu = 4
########################

def save_epoch(model, save_dir, epoch_num, cur_loss):
    torch.save(model.state_dict(), os.path.join(save_dir, str(epoch_num)+'.model'))
    target = open(os.path.join(save_dir, 'loss_list.tsv'), 'a')
    strTime = '_'.join([time.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')])
    target.write('%s\t%d\t%.3f\n' % (strTime, epoch_num, cur_loss))
    target.close()

def main():
    # Set up data loader
    dataset = DL(data_dir, 'train')
    train_size = round(0.7*len(dataset))
    train, validate = torch.utils.data.random_split(dataset, (train_size, len(dataset) - train_size))
    tr_loader = data.DataLoader(dataset=train,
                                batch_size=batch_size,
                                shuffle=True, num_workers=num_cpu)
    te_loader = data.DataLoader(dataset=validate,
                                batch_size=batch_size,
                                shuffle=False,num_workers=num_cpu)
    tr_btch_interval = len(tr_loader) // 4
    te_btch_interval = len(te_loader) // 4
    # Set up the classifier
    mdl = Classifier(num_chan=num_chan).to(device)
    # Set up loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    # Set up the optimizer
    optim = torch.optim.Adam(mdl.parameters(), lr=l_r)
    for epoch in range(num_epochs):
        # Creates a training set iterator
        print('[%03d] Training...\t' % epoch, end='', flush=True)
        run_loss = 0
        train_iter = iter(tr_loader)
        for idx, dat in enumerate(train_iter):
            pics = dat[0].to(device).float()
            clss = dat[1].to(device).float()
            out = mdl(pics)
            optim.zero_grad()
            loss = loss_func(out, clss)
            run_loss += loss.item()
            loss.backward()
            optim.step()
            if (idx + 1) % tr_btch_interval == 0:
                cur_per = ((idx +1) // tr_btch_interval) * 25
                print('%4d%%' % (cur_per) , end='', flush=True)
        print('\t%.5f' % (run_loss / len(tr_loader)))
        # Creates a test set iterator
        print('[%03d] Validating...\t' % epoch, end='', flush=True)
        run_loss = 0
        valid_iter = iter(te_loader)
        with torch.no_grad():
            for idx, dat in enumerate(valid_iter):
                pics = dat[0].to(device).float()
                clss = dat[1].to(device).float()
                out = mdl(pics)
                run_loss += F.binary_cross_entropy_with_logits(out, clss).item()
                if (idx + 1) % te_btch_interval == 0:
                    cur_per = ((idx +1) // te_btch_interval) * 25
                    print('%4d%%' % (cur_per), end='', flush=True)
        print('\t%.5f' % (run_loss / len(te_loader)))
        save_epoch(mdl, os.path.join(save_dir), epoch, run_loss / len(te_loader)) # tot_loss / tot_test)
    torch.save(mdl, finl_sve)

if __name__ == '__main__':
    main()
