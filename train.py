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
########################

t_p, t_n, f_p, f_n = [0] * 4
def resetGlobalCounts():
    global t_p, t_n, f_p, f_n
    t_p, t_n, f_p, f_n = [0] * 4

def updateCounts(pred, lbls):
    global t_p, t_n, f_p, f_n
    pred = np.round(pred.detach().cpu().numpy())
    lbls = lbls.data.cpu().numpy()
    for cur_btch in range(batch_size):
        cur_btch_lbls = lbls[cur_btch]
        cur_btch_pred = pred[cur_btch]
        for cur_idx in range(num_class):
            if cur_btch_lbls[cur_idx] == cur_btch_pred[cur_idx]:
                if cur_btch_lbls[cur_idx] == 1:
                    t_p += 1
                else:
                    t_n += 1
            elif cur_btch_lbls[cur_idx] == 1:
                f_n += 1
            else:
                f_p += 1

def calcGlobalF1():
    global t_p, t_n, f_p, f_n
    f1_msr = (2 * t_p) / (2 * t_p + f_n + f_p)
    resetGlobalCounts()
    return f1_msr

def save_epoch(model, save_dir, epoch_num, cur_loss):
    torch.save(model, os.path.join(save_dir, str(epoch_num)+'.model'))
    target = open(os.path.join(save_dir, 'loss_list.tsv'), 'w+')
    strTime = '_'.join([time.strftime('%Y-%m-%d'), time.strftime('%H:%M:%S')])
    target.write('%s\t%d\t%.3f\n' % (strTime, epoch_num, cur_loss))
    target.close()

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_f1(tp, tn, fp, fn):
    return (2 * tp) / (2 * tp + fn + fp)

def main():
    # Set up data loader
    dataset = DL(data_dir, 'train')
    train_size = round(0.7*len(dataset))
    train, validate = torch.utils.data.random_split(dataset, (train_size, len(dataset) - train_size))
    tr_loader = data.DataLoader(dataset=train,
                                batch_size=batch_size,
                                shuffle=True, num_workers=4)
    te_loader = data.DataLoader(dataset=validate,
                                batch_size=batch_size,
                                shuffle=False,num_workers=4)
    # Set up the classifier
    mdl = Classifier(num_chan=num_chan).to(device)
    # Set up loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    # Set up the optimizer
    optim = torch.optim.Adam(mdl.parameters(), lr=l_r)
    for epoch in range(num_epochs):
        # Creates a training set iterator
        train_iter = iter(tr_loader)
        for idx, dat in enumerate(train_iter):
            if idx >= 5000:
                break
            pics = dat[0].to(device).float()
            clss = dat[1].to(device).float()
            out = mdl(pics)
            optim.zero_grad()
            loss = loss_func(out, clss)
            loss.backward()
            optim.step()
            if (idx + 1) % 2000 == 0:
                print('[%3d-%05d] Training' % (epoch, idx+1))
        # save_epoch(mdl, os.path.join(save_dir), epoch, -1) # tot_loss / tot_test)
        # Creates a test set iterator
        valid_iter = iter(te_loader)
        run_loss = 0
        with torch.no_grad():
            for idx, dat in enumerate(valid_iter):
                if idx >= 2000:
                    break
                pics = dat[0].to(device).float()
                clss = dat[1].to(device).float()
                out = mdl(pics)
                run_loss += F.binary_cross_entropy_with_logits(out, clss).item()
                if (idx + 1) % 2000 == 0:
                    print('[%3d-%05d] Validation' % (epoch, idx+1))
        print('[%03d] :: %.5f' % (epoch, run_loss / len(te_loader)))
    torch.save(mdl, finl_sve)


if __name__ == '__main__':
    main()
