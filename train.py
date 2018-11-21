import os
import time
import datetime

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from data_loader import DL
# from model import Classifier
from vgg import Classifier

import pdb

###################
# Params
num_class = 28
data_dir = 'data'
save_dir = 'epochs'
finl_sve = os.path.join(save_dir, 'comp.model')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_size = 512
num_chan = 3
l_r = 1e-4
batch_size = 1
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
    target.write('%d\t%.3f\n' % (epoch_num, cur_loss))
    target.close()

def calc_precision(tp, fp):
    return tp / (tp + fp)

def calc_recall(tp, fn):
    return tp / (tp + fn)

def calc_f1(tp, tn, fp, fn):
    prec = calc_precision(tp, fp)
    recc = calc_recall(tp, fn)
    return 2 * ((prec * recc) / (prec + recc))

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
    mdl = Classifier().to(device)
    # Set up loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    # Set up the optimizer
    optim = torch.optim.Adam(mdl.parameters())
    for epoch in range(num_epochs):
        # For per batch training stats
        run_loss = 0 # running loss, resets at 100
        num_test = 0 # number of tests, resets at 100
        out_iter = 0 # current output set per epoch
        start_time = time.time()
        # For per epoch training stats
        tot_test = 0
        tot_loss = 0
        # Creates a training set iterator
        train_iter = iter(tr_loader)
        for idx, dat in enumerate(train_iter):
            optim.zero_grad()
            pics = dat[0].to(device).float()
            clss = dat[1].to(device).float()
            out = mdl(pics)
            updateCounts(out, clss)
            loss = loss_func(out, clss)
            loss.backward()
            run_loss += loss.item()
            tot_loss += loss.item()
            optim.step()
            num_test += 1
            tot_test += 1
            if num_test == 100:
                out_iter += 1
                f1_tmp = calcGlobalF1()
                print('[%03d-%05d] :: %.3f :: %.3f :: %03d' % (epoch, out_iter, run_loss / num_test, f1_tmp, time.time() - start_time))
                run_loss = 0
                num_test = 0
                start_time = time.time()
        save_epoch(mdl, os.path.join(save_dir), epoch, tot_loss / tot_test)
        # Creates a test set iterator
        valid_iter = iter(te_loader)
        tp, tn, fp, fn = [0] * 4
        with torch.no_grad():
            for idx, dat in enumerate(valid_iter):
                pics = dat[0].to(device).float()
                clss = dat[1].numpy()
                out = mdl(pics)
                out = ((out.data.cpu().numpy() + 1) / 2) >= 0.5
                for cur_b in range(batch_size):
                    for cur_idx in range(num_class):
                        if clss[cur_b][cur_idx] == out[cur_b][cur_idx]:
                            if clss[cur_b][cur_idx] == 1:
                                tp += 1
                            else:
                                tn += 1
                        elif clss[cur_b][cur_idx] == 1:
                            fn += 1
                        else:
                            fp += 1
        f1_measure = calc_f1(tp, tn, fp, fn)
        print('[%03d] :: f1_measure :: %.3f' % (epoch, f1_measure))
    torch.save(mdl, finl_sve)


if __name__ == '__main__':
    main()
