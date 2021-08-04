import numpy as np

def getScore(tgt_arr, pred_arr):
    tp, tn, fp, fn = [0]*4
    for tgt_row, pred_row in zip(tgt_arr, pred_arr):
        for idx in range(num_class):
            if tgt_row[idx] == 0 and pred_row[idx] == 0:
                tn += 1
            elif tgt_row[idx] == 1 and pred_row[idx] == 1:
                tp += 1
            elif tgt_row[idx] == 0 and pred_row[idx] == 1:
                fp += 1
            else:
                fn += 1
    return [tp, tn, fp, fn]

def calcF1Score(tgt_arr, pred_arr):
    tp, tn, fp, fn = getScore(tgt_arr, pred_arr)
    f1_score = (2 * tp) / (2 * tp + fn + fp)
    return f1_score

def apply_threshold(pred_arr, threshold):
    tx_arr = []
    for row in pred_arr:
        tmp = []
        for col in row:
            if col >= threshold:
                tmp.append(1)
            else:
                tmp.append(0)
        tx_arr.append(tmp)
    return tx_arr

def test_threshold(tgt_arr, pred_arr, threshold):
    test_arr = apply_threshold(pred_arr, threshold)
    f1_score = calcF1Score(tgt_arr, test_arr)
    return f1_score

def conv2dense(init_arr, cls_size=28):
    cls = np.zeros(cls_size)
    for i in list(map(int, init_arr.split(' '))):
        cls[i] = 1
    return cls

def conv2sparse(init_arr):
    tmp = []
    for idx in range(len(init_arr)):
        if init_arr[idx] == 1:
            tmp.append(idx)
    return tmp

def conv2string(sparse_arr):
    return str(sparse_arr).lstrip('[').rstrip(']').replace(',', '')

def write_output(fnames, pred_arr, outpath):
    target = open(outpath, 'w')
    target.write('Id,Predicted\n')
    for file, row in zip(fnames, pred_arr):
        pred_sprse_str = conv2string(conv2sparse(row))
        if len(pred_sprse_str) == 0:
            pred_sprse_str = '0'
        target.write('%s,%s\n' % (file, pred_sprse_str))
    target.close()

def calc_output_size(h_w, padding, kernel_size, stride):
    h = (h_w[0] + 2*padding[0] - kernel_size[0]) // stride[0] + 1
    w = (h_w[1] + 2*padding[1] - kernel_size[1]) // stride[1] + 1
    return (h, w)