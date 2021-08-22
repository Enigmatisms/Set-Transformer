#-*-coding:utf-8-*-
"""
    Set transformer median problem
"""
import os
import torch
import shutil
import argparse
from torch import optim
from datetime import datetime
from torch.autograd import Variable as Var
from torchvision.utils import save_image

from torch import optim
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from median import Median

def makeMedianData(batch_size, set_size):
    X = torch.rand(batch_size, set_size) * 100
    half = set_size // 2
    if set_size & 1:
        Y = [sorted(row)[half] for row in X]
    else:
        half_1 = half - 1
        Y = []
        for row in X:
            sorted_row = sorted(row)
            Y.append((sorted_row[half] + sorted_row[half_1]) / 2)
    return X, torch.Tensor(Y).view(-1, 1)

def calcAcc(pred, Y):
    tmp = torch.abs(pred - Y)
    return torch.sum(tmp < 0.1).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 40, help = "Training lasts for . epochs")
    parser.add_argument("--max_iter", type = int, default = 20, help = "max iteration number")
    parser.add_argument("--head_num", type = int, default = 4, help = "Number of heads in MAB")
    parser.add_argument("--batch_size", type = int, default = 100, help = "Batch size of median problem")
    parser.add_argument("--set_size", type = int, default = 32, help = "Batch size of median problem")
    parser.add_argument("--eval_time", type = int, default = 5, help = "Evaluate performance every <.> times")
    parser.add_argument("-d", "--del_dir", action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-c", "--cuda", default = False, action = "store_true", help = "Use CUDA to speed up training")
    args = parser.parse_args()

    epochs = args.epochs
    del_dir = args.del_dir
    use_cuda = args.cuda
    max_iter = args.max_iter
    batch_size = args.batch_size
    set_size = args.set_size
    eval_time = args.eval_time

    med_net = Median(100, 4, True)

    if use_cuda and torch.cuda.is_available():
        med_net = med_net.cuda()
    else:
        print("CUDA not available.")
    
    logdir = '../logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    writer = SummaryWriter(log_dir = logdir+time_stamp)

    opt = optim.Adam(med_net.parameters(), lr = 1e-3)
    loss_func = nn.L1Loss()
    right_cnt = 0
    for i in range(epochs):
        opt.zero_grad()
        X, Y = makeMedianData(batch_size, set_size)
        X = Var(X)
        Y = Var(Y)
        if use_cuda:
            X = X.cuda()
            Y = Y.cuda()
        pred = med_net(X)
        loss = loss_func(pred, Y)
        loss.backward()
        opt.step()
        right_cnt += calcAcc(pred, Y)
        if i % eval_time == 0:
            train_acc = (right_cnt / (batch_size * i))
            med_net.eval()
            eval_acc = 0
            with torch.no_grad():
                X, Y = makeMedianData(1000, set_size)
                X = Var(X)
                Y = Var(Y)
                if use_cuda:
                    X = X.cuda()
                    Y = Y.cuda()
                pred = med_net(X)
                test_loss = loss_func(pred, Y)
                eval_cnt = calcAcc(pred, Y) * 0.001
            print("Epoch: %5d / %5d\t train set loss: %.5f\t test set loss: %.5f\t acc: %.4f\t test acc: %.4f"%(
                i, epochs, loss.item(), test_loss.item(), train_acc, eval_acc
            ))
            writer.add_scalar('Loss/Train Loss', loss, i)
            writer.add_scalar('Loss/Eval loss', test_loss, i)
            writer.add_scalar('Acc/Train Accuracy', train_acc, i)
            writer.add_scalar('Acc/Eval Accuracy', eval_acc, i)
    writer.close()
    print("Output completed.")
    