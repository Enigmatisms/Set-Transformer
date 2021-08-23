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
    Y = [max(row) for row in X]
    return X, torch.Tensor(Y).view(-1, 1)

def calcAcc(pred, Y):
    tmp = torch.abs(pred - Y)
    return torch.sum(tmp < 0.5).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 40, help = "Training lasts for . epochs")
    parser.add_argument("--max_iter", type = int, default = 20, help = "max iteration number")
    parser.add_argument("--head_num", type = int, default = 4, help = "Number of heads in MAB")
    parser.add_argument("--batch_size", type = int, default = 100, help = "Batch size of median problem")
    parser.add_argument("--set_size", type = int, default = 32, help = "Batch size of median problem")
    parser.add_argument("--eval_time", type = int, default = 5, help = "Evaluate performance every <.> times")
    parser.add_argument("--gamma", type = float, default = 0.9995, help = "Expo lr gamma coefficient")
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
    head_num = args.head_num

    med_net = Median(head_num, True)

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
    opt_sch = optim.lr_scheduler.ExponentialLR(opt, gamma = args.gamma)
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
        opt_sch.step()
        right_cnt += calcAcc(pred, Y)
        if i % eval_time == 0:
            train_acc = (right_cnt / (batch_size * eval_time))
            right_cnt = 0
            med_net.eval()
            eval_acc = 0
            with torch.no_grad():
                test_loss = 0.0
                for _ in range(10):
                    X, Y = makeMedianData(batch_size, set_size)
                    X = Var(X)
                    Y = Var(Y)
                    if use_cuda:
                        X = X.cuda()
                        Y = Y.cuda()
                    pred = med_net(X)
                    test_loss += loss_func(pred, Y)
                    eval_acc += calcAcc(pred, Y)
                test_loss *= 0.1
                eval_acc /= (10 * batch_size)
            print("Epoch: %5d / %5d\t train set loss: %.5f\t test set loss: %.5f\t acc: %.4f\t test acc: %.4f\tlr: %.8f"%(
                i, epochs, loss.item(), test_loss.item(), train_acc, eval_acc, opt_sch.get_last_lr()[-1]
            ))
            writer.add_scalar('Loss/Train Loss', loss, i)
            writer.add_scalar('Loss/Eval loss', test_loss, i)
            writer.add_scalar('Acc/Train Accuracy', train_acc, i)
            writer.add_scalar('Acc/Eval Accuracy', eval_acc, i)
            med_net.train()
    writer.close()
    print("Output completed.")
    