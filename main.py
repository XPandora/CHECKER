# -*- coding:utf-8 -*-
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from data_loader import ClickbaitDataset
from model import ClickbaitDetector
import sklearn.metrics
import argparse
import sys
import numpy as np
import datetime
import shutil

from loss import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, required=True)
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--vocab_path', type=str, required=True)
parser.add_argument('--prob_data_path', type=str, default=None)
parser.add_argument('--thumbnail_folder', type=str,
                    default="../../crawled_data/thumbnails")
parser.add_argument('--text_encoder', type=str, default='mean')
parser.add_argument('--text_model_type', type=str, default='glove')
parser.add_argument('--text_model_path', type=str, default=None)
parser.add_argument('--fusion_layer', type=str, default='concat')
parser.add_argument('--word_vec_dim', type=int, default=100)
parser.add_argument('--text_dim', type=int, default=128)
parser.add_argument('--fusion_dim', type=int, default=256)
parser.add_argument('--att_dim', type=int, default=512)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--val_index', type=int, default=0)
parser.add_argument('--result_dir', type=str,
                    help='dir to save result txt files', default='results/')
parser.add_argument('--forget_rate', type=float,
                    help='forget rate', default=None)
parser.add_argument('--data_frac', type=float,
                    help='fraction of train set to use', default=1.0)
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--oversampling', action='store_true')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--incongruent_rate', type=float, default=0)

args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

# Prepare Dataset
train_set = ClickbaitDataset(
    args.train_path, args.thumbnail_folder,
    vocab_path=args.vocab_path,
    text_model_path=args.text_model_path,
    text_model_type=args.text_model_type,
    prob_data=args.prob_data_path,
    val_index=args.val_index,
    incongruent_rate=args.incongruent_rate,
    isTrain=True)

test_set = ClickbaitDataset(
    args.test_path, args.thumbnail_folder,
    vocab_path=args.vocab_path,
    text_model_path=args.text_model_path,
    text_model_type=args.text_model_type,
    incongruent_rate=0,
    isTrain=False)

print("training set size: {}".format(len(train_set)))

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / \
        (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1


# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*args.forget_rate
rate_schedule[:args.num_gradual] = np.linspace(
    0, args.forget_rate**args.exponent, args.num_gradual)

model_str = args.exp_name
save_dir = os.path.join(args.result_dir, model_str)

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

txtfile = os.path.join(save_dir, 'logs.txt')
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model


def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    print('Training %s...' % model_str)
    # pure_ratio_list=[]
    # pure_ratio_1_list=[]
    # pure_ratio_2_list=[]

    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0

    for i, (labels, word_vecs, thumbnail) in enumerate(train_loader):
        # ind=indexes.cpu().numpy().transpose()
        if i > args.num_iter_per_epoch:
            break

        labels = labels.cuda().long()
        word_vecs, thumbnail, = word_vecs.cuda(), thumbnail.cuda()

        # Forward + Backward + Optimize
        logits1 = model1(word_vecs, thumbnail)
        prec1, _ = accuracy(logits1, labels, topk=(1, 1))
        train_total += 1
        train_correct += prec1

        logits2 = model2(word_vecs, thumbnail)
        prec2, _ = accuracy(logits2, labels, topk=(1, 1))
        train_total2 += 1
        train_correct2 += prec2
        loss_1, loss_2 = loss_coteaching(
            logits1, logits2, labels, rate_schedule[epoch])
        # pure_ratio_1_list.append(100*pure_ratio_1)
        # pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        if (i+1) % args.print_freq == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f'
                  % (epoch, args.n_epoch, i+1, len(train_set)//batch_size, prec1, prec2, loss_1.data, loss_2.data))

    train_acc1 = float(train_correct)/float(train_total)
    train_acc2 = float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2


# Evaluate the Model on val set
def evaluate_val(train_loader, model1, model2):
    print('Evaluating val set %s...' % model_str)
    label_list, word_vecs_list, thumbnail_list = train_loader.get_val_set()
    label_list = label_list.long()
    metrics = {}

    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    label1_list = []
    pred1_list = []
    output1_list = []

    for i in range(len(label_list)):
        labels = label_list[i].view(-1)
        word_vecs = word_vecs_list[i].unsqueeze_(dim=0)
        thumbnail = thumbnail_list[i].unsqueeze_(dim=0)
        word_vecs, thumbnail = word_vecs.cuda(), thumbnail.cuda()

        with torch.no_grad():
            logits1 = model1(word_vecs, thumbnail).cpu()
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)

        label1_list.extend(labels.cpu().numpy())
        pred1_list.extend(pred1.cpu().numpy())
        output1_list.append(outputs1.cpu())
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    label2_list = []
    pred2_list = []
    output2_list = []
    for i in range(len(label_list)):
        labels = label_list[i].view(-1)
        word_vecs = word_vecs_list[i]
        thumbnail = thumbnail_list[i]
        word_vecs, thumbnail = word_vecs.cuda(), thumbnail.cuda()

        with torch.no_grad():
            logits2 = model2(word_vecs, thumbnail).cpu()
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)

        label2_list.extend(labels.cpu().numpy())
        pred2_list.extend(pred2.cpu().numpy())
        output2_list.append(outputs2.cpu())
        total2 += labels.size(0)
        # correct2 += (pred2.cpu() == labels).sum()

    metrics['val_loss1'] = F.cross_entropy(
        torch.cat(output1_list, dim=0), label_list.long())
    metrics['val_loss2'] = F.cross_entropy(
        torch.cat(output2_list, dim=0), label_list.long())
    metrics['acc1'] = 100*float(correct1)/float(total1)
    metrics['acc2'] = 100*float(correct2)/float(total2)
    
    metrics['auc1'] = sklearn.metrics.roc_auc_score(label1_list, torch.cat(output1_list, dim=0).numpy()[:,1])
    metrics['auc2'] = sklearn.metrics.roc_auc_score(label2_list, torch.cat(output2_list, dim=0).numpy()[:,1])
    metrics['fscore1'] = 100*sklearn.metrics.f1_score(label1_list, pred1_list)
    metrics['fscore2'] = 100*sklearn.metrics.f1_score(label2_list, pred2_list)
    metrics['prec1'] = 100 * \
        sklearn.metrics.precision_score(label1_list, pred1_list)
    metrics['prec2'] = 100 * \
        sklearn.metrics.precision_score(label2_list, pred2_list)
    metrics['rec1'] = 100*sklearn.metrics.recall_score(label1_list, pred1_list)
    metrics['rec2'] = 100*sklearn.metrics.recall_score(label2_list, pred2_list)

    return metrics

# Evaluate the Model on test_set


def evaluate(test_loader, model1, model2):
    print('Evaluating on test set %s...' % model_str)
    metrics = {}
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    label1_list = []
    pred1_list = []
    output1_list = []
    for labels, word_vecs, thumbnail in test_loader:
        word_vecs, thumbnail = word_vecs.cuda(), thumbnail.cuda()
        labels = labels.view(-1).long()

        with torch.no_grad():
            logits1 = model1(word_vecs, thumbnail)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)

        label1_list.extend(labels.cpu().numpy())
        pred1_list.extend(pred1.cpu().numpy())
        output1_list.append(outputs1.cpu())
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()

    model2.eval()    # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    label2_list = []
    pred2_list = []
    output2_list = []
    for labels, word_vecs, thumbnail in test_loader:
        word_vecs, thumbnail = word_vecs.cuda(), thumbnail.cuda()
        labels = labels.view(-1).long()

        with torch.no_grad():
            logits2 = model2(word_vecs, thumbnail)
            outputs2 = F.softmax(logits2, dim=1)
            _, pred2 = torch.max(outputs2.data, 1)

        label2_list.extend(labels.cpu().numpy())
        pred2_list.extend(pred2.cpu().numpy())
        output2_list.append(outputs2.cpu())
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()

    metrics['acc1'] = 100*float(correct1)/float(total1)
    metrics['acc2'] = 100*float(correct2)/float(total2)
    metrics['auc1'] = sklearn.metrics.roc_auc_score(label1_list, torch.cat(output1_list, dim=0).numpy()[:,1])
    metrics['auc2'] = sklearn.metrics.roc_auc_score(label2_list, torch.cat(output2_list, dim=0).numpy()[:,1])
    metrics['fscore1'] = 100*sklearn.metrics.f1_score(label1_list, pred1_list)
    metrics['fscore2'] = 100*sklearn.metrics.f1_score(label2_list, pred2_list)
    metrics['prec1'] = 100 * \
        sklearn.metrics.precision_score(label1_list, pred1_list)
    metrics['prec2'] = 100 * \
        sklearn.metrics.precision_score(label2_list, pred2_list)
    metrics['rec1'] = 100*sklearn.metrics.recall_score(label1_list, pred1_list)
    metrics['rec2'] = 100*sklearn.metrics.recall_score(label2_list, pred2_list)

    return metrics


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    if args.oversampling:
        weights = [2 if label == 1 else 1 for label,
                   word_vecs, thumbnail in train_set]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_set), replacement=True)
        train_dl = DataLoader(train_set, batch_size=batch_size,
                              num_workers=0, sampler=sampler)
    else:
        train_dl = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_dl = DataLoader(test_set, batch_size=batch_size)
    # Define models
    print('building model...')
    model1 = ClickbaitDetector(args)
    model2 = ClickbaitDetector(args)
    if args.multigpu:
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)

    model1.cuda()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

    model2.cuda()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    mean_pure_ratio1 = 0
    mean_pure_ratio2 = 0

    with open(txtfile, "a") as myfile:
        myfile.write('------------ Options -------------\n')
        dic = vars(args)
        for k, v in sorted(dic.items()):
            myfile.write('%s: %s\n' % (str(k), str(v)))
        myfile.write('\n')
        myfile.write('------------ Training -------------\n')
        myfile.write(
            'epoch: train_acc1 train_acc2 val_loss1 val_loss2 val_fscore1 val_fscore2 val_auc1 val_auc2 test_auc1 test_auc2 test_acc1 test_acc2 test_fscore1 test_fscore2 test_prec1 test_prec2 test_rec1 test_rec2\n')

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0
    # evaluate models with random weights
    val_metrics = evaluate_val(train_set, model1, model2)
    test_metrics = evaluate(test_dl, model1, model2)

    print('Epoch [%d/%d] Test on the %s test images: \
            Model1 val loss %.4f, Model2 val loss %.4f ; \
            Model1 val fscore %.4f, Model2 val fscore %.4f ; \
            Model1 val auc %.4f, Model2 val auc %.4f ; \
            Model1 acc %.4f , Model2 acc %.4f ; \
            Model1 auc %.4f, Model2 auc %.4f ; \
            Model1 f1 %.4f , Model2 f1 %.4f ; \
            Model1 prec %.4f , Model2 prec %.4f ; \
            Model1 recall %.4f , Model2 recall %.4f'
          % (epoch, args.n_epoch, len(test_set),
             val_metrics['val_loss1'], val_metrics['val_loss2'],
             val_metrics['fscore1'], val_metrics['fscore2'],
             val_metrics['auc1'], val_metrics['auc2'],
             test_metrics['acc1'], test_metrics['acc2'],
             test_metrics['auc1'], test_metrics['auc2'],
             test_metrics['fscore1'], test_metrics['fscore2'],
             test_metrics['prec1'], test_metrics['prec2'],
             test_metrics['rec1'], test_metrics['rec2'])
          )

    val_loss_min = min(val_metrics['val_loss1'], val_metrics['val_loss2'])
    f1_score_max = max(val_metrics['fscore1'], val_metrics['fscore2'])
    auc_max = max(val_metrics['auc1'], val_metrics['auc2'])
    best_epoch_loss = 0
    best_epoch_fscore = 0
    best_epoch_auc = 0

    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(
            '{}: {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                epoch, train_acc1, train_acc2, val_metrics['val_loss1'],
                val_metrics['val_loss2'], val_metrics['fscore1'], val_metrics['fscore2'],
                val_metrics['auc1'], val_metrics['auc2'], test_metrics['auc1'],
                test_metrics['auc2'], test_metrics['acc1'], test_metrics['acc2'],
                test_metrics['fscore1'], test_metrics['fscore2'], test_metrics['acc1'],
                test_metrics['acc2'], test_metrics['prec1'], test_metrics['prec2'],
                test_metrics['rec1'], test_metrics['rec2']
            )
        )

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        model1.train()
        adjust_learning_rate(optimizer1, epoch)
        model2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2 = train(
            train_dl, epoch, model1, optimizer1, model2, optimizer2)

        # evaluate models
        val_metrics = evaluate_val(train_set, model1, model2)
        test_metrics = evaluate(test_dl, model1, model2)

        print('Epoch [%d/%d] Test on the %s test images: \
            Model1 val loss %.4f, Model2 val loss %.4f ; \
            Model1 val fscore %.4f, Model2 val fscore %.4f ; \
            Model1 val auc %.4f, Model2 val auc %.4f ; \
            Model1 acc %.4f , Model2 acc %.4f ; \
            Model1 auc %.4f, Model2 auc %.4f ; \
            Model1 f1 %.4f , Model2 f1 %.4f ; \
            Model1 prec %.4f , Model2 prec %.4f ; \
            Model1 recall %.4f , Model2 recall %.4f'
              % (epoch, args.n_epoch, len(test_set),
                 val_metrics['val_loss1'], val_metrics['val_loss2'],
                 val_metrics['fscore1'], val_metrics['fscore2'],
                 val_metrics['auc1'], val_metrics['auc2'],
                 test_metrics['acc1'], test_metrics['acc2'],
                 test_metrics['auc1'], test_metrics['auc2'],
                 test_metrics['fscore1'], test_metrics['fscore2'],
                 test_metrics['prec1'], test_metrics['prec2'],
                 test_metrics['rec1'], test_metrics['rec2'])
              )

        # save results
        # mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        # mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)

        with open(txtfile, "a") as myfile:
            myfile.write(
                '{}: {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    epoch, train_acc1, train_acc2, val_metrics['val_loss1'],
                    val_metrics['val_loss2'], val_metrics['fscore1'], val_metrics['fscore2'],
                    val_metrics['auc1'], val_metrics['auc2'], test_metrics['auc1'],
                    test_metrics['auc2'], test_metrics['acc1'], test_metrics['acc2'],
                    test_metrics['fscore1'], test_metrics['fscore2'], test_metrics['acc1'],
                    test_metrics['acc2'], test_metrics['prec1'], test_metrics['prec2'],
                    test_metrics['rec1'], test_metrics['rec2']
                )
            )

        # use val loss to choose better model
        if min(val_metrics['val_loss1'], val_metrics['val_loss2']) < val_loss_min:
            best_epoch_loss = epoch
            val_loss_min = min(
                val_metrics['val_loss1'], val_metrics['val_loss2'])

        if max(val_metrics['fscore1'], val_metrics['fscore2']) > f1_score_max:
            best_epoch_fscore = epoch
            f1_score_max = max(val_metrics['fscore1'], val_metrics['fscore2'])

        if max(val_metrics['auc1'], val_metrics['auc2']) > auc_max:
            best_epoch_auc = epoch
            auc_max = max(val_metrics['auc1'], val_metrics['auc2'])

        # save models
        # model1_path = os.path.join(
        #     save_dir, 'model1_Epoch{}.pth'.format(epoch))
        # model2_path = os.path.join(
        #     save_dir, 'model2_Epoch{}.pth'.format(epoch))
        # torch.save(model1.state_dict(), model1_path)
        # torch.save(model2.state_dict(), model2_path)

    print('Best model(val loss) is on Epoch {}'.format(best_epoch_loss))
    print('Best model(val fscore) is on Epoch {}'.format(best_epoch_fscore))
    print('Best model(val auc) is on Epoch {}'.format(best_epoch_auc))
    with open(txtfile, "a") as myfile:
        myfile.write(
            'Best model(val loss) is on Epoch {}\n'.format(best_epoch_loss))
        myfile.write(
            'Best model(val fscore) is on Epoch {}\n'.format(best_epoch_fscore))
        myfile.write(
            'Best model(val auc) is on Epoch {}\n'.format(best_epoch_auc))


if __name__ == '__main__':
    main()
