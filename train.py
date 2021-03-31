import argparse
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ClickbaitDetector
from data_loader import ClickbaitDataset
from torch.utils.data.sampler import WeightedRandomSampler
import torch.multiprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

torch.multiprocessing.set_sharing_strategy('file_system')


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  Creating %s...  ---" % path)
        print("---  OK  ---")

    else:
        print("---  %s already exists!  ---" % path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--thumbnail_folder', type=str,
                        default="../../crawled_data/thumbnails")
    parser.add_argument('--vocab_path', required=True)
    parser.add_argument('--text_encoder', type=str, default='mean')
    parser.add_argument('--fusion_layer', type=str, default='concat')
    parser.add_argument('--word_vec_dim', type=int, default=100)
    parser.add_argument('--text_dim', type=int, default=128)
    parser.add_argument('--fusion_dim', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--save_freq', type=int, default=200)
    parser.add_argument('--prob_data', default=None)
    parser.add_argument('--multigpu', dest='multigpu',
                        default=False, action='store_true')
    parser.add_argument('--oversampling', action='store_true')
    parser.add_argument('--prob_frac', type=float, default=1.0)

    args = parser.parse_args()

    save_dir = args.save_dir + '/' + args.experiment_name
    mkdir(save_dir)

    # load dataset...
    train = ClickbaitDataset(args.train_path,
                             args.thumbnail_folder,
                             args.vocab_path,
                             prob_data=args.prob_data,
                             val_num=100,
                             prob_frac=args.prob_frac)

    test = ClickbaitDataset(args.test_path,
                            args.thumbnail_folder,
                            args.vocab_path,
                            val_num=0)

    print('training set size: {}'.format(len(train)))

    if args.oversampling:
        weights = [2 if label == 1 else 1 for label,
                   thumbnail, frames, other_feature in train]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train), replacement=True)
        train_dl = DataLoader(
            train, batch_size=args.batch_size, num_workers=4, sampler=sampler)
    else:
        train_dl = DataLoader(
            train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dl = DataLoader(test, batch_size=args.batch_size)

    # prepare model
    model = ClickbaitDetector(args)

    if torch.cuda.is_available():
        print("use gpu...")
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")

    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.0001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.multigpu:
        model = nn.DataParallel(model)

    # training...
    val_loss_list = []
    val_acc_list = []
    val_prec_list = []
    val_recall_list = []
    val_f1_list = []
    test_acc_list = []
    test_prec_list = []
    test_recall_list = []
    test_f1_list = []

    min_val_idx = -1
    idx = 0
    stop_buff = 0
    print("-------Start Training------")
    model.train()
    niter = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_dl):
            label, word_vecs, thumbnail = data
            label = label.float().to(DEVICE)
            word_vecs, thumbnail = word_vecs.to(DIVICE), thumbnail.to(DEVICE)

            optimizer.zero_grad()
            label_hat = model(word_vecs, thumbnail)

            loss = criterion(label_hat, label)
            loss.backward()
            optimizer.step()
            niter += 1
            running_loss += loss.item()

            if niter % 10 == 9:
                print('[Epoch %d, iter %d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss/10))
                running_loss = 0.0

            # take a test in val set and test set
            if niter % args.save_freq == 0:
                print('Saving model....')
                torch.save(model.state_dict(), save_dir + '/' + args.experiment_name +
                           '_Epoch' + str(epoch).zfill(3) + '_niter' + str(niter).zfill(5) + '.pkl')
                model.eval()
                print('Take a test in val set, Epoch:',
                      str(epoch), ', Niter:', str(niter))

                # val set evaluation...
                label_list, word_vecs_list, thumbnail_list = train.get_val_set()
                Y = []
                Y_hat = []
                val_loss = 0
                outputs_list = []
                for i in range(len(label_list)):
                    label, word_vecs, thumbnail = label_list[i], word_vecs_list[i], thumbnail_list[i]
                    label = label.float()
                    word_vecs, thumbnail = word_vecs.to(
                        DEVICE), thumbnail.to(DEVICE)
                    word_vecs, thumbnail = word_vecs.unsqueeze(
                        0), thumbnail.unsqueeze(0)

                    with torch.no_grad():
                        label_hat = model(word_vecs, thumbnail).cpu()

                    outputs_list.append(label_hat)
                    label_hat = torch.round(label_hat)
                    Y.append(label.numpy())
                    Y_hat.extend(label_hat.detach().numpy())

                val_loss += criterion(torch.cat(outputs_list).view(-1),
                                      torch.cat(label_list).view(-1))

                val_loss_list.append(val_loss)
                val_acc = accuracy_score(Y, Y_hat)
                val_prec = precision_score(Y, Y_hat)
                val_recall = recall_score(Y, Y_hat)
                val_f1 = f1_score(Y, Y_hat)

                print("Loss", val_loss)
                print("Accuracy: ", val_acc)
                print("Precision: ", val_prec)
                print("Recall: ", val_recall)
                print("F1 score: ", val_f1)

                val_acc_list.append(val_acc)
                val_prec_list.append(val_prec)
                val_recall_list.append(val_recall)
                val_f1_list.append(val_f1)

                if min_val_idx < 0:
                    min_val_idx = 0
                else:
                    if val_acc_list[min_val_idx] <= val_acc:
                        min_val_idx = idx
                        stop_buff = 0
                    else:
                        stop_buff += 1

                # test set evalution...
                print('Take a test in test set, Epoch:',
                      str(epoch), ', Niter:', str(niter))
                Y = []
                Y_hat = []
                for i, data in enumerate(test_dl):
                    label, word_vecs, thumbnail = data
                    label = label.float()
                    word_vecs, thumbnail = word_vecs.to(DEVICE), thumbnail.to(DEVICE)

                    with torch.no_grad():
                        label_hat = model(word_vecs, thumbnail).cpu()

                    label_hat = torch.round(label_hat)
                    Y.extend(label.numpy())
                    Y_hat.extend(label_hat.detach().numpy())

                test_acc = accuracy_score(Y, Y_hat)
                test_prec = precision_score(Y, Y_hat)
                test_recall = recall_score(Y, Y_hat)
                test_f1 = f1_score(Y, Y_hat)

                print("Accuracy: ", test_acc)
                print("Precision: ", test_prec)
                print("Recall: ", test_recall)
                print("F1 score: ", test_f1)

                test_acc_list.append(test_acc)
                test_prec_list.append(test_prec)
                test_recall_list.append(test_recall)
                test_f1_list.append(test_f1)

                if stop_buff >= 15:
                    break
                idx += 1
                model.train()

        if stop_buff >= 15:
            break

    print("----Best model evaluation-----")
    print("Accuracy:", test_acc_list[min_val_idx])
    print("Precision:", test_prec_list[min_val_idx])
    print("Recall:", test_recall_list[min_val_idx])
    print("F1 score", test_f1_list[min_val_idx])
