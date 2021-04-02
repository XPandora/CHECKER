import numpy as np
import pandas as pd
import random
import torch
import spacy
import glob
from gensim.models import KeyedVectors
from torchtext import data, datasets
from torchtext.vocab import Vectors
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from PIL import Image, ImageOps

# use 5-fold cross validation
K = 5


class ClickbaitDataset(data.Dataset):
    def __init__(self, data_path, thumbnail_folder, vocab_path, text_model_path=None, text_model_type='glove',
                 prob_data=None, val_index=0, prob_frac=1.0, isTrain=False, use_img_aug=False, incongruent_rate=0):
        super(ClickbaitDataset, self).__init__()
        # read labeled training set
        self.dataset = pd.read_csv(data_path)[['ID', 'title', 'label']]
        self.dataset['label'] = self.dataset['label'].map(
            {'clickbait': 1.0, 'non-clickbait': 0.0})
        self.thumbnail_folder = thumbnail_folder
        self.isTrain = isTrain
        self.incongruent_rate = incongruent_rate

        # prepare train and val set
        if isTrain:
            assert val_index < K and val_index >= 0
            len_per_set = len(self.dataset) // 5

            if val_index == 0:
                self.train_dataset = self.dataset[len_per_set:]
                self.val_dataset = self.dataset[:len_per_set]
            else:
                val_start_index = len_per_set * val_index
                val_end_index = val_start_index + len_per_set
                
                self.train_dataset = pd.concat(
                    [self.dataset[:val_start_index], self.dataset[val_end_index:]],
                    axis=0)
                self.val_dataset = self.dataset[val_start_index:val_end_index]
                
        else:
            self.train_dataset = self.dataset
            self.val_dataset = None

        # optionally use data with generated labels
        if prob_data != None:
            df = pd.read_csv(prob_data)[['ID', 'title', 'label']]
            df = df.sample(frac=prob_frac)
            df['label'] = df['label'].map(
                {'clickbait': 1.0, 'nonclickbait': 0.0, 'non-clickbait': 0.0})
            self.train_dataset = pd.concat([self.train_dataset, df], axis=0)

        # prepare text embedding model
        self.text_model_type = text_model_type
        if text_model_type == 'glove':
            self.vocab = KeyedVectors.load_word2vec_format(vocab_path)
            self.spacy_en = spacy.load('en')
        elif text_model_type == 'infersent':
            from InferSent.models import InferSent
            model_version = 1
            MODEL_PATH = text_model_path
            params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                            'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
            infersent = InferSent(params_model)
            infersent.load_state_dict(torch.load(MODEL_PATH))
            infersent = infersent.cuda()
            infersent.set_w2v_path(vocab_path)
            infersent.build_vocab_k_words(K=400000)
            self.infersent = infersent
        else:
            raise ValueError(
                '{} not supported for text model'.format(text_model_type))

        # prepare transform for images
        self.val_transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
        if isTrain & use_img_aug:
            self.transform = transforms.Compose([transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
                                                 transforms.RandomHorizontalFlip(
                                                     p=0.5),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])

    def tokenizer(self, text):
        return [toke.text for toke in self.spacy_en.tokenizer(text)]

    def __getitem__(self, index):
        # get label
        label = self.train_dataset['label'].values[index]
        
        # set index for title and thumbnail
        index_title = index
        index_thumbnail = index
        
        # make the incongruent pair based on rate, which is also clickbait
        if label == 1 and self.incongruent_rate > 0:
            p = random.random()
            
            if p < self.incongruent_rate:
                # mismatch title and thumbnail index
                while index_title == index_thumbnail:
                    index_title = random.randint(0, len(self.train_dataset)-1)
                    index_thumbnail = random.randint(0, len(self.train_dataset)-1)
        
        label = torch.tensor([label])
        
        # get word embedding
        title = self.train_dataset['title'].values[index_title]
        if self.text_model_type == 'glove':
            tokens = self.tokenizer(title)
            word_vecs = []
            for token in tokens:
                if self.vocab.has_index_for(token):
                    word_vecs.append(torch.tensor(self.vocab[token]))
                # word_vecs.append(torch.randn(100))

            if len(word_vecs) == 0:
                print('empty token detected: {}'.format(title))
                # use 'a' to denote empty token...
                word_vecs = [torch.tensor(self.vocab['a'])]

            word_vecs = torch.stack(word_vecs, dim=0)
            word_vecs = torch.mean(word_vecs, dim=0).squeeze()
        else:
            word_vecs = self.infersent.encode(title)[0]
            word_vecs = torch.tensor(word_vecs)

        # get thumbnail
        id = self.train_dataset['ID'].values[index_thumbnail]

        thumbnail_path = self.thumbnail_folder + "/" + id + '.jpg'
        thumbnail = Image.open(thumbnail_path)
        thumbnail = thumbnail.crop((0, 45, 480, 315))
        thumbnail = self.transform(thumbnail)

        return label, word_vecs, thumbnail

    def __len__(self):
        return len(self.train_dataset)

    def get_val_set(self):
        # get label
        label_list = torch.tensor(self.val_dataset['label'].values)

        # get word embedding
        title_list = self.val_dataset['title'].values
        word_vecs_list = []

        for title in title_list:
            if self.text_model_type == 'glove':
                tokens = self.tokenizer(title)
                word_vecs = []
                for token in tokens:
                    if self.vocab.has_index_for(token):
                        word_vecs.append(torch.tensor(self.vocab[token]))
                    # word_vecs.append(torch.randn(100))

                if len(word_vecs) == 0:
                    print('empty token detected: {}'.format(title))
                    # use 'a' to denote empty token...
                    word_vecs = [torch.tensor(self.vocab['a'])]

                word_vecs = torch.stack(word_vecs, dim=0)
                word_vecs = torch.mean(word_vecs, dim=0).squeeze()
            else:
                word_vecs = self.infersent.encode(title)[0]
                word_vecs = torch.tensor(word_vecs)

            word_vecs_list.append(word_vecs)

        # get thumbnail
        id_list = self.val_dataset['ID'].values
        thumbnail_list = []

        for id in id_list:
            thumbnail_path = self.thumbnail_folder + "/" + id + '.jpg'
            thumbnail = Image.open(thumbnail_path)
            thumbnail = thumbnail.crop((0, 45, 480, 315))
            thumbnail = self.val_transform(thumbnail)
            thumbnail_list.append(thumbnail)

        return label_list, word_vecs_list, thumbnail_list