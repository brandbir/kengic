import os
import numpy as np
import pandas as pd
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch

from torchvision import datasets as datasets
from PIL import Image


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nlp import ngrams
import shutil

# Colab
# path_karpathy_coco_splits = '/content/drive/MyDrive/PhD_Brandon/Data/Datasets/mscoco/dataset_coco.json'
# keywords_folder = '/content/drive/MyDrive/Colab Notebooks/phd-ml-decoder/keywords'
# path_keywords_file = keywords_folder + '/vocabulary_350_pos_hist.csv'
# path_keywords_file = keywords_folder + '/vocabulary_cleaned_lemma_pos_filtered_dist_1000.csv'
# path_keywords_file = keywords_folder + '/openimages_full_intersection_mscoco_dist.csv'

path_karpathy_coco_splits = '/home/brandon/Documents/datasets/karpathy_splits/dataset_coco.json'
path_keywords_data = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/data'
path_keywords_file = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/cnn/vocabularies/vocabulary_350_pos_hist.csv'
path_keywords_file = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/cnn/vocabularies/vocabulary_cleaned_lemma_pos_filtered_dist_1000.csv'
path_keywords_file = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/openimages/openimages_full_intersection_mscoco_dist.csv'


coco_splits = {}
data = []

def initialise_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)


def read_karpathy_splits(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def load_splits(data):
    def update_dic(dic, key, value):
        if(key not in dic.keys()):
            dic[key] = []
        dic[key] += [value]
        
        return dic

    splits = {}
    filepaths_dic = {}

    for img in data['images']:
        update_dic(splits, img['split'], img)
        update_dic(filepaths_dic, img['filepath'], img['imgid'])
    
    fulltrain = splits['train'] + splits['restval']
    splits['fulltrain'] = fulltrain
    
    return splits, filepaths_dic

def load_keywords(path_file):
    keywords = list(pd.read_csv(path_file)['keyword'])
    return keywords


def filter_pos(tokens, by):
    filtered_pos_tokens = []
    
    for pt in nltk.pos_tag(tokens):
        word = pt[0]
        tag = nltk_pos_tagger(pt[1])
        if tag == by:
            filtered_pos_tokens += [word]
    
    return filtered_pos_tokens


def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = []

    for token in tokens:
        nltk_tagged = nltk.pos_tag([token])
        
        for t in nltk_tagged:
            word = t[0]
            tag = nltk_pos_tagger(t[1])
            
            if tag is not None:
                lemmatized_tokens += [lemmatizer.lemmatize(word, tag)]
            else:
                lemmatized_tokens += [word]
    
    return lemmatized_tokens

def get_ds_num_of_keywords_per_img(ds):
    keywords_sizes = []

    keywords = ds.keywords

    for idx, img in enumerate(ds.images):
        if idx % 1000 == 0:
            print(idx, len(ds.images))
        
        sentences = img['sentences']
        img_keywords = []
        target = np.zeros(len(keywords))

        for s in sentences:
            tokens = s['tokens']
            tokens_lemmatized = lemmatize_tokens(tokens)

            for token in tokens_lemmatized:
                if token in keywords and token not in img_keywords:
                    img_keywords += [token]
                    target[keywords.index(token)] = 1

        keywords_sizes += [int(np.sum(target))]

    return keywords_sizes

def count_keyword(keyword, sentences):
    count = 0

    for s in sentences:
        if keyword in s['raw'].lower():
            count += 1

    return count

    
class KarpathySplits(datasets.coco.CocoDetection):
    def __init__(self, root, splits_file, split, path_keywords_file,
                 keywords_extractor_count=1, transform=None,
                 target_transform=None):

        self.root = root
        self.splits_file = splits_file
        self.split = split

        self.data = read_karpathy_splits(self.splits_file)
        splits, self.filepaths_dic = load_splits(self.data)
        self.path_keywords_file = path_keywords_file

        self.keywords = load_keywords(self.path_keywords_file)
        self.keywords_extractor_count = keywords_extractor_count        
      
        self.transform = transform
        self.target_transform = target_transform
        
        if split == 'train':
            self.images = splits['fulltrain']
        elif split == 'val':
            self.images = splits['val']
        elif split == 'test':
            self.images = splits['test']

        print('------------------------------------------------')
        print('Karpathy Splits Load Configuration:')
        print('Root:', self.root)
        print('Split:', self.split)
        print('Splits file:', self.splits_file)
        print('Keywords file :', self.path_keywords_file)
        print('Keywords count :', self.keywords_extractor_count)
        print('------------------------------------------------')


    def __getitem__(self, index):
        img = self.images[index]
        keywords = self.keywords
        sentences = img['sentences']
        img_keywords = []
        
        target = torch.zeros(len(keywords), dtype=torch.long)
        all_tokens  = []

        for s in sentences:
            all_tokens += s['tokens']

        # tokens_dist = pd.DataFrame(pd.DataFrame(all_tokens, columns=['token']).value_counts().reset_index(name='count'))
        # filtered_tokens = list(tokens_dist[tokens_dist['count'] >= self.keywords_extractor_count]['token'].values)        
        # tokens_lemmatized = lemmatize_tokens(filtered_tokens)
        # print(tokens_lemmatized)

        max_multi_word = pd.DataFrame(keywords, columns=['keyword'])['keyword'].apply(lambda x: len(x.split())).max()

        for i in reversed(range(1,max_multi_word+1)):
            for s in sentences:
                for ng in ngrams(s['tokens'], i):
                    keyword = ' '.join(ng)
                    lemmatised_keyword = lemmatize_tokens([keyword])[0]
                    if lemmatised_keyword in keywords:
                        img_keywords_str = ' '.join(img_keywords)
                        
                        keyword_count = count_keyword(keyword, sentences)

                        if lemmatised_keyword not in img_keywords_str and keyword_count >= self.keywords_extractor_count:
                            img_keywords += [lemmatised_keyword]
                            target[keywords.index(lemmatised_keyword)] = 1

        file_path = img['filepath']
        file_name = img['filename']
        full_file_path = os.path.join(self.root, file_path, file_name)

        raw_img = Image.open(full_file_path).convert('RGB')

        if self.transform is not None:
            raw_img = self.transform(raw_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return raw_img, target, full_file_path, img['imgid']

    def __len__(self) -> int:
        return len(self.images)

    def img_captions(self, img_id):
        img = self.data['images'][img_id]
        captions = img['sentences']
        captions_list = []
        for i in range(len(captions)):
            captions_list += [captions[i]['raw']]
            
        return captions_list

    def show_image(self, index, path=None, captions=True):
        img = self.images[index]
        img_path = self.root + '/' + img['filepath'] + '/' + img['filename']
        print('index:', index, 'imgid:', img['imgid'], 'path:', img_path)
            
        img_raw = mpimg.imread(img_path)
        channels = len(img_raw.shape)

        plt.figure(figsize = (7,7))
        
        if channels == 2:    
            imgplot = plt.imshow(img_raw, cmap='gray')
        else:
            imgplot = plt.imshow(img_raw, cmap='gray')

        plt.axis('off')
        plt.show

        if captions:
            for i, ci in enumerate(self.img_captions(img['imgid'])):
                print(i+1, ci)
            
    def convert_target_to_keywords(self, target):
        return sorted(list(np.array(self.keywords)[np.array((target==1).tolist())]))


    def get_distribution_of_keywords(self):
        dist = {}
        for idx, ds_obj in enumerate(self.images):
            if idx % 1000 == 0:
                print(idx, len(self.images))
                print(len(dist.keys()))
            
            target = ds_obj[1]
            keywords = self.convert_target_to_keywords(target, self.keywords)
            
            for k in keywords:
                if k in dist.keys():
                    dist[k] += 1
                else:
                    dist[k] = 1

        return dist


    def print_img_details(self, img_idx):
        #ds_test_k3940.keywords_extractor_count = 3
        self.show_image(img_idx)
        keywords = self.convert_target_to_keywords(self[img_idx][1])
        print('\nKeywords (frequency:' + str(self.keywords_extractor_count) + '):', ', '.join(keywords))