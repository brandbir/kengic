import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.parse.corenlp import CoreNLPDependencyParser

import pandas as pd
import traceback
import numpy as np

PATH_STOPWORDS = '/home/brandon/Documents/git/phd/phd-image-captioning/KGCap/src/v2/stopwords.csv'


multi_word_preps = ['close to', 'far from', 'forward of', 'in between', 'in front of', 'near to',
                    'next to', 'on board', 'on top of', 'out of', 'outside of', 'together with', 'up against', 'along with', 'away from', 'by means of',
                    'further to', 'in between', 'off of', 'out of', 'prior to', 'up to', 'up until', 'apart from']

subordinate_conjunctions = ['after', 'although', 'as', 'because', 'before', 'even if',
                            'even though', 'if', 'in order that', 'once', 'provided that',
                            'rather than', 'since', 'so that', 'than', 'that', 'though', 'unless',
                            'until', 'when', 'whenever', 'where', 'whereas', 'wherever', 'whether', 'while', 'why']

visen_coco_prepositions = ['with', 'on', 'in', 'of', 'near', 'at', 'next to', 'by', 'under', 'beside', 'over', 'behind',
                            'inside', 'underneath', 'above', 'in front of', 'between']


def clean_tokens(tokens):
    stopwords = get_stopwords()

    cleaned_tokens = []

    for t in tokens:
        if t not in stopwords and t.isalpha():
            cleaned_tokens += [t]
            
    return cleaned_tokens


def ngrams(x, n):
    ngrams = []
    for idx, ix in enumerate(x[0:len(x)-n+1]):
        #print(idx, ix)
        gram = x[idx:idx+n]
        ngrams += [gram]
        
    return ngrams

def filter_pos(tokens, by):
    filtered_pos_tokens = []
    
    for pt in nltk.pos_tag(tokens):
        word = pt[0]
        tag = nltk_pos_tagger(pt[1])
        if tag == by:
            filtered_pos_tokens += [word]
    
    return filtered_pos_tokens

def get_pos(tokens):
    return nltk.pos_tag(tokens)

def nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    elif nltk_tag == 'IN':
        return 'p'
    else:
        return None
    
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = []

    nltk_tagged = nltk.pos_tag(tokens)
    for t in nltk_tagged:
        word = t[0]
        tag = nltk_pos_tagger(t[1])
        if tag is not None:
            lemmatized_tokens += [lemmatizer.lemmatize(word, tag)]
        else:
            lemmatized_tokens += [word]
    
    return lemmatized_tokens


def get_stopwords():
        return list(pd.read_csv(PATH_STOPWORDS)['stopword'].values)


def is_stopword(word):
    stopwords = get_stopwords()
    if word in stopwords:
        return True
    else:
        return False

def remove_stopwords(words):    
    stopwords = get_stopwords()
    filtered_words = []

    for w in words:
        if w not in stopwords:
            filtered_words += [w]

    return filtered_words


def get_parent(n, keywords):
    def get_synset_parent(init_n, n, keywords, i=0):
        i += 1
        syn = wordnet.synsets(n)
        if len(syn) > 0:
            syn = wordnet.synsets(n)[0]
            h = syn.hypernyms()

            for hi in h:
                hi = hi.name()
                word = hi[0:hi.index('.')]
                if word in keywords:
                    return word
                else:
                    if word == 'concept' or i == 2000:
                        return None
                    else:
                        return get_synset_parent(init_n, word, keywords, i)

            return None

    return get_synset_parent(n, n, keywords)


def get_filtered_keywords(keywords_list, type='all'):
    if type == 'all':
        return keywords_list
    else:
        keywords_filtered = []
        keywords_hypo = keywords_list[:]
        keywords_hyper = keywords_list[:]

        for keyword in keywords_list:
            m = get_parent(keyword, keywords_list)
            if m is not None:
                if keyword in keywords_hyper:
                    keywords_hyper.remove(keyword)
                
                if m in keywords_hypo:
                    keywords_hypo.remove(m)

        
        if type == 'hyponyms':
            keywords_filtered = keywords_hypo
        elif type == 'hypernyms':
            keywords_filtered = keywords_hyper
        else:
            raise Exception('Type', type, 'is not supported. Type can be "all", "hyponyms", or "hypernyms".')

        return keywords_filtered

def filter_keywords(keywords_file_path, column, keyword_file_path_export):
    """
    Removes hypernyms and hyponyms from the keyword set specified by column.
    Generates both sets in keywords_file_path_export.
    """
    
    ml_predictions = pd.read_csv(keywords_file_path)
    keywords_hyper_list = []
    keywords_hypo_list = []

    len_keywords_hyper_list = []
    len_keywords_hypo_list = []

    for i, row in ml_predictions.iterrows():
        if i % 100 == 0:
            print(i, '/', len(ml_predictions))

        keywords_list = eval(row[column])
        keywords_hyper = keywords_list[:]
        keywords_hypo = keywords_list[:]

        for keyword in keywords_list:
            m = get_parent(keyword, keywords_list)
            if m is not None:
                if keyword in keywords_hyper:
                    keywords_hyper.remove(keyword)
                
                if m in keywords_hypo:
                    keywords_hypo.remove(m)
                
        keywords_hyper_list += [keywords_hyper]
        len_keywords_hyper_list += [len(keywords_hyper)]

        keywords_hypo_list += [keywords_hypo]
        len_keywords_hypo_list += [len(keywords_hypo)]

    ml_predictions['pred_classes_hyper'] = keywords_hyper_list
    ml_predictions['pred_classes_hypo'] =  keywords_hypo_list
    ml_predictions['num_pred_classes_hyper'] = len_keywords_hyper_list
    ml_predictions['num_pred_classes_hypo'] = len_keywords_hypo_list

    ml_predictions = ml_predictions[['img_id', 'img_path', 'actual_classes', 'pred_classes', 'pred_classes_hyper', 'pred_classes_hypo', 'num_actual_classes', 'num_pred_classes', 'num_pred_classes_hyper', 'num_pred_classes_hypo', 'pred_scores']]
    ml_predictions.to_csv(keyword_file_path_export, index=False)


def get_triplets(caption):
    def is_in(sr, srs):
        for s in srs:
            if sr in s:
                return s

        return None

    def get_visen_preps():
        preps_list = []

        for prep in visen_coco_prepositions:
            #print(prep)
            if ' ' + prep + ' ' in caption:
                prep = prep.split()
                tokens = caption.split()
                #print(prep, tokens)

                for i, token in enumerate(tokens):
                    #print(token, prep, len(prep))
                    if token == prep[0] and prep == tokens[i:i+len(prep)]:
                        #print(prep, 'found')
                        preps_list += [' '.join(prep)]
                        break

        filtered_prep_list = preps_list.copy()

        for p in preps_list:
            others = preps_list.copy()
            others.remove(p)
            for o in others:
                if p in o.split():
                    filtered_prep_list.remove(p)

        return filtered_prep_list

    
    def filter_no_subj(word, tag, obj1, obj2, sr):
        if tag == 'n':
            if len(obj2) < len(obj1):
                obj2 += [word]

            else:
                obj1 += [word]

        elif word == 'next' or (tag == 'p' and word not in subordinate_conjunctions):
            word = is_in(word, filtered_prep_list)
            
            if word is not None and word not in sr:
                if len(obj1) != len(obj2):
                    sr += [word]
                elif len(obj2) > 0:
                    obj1 += [obj2[-1]]
                    sr += [word]
        
        elif tag == 'v' and not is_stopword(word):
            if i+1 < len(nltk_tagged):
                if nltk_pos_tagger(nltk_tagged[i+1][1]) != 'p':
                    sr += [word]
        
        return obj1, obj2, sr

    def filter_with_subj(word, tag, obj1, obj2, sr):        
        if tag == 'n':
            if subj not in obj1:
                obj1 += [subj]
            else:
                obj2 += [word]
        
        elif word == 'next' or (tag == 'p' and word not in subordinate_conjunctions):
            word = is_in(word, filtered_prep_list)
            if word is not None and word not in sr:
                sr += [word]
        
        elif tag == 'v' and not is_stopword(word):
            if i+1 < len(nltk_tagged):
                if nltk_pos_tagger(nltk_tagged[i+1][1]) != 'p':
                    sr += [word]

        return obj1, obj2, sr

    triplets = []

    try:
        subj = None
        dep_parser = CoreNLPDependencyParser(url='http://localhost:9077')
        parse, = dep_parser.raw_parse(caption)
        
        obj1 = []
        obj2 = []
        sr = []
        verbs = []

        triplets = []
        
        for governor, dep, dependent in parse.triples():
            if dep == 'nsubj':
                subj = dependent[0]
                #print('subject:', subj)
            
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        #tokens = caption.split()

        tokens = []
        for ci in caption.lower().split():
            if ci.isalpha():
                tokens += [ci]
        
        filtered_prep_list = get_visen_preps()

        #print('preps found:', filtered_prep_list)
        #print('tokens', tokens)
        
        nltk_tagged = nltk.pos_tag(tokens)
        
        for i, t in enumerate(nltk_tagged):
            word = t[0]
            tag = nltk_pos_tagger(t[1])
            #print(word, tag)
            if subj == None:
                obj1, obj2, sr = filter_no_subj(word, tag, obj1, obj2, sr)
                        
            else:
                obj1, obj2, sr = filter_with_subj(word, tag, obj1, obj2, sr)               
        
        if subj != None and len(sr) > 0:
            obj1 = obj1 * len(obj2)
            last_sr = sr[-1]
            for i in range(len(obj2) - len(sr)):
                sr += [last_sr]
        else:
            obj1 = obj1[0:len(sr)]
            obj2 = obj2[0:len(sr)]
                    
        #assert len(obj1) == len(obj2) == len(sr), print(obj1, obj2, sr)
        
        for i in range(len(obj1)):
            if i < len(sr) and i < len(obj2):
                triplets += [[obj1[i], sr[i], obj2[i]]]

    
    except Exception as e:
        traceback.print_exc()
        pass
    
    return triplets 