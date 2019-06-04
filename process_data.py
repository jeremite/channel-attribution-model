#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:28:20 2019

@author: Travis
"""
import numpy as np
import random
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from pandas.api.types import is_string_dtype, is_numeric_dtype
import warnings
warnings.filterwarnings("ignore")

def cat(data,field,leng,s=','):
    data[field] = data[field].map(lambda x:x.split(s))
    data[leng] = data[field].map(lambda x:len(x))
    
def most_fre(lt):
    return max(lt, key=lt.count)

def sca(col):
    col = [int(v) for v in col]
    return [(v-min(col))/(max(col)-min(col)) for v in col]

def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
            
def process_data(data,seq_length = 20):
    #data_a = data_att.loc[:,['path_id','path','total_conversions','last_time_lapse','null_conversion']]
    data.dropna(axis=0, how='any', inplace=True)
    #merge target variables
    cat(data,'path',"leng_path",s='>')
    cat(data,'marketing_area',"leng_area",s=',')
    cat(data,'tier',"leng_tier",s=',')
    cat(data,'customer_type',"leng_type",s=',')
    # remove those path with channels less than 3
    data_new = data[(data.leng_path >=3)]
    data_new = data_new.reset_index()
    # leave with the most common value
    data_new.marketing_area = data_new.marketing_area.map(lambda x:most_fre(x))
    data_new.tier = data_new.tier.map(lambda x:most_fre(x))
    data_new.customer_type = data_new.customer_type.map(lambda x:most_fre(x))
    data_new.replace('','NA',inplace=True)

    y = data_new.total_conversions
    
    # got train and test data indices
    idx = [x for x in range(data_new.shape[0])]
    np.random.seed(111)
    random.shuffle(idx)
    tr_idx = idx[0:int(0.9*len(idx))]
    te_idx = idx[int(0.9*len(idx)):]
    
    # got data for time decay
    cat(data_new,'last_time_lapse',"leng_time_lapse",s=',')
    data_new.last_time_lapse=data_new.last_time_lapse.map(lambda x:sca(x))
    pad_sequence = tf.contrib.keras.preprocessing.sequence.pad_sequences
    time_decay =pad_sequence(data_new.last_time_lapse,maxlen=seq_length,padding='pre',truncating='pre',dtype='float32')
    time_decay = time_decay.reshape(-1,20,1)
    time_decay_tr = time_decay[tr_idx]
    time_decay_te = time_decay[te_idx]
    
    # got data for attribution
    text = data_new.path
        # encoding
    t = Tokenizer()
    t.fit_on_texts(text)
        # vocabulary size
    vocab_size = len(t.word_index) + 1
        # integer encode the documents
    encoded_docs = t.texts_to_sequences(text)  
        # padding and truncating path data
    newlines =pad_sequence(encoded_docs,maxlen=seq_length,padding='pre',truncating='post')
    X_train = newlines[tr_idx]
    Y_train = y[tr_idx]
    X_test = newlines[te_idx]
    Y_test = y[te_idx]
    all_X = np.array(list(map(lambda x: to_categorical(x, num_classes=vocab_size),newlines)), ndmin=3)
    X_tr = np.array(list(map(lambda x: to_categorical(x, num_classes=vocab_size), X_train)), ndmin=3)
    X_te = np.array(list(map(lambda x: to_categorical(x, num_classes=vocab_size), X_test)), ndmin=3)
    paths = text[tr_idx].reset_index().path
    # got customer data (control data)
    data_lr = data_new.loc[:,['marketing_area','tier','customer_type']]
    train_cats(data_lr)
    data_lr['c_type'+'_na'] = [1 if v=='NA' else 0 for v in data_lr['customer_type']]
    for col in data_lr.columns:
        if not is_numeric_dtype(data_lr[col]):
            data_lr[col] = data_lr[col].cat.codes+1
    X_tr_lr = data_lr.iloc[tr_idx,:]
    X_te_lr = data_lr.iloc[te_idx,:]        
    scaler = MinMaxScaler()
    scaler.fit(X_tr_lr)
    X_tr_lr[['marketing_area', 'tier','customer_type']] = scaler.fit_transform(X_tr_lr[['marketing_area', 'tier','customer_type']])
    X_te_lr[['marketing_area', 'tier','customer_type']] = scaler.transform(X_te_lr[['marketing_area', 'tier','customer_type']])
    categorical_vars = data_lr.columns[0:3]
    
    return [time_decay_tr,time_decay_te, X_tr,X_te, X_tr_lr, X_te_lr, Y_train, Y_test, 
            all_X,time_decay, newlines, y , categorical_vars, paths]


    
    