#!/usr/bin/env python
# coding: utf-8

import click as ck
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import function
import re
import math
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import (
    Input,
)
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from scipy.stats import rankdata
import os

import click as ck
import numpy as np
import pandas as pd
import logging
import math
import os
from collections import deque
import pandas as pd

from elembeddings import ELModel, load_data, load_valid_data, Generator, MyModelCheckpoint
# print(tf.__version__)

d = 300
# Parameters
batch_size = 1
embedding_size = d
margin = 0.00001

reg_norm = 5
reg_radius = 1

learning_rate = 1e-3
epochs = 100


def prepare_data(DATA_PATH):

    #Load the axioms from owl file
    train_data, classes, relations = load_data(DATA_PATH)


    ###--->> Getting the concept heirarchy levels
    class_level = {}
    with open('labelswsuperclasses.txt', 'r') as f:
        temp_list = []
        for line in f:
            temp_list.append(line.split(' \n')[0].split(' '))
        
        for l in temp_list:
            for cls in l:
                if cls not in class_level.keys():
                    class_level[cls] = l.index(cls) + 1
                    


    ##--> Adiing classes and levels to train_data
    cl = []
    for cls_n, indx in classes.items():
        for cls_n2, level in class_level.items():
            if cls_n.split(':')[1] == cls_n2:
                cl.append([indx, level])
    train_data['cl'] = np.array(cl)            


    classses_list = list(classes.keys())

    relations_list = list(relations.keys())

    #Translating extracted axioms into readable format
    readable_train_data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': [], 'nf3_neg':[], 'top':[], 'nf1_neg':[], 'cl':[]}
    for loss, axioms in train_data.items():
        for a in axioms:
    #         print(a)
            if loss == 'top':
                readable_train_data[loss].append(classses_list[0])
                continue
            if len(a) == 2:
                readable_train_data[loss].append((classses_list[a[0]], classses_list[a[1]]))
            else:
                if loss == 'nf3':
                    readable_train_data[loss].append((classses_list[a[0]], relations_list[a[1]], classses_list[a[2]]))
                elif loss == 'nf3_neg':
                    readable_train_data[loss].append((classses_list[a[0]], relations_list[a[1]], classses_list[a[2]]))
                else:
                    readable_train_data[loss].append((classses_list[a[0]], classses_list[a[1]], classses_list[a[2]]))


    _classes = []
    _class_ids_and_names = []
    with open('miniImageNet_classes.txt', 'r') as f:
        _class_ids_and_names.append(f.readlines())

    # Include class ids and names in a dictionary
    for l in _class_ids_and_names[0]:
        if not l.startswith('#'):
            _classes.append(l.split(' ')[1].split('\n')[0].lower())
            

    readable_train_data_only_mini = []
    for r in readable_train_data['nf1']:
        if r[0].split('.')[0].split(':')[1] in _classes :
            readable_train_data_only_mini.append(r)


    all_nf1_trans = []

    list_nf1_prev = train_data['nf1']
    for _ in range(20):
        nf1_temp = []
        for s1 in list_nf1_prev:
            for s2 in train_data['nf1']:
                if s1[1] == s2[0]:
                    nf1_temp.append([s1[0], s2[1]])
        print(len(nf1_temp)) 
        all_nf1_trans = all_nf1_trans + nf1_temp
        list_nf1_prev = nf1_temp
        

    train_data['nf1'] = np.concatenate([train_data['nf1'], all_nf1_trans])


    classses_list = list(classes.keys())

    relations_list = list(relations.keys())

    #Translating extracted axioms into readable format
    readable_train_data = {'nf1': [], 'nf2': [], 'nf3': [], 'nf4': [], 'disjoint': [], 'nf3_neg':[], 'top':[], 'nf1_neg':[], 'cl': []}
    for loss, axioms in train_data.items():
        for a in axioms:
    #         print(a)
            if loss == 'top':
                readable_train_data[loss].append(classses_list[0])
                continue
            if len(a) == 2:
                readable_train_data[loss].append((classses_list[a[0]], classses_list[a[1]]))
            else:
                if loss == 'nf3':
                    readable_train_data[loss].append((classses_list[a[0]], relations_list[a[1]], classses_list[a[2]]))
                elif loss == 'nf3_neg':
                    readable_train_data[loss].append((classses_list[a[0]], relations_list[a[1]], classses_list[a[2]]))
                else:
                    readable_train_data[loss].append((classses_list[a[0]], classses_list[a[1]], classses_list[a[2]]))




    # Prepare data for training the model
    nb_classes = len(classes)   
    nb_relations = len(relations)

    nb_train_data = 0
    for key, val in train_data.items():
        nb_train_data = max(len(val), nb_train_data)
    train_steps = int(math.ceil(nb_train_data / (1.0 * batch_size)))
    train_generator = Generator(train_data, batch_size, steps=train_steps)

    # id to entity maps
    cls_dict = {v: k for k, v in classes.items()}
    rel_dict = {v: k for k, v in relations.items()}

    cls_list = []
    rel_list = []
    for i in range(nb_classes):
        cls_list.append(cls_dict[i])
    for i in range(nb_relations):
        rel_list.append(rel_dict[i])


def build_model():

    # Input layers for each loss type
    nf1 = Input(shape=(2,), dtype=np.int32)
    nf1_neg = Input(shape=(2,), dtype=np.int32)
    nf2 = Input(shape=(3,), dtype=np.int32)
    nf3 = Input(shape=(3,), dtype=np.int32)
    nf4 = Input(shape=(3,), dtype=np.int32)
    dis = Input(shape=(3,), dtype=np.int32)
    top = Input(shape=(1,), dtype=np.int32)
    nf3_neg = Input(shape=(3,), dtype=np.int32)
    nf1_neg = Input(shape=(2,), dtype=np.int32)
    cl = Input(shape=(2,), dtype=np.int32)

    # Build model
    el_model = ELModel(nb_classes, nb_relations, embedding_size, batch_size, [0], classes, class_level, margin, reg_norm, reg_radius)
    out = el_model([nf1, nf1_neg ,dis, cl])
    model = tf.keras.Model(inputs=[nf1, nf1_neg, dis, cl], outputs=out)
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')



    emb_save_path = 'embeddings/elembeddings_classes_%s' %(str(d))

    # Pandas files to store embeddings
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)
    out_classes_file = emb_save_path+'/_cls_embeddings.pkl'
    out_relations_file = emb_save_path+'/_rel_embeddings.pkl' 

    # ModelCheckpoint which runs at the end of each epoch
    checkpointer = MyModelCheckpoint(
        out_classes_file=out_classes_file,
        out_relations_file=out_relations_file,
        cls_list=cls_list,
        rel_list=rel_list,
        valid_data=valid_data,
        monitor='loss')

    # Start training
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_steps,
        epochs=epochs,
        workers=12,
        callbacks=[checkpointer,])

if __name__ == "__main__":
    prepare_data('data/miniimagenet.owl')
    build_model()