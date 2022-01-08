#!/usr/bin/env python

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

from tensorflow.python.keras.utils.data_utils import Sequence


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

logging.basicConfig(level=logging.INFO)

relations = False


class ELModel(tf.keras.Model):

    def __init__(self, nb_classes, nb_relations, embedding_size, batch_size, input_losses, classes, class_level, margin=0.01, reg_norm=1, reg_radius=0.5):
        super(ELModel, self).__init__()
        self.nb_classes = nb_classes
        self.nb_relations = nb_relations
        self.margin = margin
        self.reg_norm = reg_norm
        self.batch_size = batch_size
        self.inf = 100.0  # For top radius
        self.input_losses = input_losses
        self.reg_radius = reg_radius
        self.classes = classes
        self.class_level = class_level

        cls_weights = np.random.uniform(
            low=-1, high=1, size=(nb_classes, embedding_size + 1))

        rel_weights = np.random.uniform(
            low=-1, high=1, size=(nb_relations, embedding_size))

        self.cls_embeddings = tf.keras.layers.Embedding(
            nb_classes,
            embedding_size + 1,
            input_length=1,
            weights=[cls_weights, ])
        self.rel_embeddings = tf.keras.layers.Embedding(
            nb_relations,
            embedding_size,
            input_length=1,
            weights=[rel_weights, ])
        print('layers done')
        print(cls_weights)
        print(rel_weights)

    def call(self, input):
        """Run the model."""
        # nf1, nf2, nf3, nf4, dis, top, nf3_neg, nf1_neg = input
        nf1, nf1_neg, dis, cl = input

        loss1 = self.nf1_loss(nf1)

        loss_dis = self.dis_loss(dis)

        loss_r = self.radii_rag(cl)


        loss = loss1 + loss_dis + loss_r

        return loss


    def radii_rag(self, input):
        c = input[:, 0]
        level = input[:, 1]

        c = self.cls_embeddings(c)

        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        level = tf.dtypes.cast(level, dtype=tf.float32)

        res = tf.math.sqrt(level) * self.reg_radius - \
            tf.abs(tf.norm(rc, axis=1))

        res = tf.reshape(res, [-1, 1])
        res = tf.nn.relu(res)

        return res

    def reg(self, x):
        res = tf.abs(tf.norm(x, axis=1) - self.reg_norm)
        res = tf.reshape(res, [-1, 1])
        return res


    def nf1_loss(self, input):
        # print(input)
        c = input[:, 0]
        # print('c - %s' % (c))
        d = input[:, 1]
        # print('checking class index structure')
        # print(c)
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)

        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        # print('radius in nf1 - %s' % (rc))
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        _rc = tf.math.abs(c[:, -1])
        _rd = tf.math.abs(d[:, -1])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        euc = tf.reshape(tf.norm(x1 - x2, axis=1), [-1, 1])
        dst = tf.nn.relu(euc + rc - rd - self.margin)

        return dst + self.reg(x1) + self.reg(x2)
        # return dst

    def nf1_neg_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)

        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        _rc = tf.math.abs(c[:, -1])
        _rd = tf.math.abs(d[:, -1])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        euc = tf.reshape(tf.norm(x1 - x2, axis=1), [-1, 1])

        dst = tf.nn.relu(- euc - rc + rd + self.margin)

        return dst + self.reg(x1) + self.reg(x2)

    def nf2_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        e = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        e = self.cls_embeddings(e)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        re = tf.reshape(tf.math.abs(e[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x3 = e[:, 0:-1]

        x = x2 - x1
        dst = tf.reshape(tf.norm(x, axis=1), [-1, 1])
        dst2 = tf.reshape(tf.norm(x3 - x1, axis=1), [-1, 1])
        dst3 = tf.reshape(tf.norm(x3 - x2, axis=1), [-1, 1])
        rdst = tf.nn.relu(tf.math.minimum(rc, rd) - re)
        dst_loss = (tf.nn.relu(dst - sr - self.margin)
                    + tf.nn.relu(dst2 - rc - self.margin)
                    + tf.nn.relu(dst3 - rd - self.margin)
                    + rdst - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3)

    def nf3_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        x3 = x1 + r

        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        euc = tf.reshape(tf.norm(x3 - x2, axis=1), [-1, 1])
        dst = tf.nn.relu(euc + rc - rd - self.margin)

        return dst + self.reg(x1) + self.reg(x2)

    def nf3_neg_loss(self, input):
        # C subClassOf R some D
        c = input[:, 0]
        r = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        x3 = x1 + r

        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        euc = tf.reshape(tf.norm(x3 - x2, axis=1), [-1, 1])
        dst = -(euc - rc - rd - self.margin)

        return tf.nn.relu(dst) + self.reg(x1) + self.reg(x2)

    def nf4_loss(self, input):
        # R some C subClassOf D
        r = input[:, 0]
        c = input[:, 1]
        d = input[:, 2]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        r = self.rel_embeddings(r)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        # x1 = x1 / tf.reshape(tf.norm(x1, axis=1), [-1, 1])
        # x2 = x2 / tf.reshape(tf.norm(x2, axis=1), [-1, 1])

        # c - r should intersect with d
        x3 = x1 - r
        dst = tf.reshape(tf.norm(x3 - x2, axis=1), [-1, 1])
        dst_loss = tf.nn.relu(dst - sr - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2)

    def dis_loss(self, input):
        c = input[:, 0]
        d = input[:, 1]
        c = self.cls_embeddings(c)
        d = self.cls_embeddings(d)
        rc = tf.reshape(tf.math.abs(c[:, -1]), [-1, 1])
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        _rc = tf.math.abs(c[:, -1])
        _rd = tf.math.abs(d[:, -1])

        sr = rc + rd
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        # x1 = x1 / tf.reshape(tf.norm(x1, axis=1), [-1, 1])
        # x2 = x2 / tf.reshape(tf.norm(x2, axis=1), [-1, 1])

        dst = tf.reshape(tf.norm(x2 - x1, axis=1), [-1, 1])
        # return tf.nn.relu(sr - dst + self.margin) + self.reg(x1) + self.reg(x2) + self.reg_r(rc) + self.reg_r(rd)
        return tf.nn.relu(sr - dst + self.margin) + self.reg(x1) + self.reg(x2)
        # return tf.nn.relu(sr - dst + self.margin)
        # return tf.nn.relu(sr - dst + self.margin)
        # return tf.nn.relu(sr - dst + 10*self.margin)

    def top_loss(self, input):
        d = input[:, 0]
        d = self.cls_embeddings(d)
        rd = tf.reshape(tf.math.abs(d[:, -1]), [-1, 1])
        return tf.math.abs(rd - self.inf)


class MyModelCheckpoint(ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super(ModelCheckpoint, self).__init__()
        self.out_classes_file = kwargs.pop('out_classes_file')
        self.out_relations_file = kwargs.pop('out_relations_file')
        self.monitor = kwargs.pop('monitor')
        self.cls_list = kwargs.pop('cls_list')
        self.rel_list = kwargs.pop('rel_list')
        self.valid_data = kwargs.pop('valid_data')
        self.proteins = kwargs.pop('proteins')
        self.prot_index = list(self.proteins.values())
        self.prot_dict = {v: k for k, v in enumerate(self.prot_index)}

        self.best_rank = 100000

    def on_epoch_end(self, epoch, logs=None):
        # Save embeddings every epoch
        current_loss = logs.get(self.monitor)
        # print(current_loss)
        if math.isnan(current_loss):
            print(current_loss)
            print('NAN loss, stopping training')
            self.model.stop_training = True
            return
        el_model = self.model.layers[-1]
        # print(el_model.cls_embeddings.get_weights())
        # print(el_model.rel_embeddings.get_weights())
        cls_embeddings = el_model.cls_embeddings.get_weights()[0]
        if relations:
            rel_embeddings = el_model.rel_embeddings.get_weights()[0]

        prot_embeds = cls_embeddings[self.prot_index]
        prot_rs = prot_embeds[:, -1].reshape(-1, 1)
        prot_embeds = prot_embeds[:, :-1]

        cls_file = self.out_classes_file + '_test.pkl' + str(epoch)
        if relations:
            rel_file = self.out_relations_file + '_test.pkl' + str(epoch)
        # cls_file = f'{cls_file}_{epoch + 1}.pkl'
        # rel_file = f'{rel_file}_{epoch + 1}.pkl'

        df = pd.DataFrame(
            {'classes': self.cls_list, 'embeddings': list(cls_embeddings)})
        df.to_pickle(cls_file)

        if relations:
            df = pd.DataFrame(
                {'relations': self.rel_list, 'embeddings': list(rel_embeddings)})
            df.to_pickle(rel_file)

        prot_embeds = prot_embeds / \
            np.linalg.norm(prot_embeds, axis=1).reshape(-1, 1)

        mean_rank = 0

        print(f'\n Skipping validation')
        # print(f'\n Validation {epoch + 1} {mean_rank}\n')
        # if mean_rank < self.best_rank:
        #     self.best_rank = mean_rank
        print(f'\n Saving embeddings at epoch - {epoch + 1} {mean_rank}\n')

        cls_file = self.out_classes_file + str(epoch)
        if relations:
            rel_file = self.out_relations_file + str(epoch)
        # Save embeddings of every thousand epochs
        # if (epoch + 1) % 1000 == 0:
        # cls_file = f'{cls_file}_{epoch + 1}.pkl'
        # rel_file = f'{rel_file}_{epoch + 1}.pkl'

        df = pd.DataFrame(
            {'classes': self.cls_list, 'embeddings': list(cls_embeddings)})
        df.to_pickle(cls_file)

        if relations:
            df = pd.DataFrame(
                {'relations': self.rel_list, 'embeddings': list(rel_embeddings)})
            df.to_pickle(rel_file)



# --->>NOTE - modified the generator to give class names out

class Generator(object):
    # class Generator(Sequence):

    def __init__(self, data, batch_size=128, steps=100):
        self.data = data

        # self.classes = classes

        self.batch_size = batch_size
        self.steps = steps
        self.start = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.start = 0

    # def shape(self):
    #     return self.data.shape

    def next(self):
        output = {}
        if self.start < self.steps:

            for k, l in self.data.items():
                if len(l) == 0:
                    # print('empty data for %s' % (k))
                    pass
                else:
                    idx = np.random.choice(
                        self.data[k].shape[0], self.batch_size)
                    output[k] = self.data[k][idx]


            labels = np.zeros((self.batch_size, 1), dtype=np.float32)
            self.start += 1

            return ([output['nf1'], output['nf1_neg'], output['disjoint'], output['cl']], labels)
        else:
            self.reset()


def load_data(filename):
    classes = {}
    relations = {}
    data = {'nf1': [], 'nf1_neg': [], 'nf2': [],
            'nf3': [], 'nf4': [], 'disjoint': []}

    if 'owl:Thing' not in classes:
        classes['owl:Thing'] = len(classes)
    if 'owl:Nothing' not in classes:
        classes['owl:Nothing'] = len(classes)

    # Here we read the OWL file and filter the lines we want
    lines = read_owl_file(filename)

    for line in lines:
        # Ignore SubObjectPropertyOf
        if line.startswith('SubObjectPropertyOf'):
            continue
        # Ignore SubClassOf()
        if line.startswith('SubClassOf'):
            line = line.strip()[11: -1]
        # if line.startswith('DisjointClasses'):
        #     line = line.strip()[16: -1]
        print(line)
        # if not line:
        #     continue
        if line.startswith('ObjectIntersectionOf('):
            # C and D SubClassOf E
            it = line.split(' ')
            c = it[0][21:]
            d = it[1][: -1]
            e = it[2]
            if c not in classes:
                classes[c] = len(classes)
            if d not in classes:
                classes[d] = len(classes)
            if e not in classes:
                classes[e] = len(classes)
            form = 'nf2'
            if e == 'owl:Nothing':
                form = 'disjoint'
            data[form].append((classes[c], classes[d], classes[e]))

        elif line.startswith('ObjectSomeValuesFrom('):
            # R some C SubClassOf D
            it = line.split(' ')
            r = it[0][21:]
            c = it[1][: -1]
            d = it[2]
            if c not in classes:
                classes[c] = len(classes)
            if d not in classes:
                classes[d] = len(classes)
            if r not in relations:
                relations[r] = len(relations)
            data['nf4'].append((relations[r], classes[c], classes[d]))
        elif line.find('ObjectSomeValuesFrom') != -1:
            # C SubClassOf R some D
            it = line.split(' ')
            c = it[0]
            r = it[1][21:]
            d = it[2][: -1]
            if c not in classes:
                classes[c] = len(classes)
            if d not in classes:
                classes[d] = len(classes)
            if r not in relations:
                relations[r] = len(relations)
            data['nf3'].append((classes[c], relations[r], classes[d]))
        elif line.startswith('DisjointClasses('):
            line = line.strip()[16: -1]
            it = line.split(' ')
            if len(it) > 2:
                for i in it:
                    for j in it:
                        if i != j:
                            c = i
                            d = j
                            if c not in classes:
                                classes[c] = len(classes)
                            if d not in classes:
                                classes[d] = len(classes)
                            data['disjoint'].append(
                                (classes[c], classes[d], classes['owl:Nothing']))
            else:
                c = it[0]
                d = it[1]
                if c not in classes:
                    classes[c] = len(classes)
                if d not in classes:
                    classes[d] = len(classes)
                data['disjoint'].append(
                    (classes[c], classes[d], classes['owl:Nothing']))
        else:
            # C SubClassOf D
            it = line.split(' ')
            c = it[0]
            d = it[1]
            if c not in classes:
                classes[c] = len(classes)
            if d not in classes:
                classes[d] = len(classes)
            data['nf1'].append((classes[c], classes[d]))

            data['nf1_neg'].append((classes[d], classes[c]))

    # Check if TOP in classes and insert if it is not there
    # if 'owl:Thing' not in classes:
    #     classes['owl:Thing'] = len(classes)
    # if 'owl:Nothing' not in classes:
    #     classes['owl:Nothing'] = len(classes)

    prot_ids = []
    for k, v in classes.items():
        # if not k.startswith('<http://purl.obolibrary.org/obo/GO_'):
        prot_ids.append(v)
    prot_ids = np.array(prot_ids)



    # Add corrupted triples for nf3
    n_classes = len(classes)
    data['nf3_neg'] = []
    # inter_ind = 0
    # for k, v in relations.items():
    #     if k == '<http://interacts>':
    #         inter_ind = v
    for c, r, d in data['nf3']:
        # if r != inter_ind:
        #     continue
        for _ in range(5):
            data['nf3_neg'].append((c, r, np.random.choice(
                [item for item in prot_ids if item not in [d]])))
            data['nf3_neg'].append(
                (np.random.choice([item for item in prot_ids if item not in [c]]), r, d))

 

    data['nf1'] = np.array(data['nf1'])
    data['nf2'] = np.array(data['nf2'])
    data['nf3'] = np.array(data['nf3'])
    data['nf4'] = np.array(data['nf4'])
    data['disjoint'] = np.array(data['disjoint'])
    data['top'] = np.array([classes['owl:Thing'], ])
    data['nf3_neg'] = np.array(data['nf3_neg'])
    data['nf1_neg'] = np.array(data['nf1_neg'])

    for key, val in data.items():
        index = np.arange(len(data[key]))
        np.random.seed(seed=100)
        np.random.shuffle(index)
        data[key] = val[index]

    # Handle empty relation
    # if len(relations.keys()) == 0:
    #     relations['dummy_relation'] = len(relations)
    return data, classes, relations


def load_valid_data(valid_data_file, classes, relations):
    data = []
    rel = f'<http://interacts>'
    with open(valid_data_file, 'r') as f:
        for line in f:
            it = line.strip().split()
            id1 = f'<http://{it[0]}>'
            id2 = f'<http://{it[1]}>'
            if id1 not in classes or id2 not in classes or rel not in relations:
                continue
            data.append((classes[id1], relations[rel], classes[id2]))
    return data


def read_owl_file(filename):
    lines = []

    with open(filename) as f:
        for line in f:
            if 'SubClassOf' in line:
                lines.append(line)
            elif 'DisjointClasses' in line:
                lines.append(line)
            else:
                pass
    return lines


