#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import random
import os.path
import shutil, random
from math import sqrt
from collections import OrderedDict
import argparse




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True   


def criterion(outputs, labels, labels_neg):
    sum_value = 0
    
    
    for output, label, label_neg in zip(outputs, labels, labels_neg):
        
        label = label[:-1]
        label_neg = label_neg[:-1]
        
        _value = torch.dot(output, label)
        
        __value = - torch.dot(output, label) + torch.dot(output, label_neg) + torch.tensor(80).to(device)
        
        
        value = torch.max(torch.zeros_like(_value), __value)
        
        sum_value = sum_value + value
    return sum_value



def train_few_shot(model, dataloaders, criterion, optimizer, image_datasets_few, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 500
    
    #Selecting vectors for the label indices
    class_indices_dict = image_datasets_few['train'].class_to_idx
    
    embeddings = OrderedDict()
    with open(embeddings_file, "r") as file:
#         next(file)
        for line in file:
            vector_ = line.split(' ')[1:]
            vector_f = [float(i) for i in vector_]
            if line.split(' ')[0].split(':')[1].split('.')[0] in _classes.keys():
                if 'n' + _classes[line.split(' ')[0].split(':')[1].split('.')[0]] in class_indices_dict.keys():
                    embeddings[_classes[line.split(' ')[0].split(':')[1].split('.')[0]]] = vector_f

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = torch.stack(tuple(inputs)).to(device)
    
                label_vectors = []
        
                ##--> Negative label vectors
                label_vectors_neg = []
            
                #NOTE: Assign the vectors here
                for label in labels:
                    for syn_id, idx in class_indices_dict.items():
                        if label == idx:
                            label_vectors.append(embeddings[syn_id.split('n')[1]])
                            
                    random_label = random.choice([x for x in list(class_indices_dict.values()) if x != label])
                    for syn_id, idx in class_indices_dict.items():
                        if random_label == idx:
                            label_vectors_neg.append(embeddings[syn_id.split('n')[1]])        
                            
                label_vectors = torch.Tensor(label_vectors).to(device)
                label_vectors_neg = torch.Tensor(label_vectors_neg).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)

                        outputs = 2.*(outputs - torch.min(outputs))/np.ptp(outputs.cpu().detach().numpy())-1
    

                        loss = criterion(outputs, label_vectors, label_vectors_neg)


                    preds = []

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
#                 if phase == 'val':
                for i in range(len(preds)):
                    if preds[i] == labels.data[i]:
                        running_corrects += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = epoch_loss
    

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('saved model at loss - %s' %(best_acc))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Passing the required paths')
    parser.add_argument('--d', metavar='path', required=True,
                        help='image data path')
    parser.add_argument('--ef', metavar='path', required=True,
                        help='embeddings file path')
    parser.add_argument('--cf', metavar='path', required=True,
                        help='classnames file path')
    parser.add_argument('--model', metavar='path', required=True,
                        help='trained model path')
    args = parser.parse_args()


    data_dir = args.d

    # file containing the computed concept embeddings
    embeddings_file = args.ef

    # tranied model path after base learning
    TRAINED_MODEL_PATH = args.model

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"


    _classes = {}
    _class_ids_and_names = []
    with open(args.cf, 'r') as f:
        _class_ids_and_names.append(f.readlines())

    # Include class ids and names in a dictionary
    for l in _class_ids_and_names[0]:
        if not l.startswith('#'):
            _classes[l.split(' ')[1].split('\n')[0]] = l.split(' ')[0].split('n')[1]

    no_of_experiments = 10

    for i in range(no_of_experiments):

        PATH = TRAINED_MODEL_PATH
        model_ft_after_training = torch.load(PATH)

        #Finetuning of the top layers
        feature_extract = True
        set_parameter_requires_grad(model_ft_after_training, feature_extract)

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft_after_training = model_ft_after_training.to(device)

        params_to_update = model_ft_after_training.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft_after_training.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft_after_training.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
                
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        

        val_dir = data_dir + '/val/'
        train_dir = data_dir + '/train/'

        selected_classes = []
        for foldername in os.listdir(train_dir):
            selected_classes.append(foldername)

        for class_ in selected_classes:
            filenames = random.sample(os.listdir(os.path.join(train_dir, class_)), 5)
            for fname in filenames:
                srcpath = os.path.join(os.path.join(train_dir, class_), fname)
                shutil.move(srcpath, os.path.join(val_dir, class_))
                
        for class_ in selected_classes:
            filenames = random.sample(os.listdir(os.path.join(val_dir, class_)), 5)
            for fname in filenames:
                srcpath = os.path.join(os.path.join(val_dir, class_), fname)
                shutil.move(srcpath, os.path.join(train_dir, class_))            
        
        
        
        input_size = 224
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


        print("Initializing Few-shot Datasets and Dataloaders...")

        data_dir_fewshot = data_dir
        batch_size_fewshot = 5
        num_epochs_fewshot = 100

        # Create training and validation datasets
        image_datasets_fewshot = {x: datasets.ImageFolder(os.path.join(data_dir_fewshot, x), data_transforms[x]) for x in ['train', 'val']}

        # Create training and validation dataloaders
        dataloaders_dict_fewshot = {x: torch.utils.data.DataLoader(image_datasets_fewshot[x], batch_size=batch_size_fewshot, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        
        ### Training
        model_ft_fewshot, hist_fewshot = train_few_shot(model_ft_after_training, dataloaders_dict_fewshot, criterion, optimizer_ft, image_datasets_fewshot, num_epochs=num_epochs_fewshot, is_inception=(model_name=="inception"))

        
        PATH = 'few_shot_model_%s.pth' % (i)
        torch.save(model_ft_fewshot, PATH)