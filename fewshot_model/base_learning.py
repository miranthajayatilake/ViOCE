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
# print("PyTorch Version: ",torch.__version__)
import random
from sklearn.preprocessing import minmax_scale
from pytorch_metric_learning import miners, losses
import argparse



def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 500.0
    
    #Selecting vectors for the label indices
    class_indices_dict = image_datasets['train'].class_to_idx
    
    embeddings = {}
    with open(embeddings_file, "r") as file:
        for line in file:
            vector_ = line.split(' ')[1:]
            vector_f = [float(i) for i in vector_]
            if line.split(' ')[0].split(':')[1].split('.')[0] in _classes.keys():
                embeddings[_classes[line.split(' ')[0].split(':')[1].split('.')[0]]] = vector_f

    print(len(embeddings))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        count_goesin_all = 0
        count_batch_all = 0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
#                 for i in label_vectors_neg:
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

                        criterion_return = criterion(outputs, label_vectors, label_vectors_neg)
                        loss = criterion_return['loss']
                        count_goesin = criterion_return['count_goesin']
                        count_batch = criterion_return['count_batch']


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = epoch_loss


            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


            # deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("model saved with loss - %s" %(best_acc))
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
            
            scheduler.step()    

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True    



def initialize_model(model_name, num_classes, feature_extract):

    model_ft = None
    input_size = 0

    # Resrnet is used in our experiments, hence the top layers are modified
    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
                         nn.Linear(in_features=num_ftrs, out_features=2048, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(2048),
                         nn.Linear(in_features=2048, out_features=1024, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(1024),
                         nn.Linear(in_features=1024, out_features=512, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(512),
                         nn.Linear(in_features=512, out_features=512, bias=True),
                         nn.ReLU(),
                         nn.BatchNorm1d(512),
                         nn.Linear(in_features=512, out_features=num_classes, bias=True))
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size



# Exteding the Imagefolder loader function to output vectors
class Imagefolder_for_vectors(datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None):
        super(Imagefolder_for_vectors, self).__init__(root = root, transform=transform, target_transform=target_transform)


    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target




def criterion(outputs, labels, labels_neg):
    all_value_p = []
    all_value_n = []

    all_outputs = []
    all_labels = []
    values = []
    sum_value = 0
    
    count_batch = 0
    count_goesin = 0
    
    for output, label, label_neg in zip(outputs, labels, labels_neg):
        
        count_batch += 1
        
        value_p = torch.norm(output-label[:-1]) - 0.2 * torch.abs(label[-1:])

        if torch.norm(output-label[:-1]) - torch.abs(label[-1:]) < 0:
            print('Goes in!!')
            count_goesin += 1
        
        value_p = torch.max(torch.zeros_like(value_p), value_p)
        all_value_p.append(value_p)
        
        
        #adding the negative labels
        value_n = torch.abs(label_neg[-1:]) - torch.norm(output-label_neg[:-1])

        value_n = torch.max(torch.zeros_like(value_n), value_n)
        all_value_n.append(value_n)

        value = value_p + value_n

        sum_value = sum_value + value
    
        all_outputs.append(output.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())


    return {'loss': sum_value, 'count_goesin': count_goesin ,'count_batch': count_batch}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Passing the required paths')
    parser.add_argument('--d', metavar='path', required=True,
                        help='image data path')
    parser.add_argument('--ef', metavar='path', required=True,
                        help='embeddings file path')
    parser.add_argument('--cf', metavar='path', required=True,
                        help='classnames file path')
    args = parser.parse_args()

    data_dir = args.d

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    # Number of classes in the dataset
    num_classes = 300  # NOTE: THis is set to the embedding size, because that is our output

    # Batch size for training (change depending on how much memory you have)
    batch_size = 64

    # Number of epochs to train for
    num_epochs = 50

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # file containing the computed concept embeddings
    embeddings_file = args.ef



    _classes = {}
    _class_ids_and_names = []
    with open(args.cf, 'r') as f:
        _class_ids_and_names.append(f.readlines())

    # Include class ids and names in a dictionary
    for l in _class_ids_and_names[0]:
        if not l.startswith('#'):
            _classes[l.split(' ')[1].split('\n')[0].lower()] = l.split(' ')[0].split('n')[1]
             
                
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract)

    # Print the model we just instantiated
    print(model_ft)

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

    print("Initializing Datasets and Dataloaders...")

    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    # Setup learning rate decay
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    # Train and evaluate
    model_ft_after_training, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs=num_epochs)


    # Saving the final few-shot model
    PATH = 'trained_vision_model.pth'
    torch.save(model_ft_after_training, PATH)
