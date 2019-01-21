import os
import torch
import numpy as np
import pandas as pd
from shutil import copyfile, rmtree
from sklearn.model_selection import  train_test_split
from torchvision import transforms, datasets
from torchvision.models.resnet import resnet18

def get_params(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def make_dataset(path, train_size=0.6, batch_size=20):
    '''Create train, validation and test data generators with augmentation
    
    Args:
        path: absolute path to folder, that contains 'clock' and 'crocodile' 
              folders with images
        
    '''
    
    #split data on train, test and val
    image_clock = os.listdir(path + '/clock')
    image_crocodile = os.listdir(path + '/crocodile')

    data_im = []
    for im_name in image_clock:
        data_im.append({'image':'clock/' + im_name, 'target':0})

    for im_name in image_crocodile:
        data_im.append({'image':'crocodile/' + im_name, 'target':1})

    data = pd.DataFrame(data_im)

    X = data.image
    y = data.target.values
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=(1-train_size), shuffle=True, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_, y_, test_size=0.5, shuffle=True, stratify=y_)

    #create folders for train, test, validation
    for dir_ in ['train', 'test', 'val']:
        if dir_ in os.listdir():
            rmtree(dir_)

    os.mkdir('train')
    os.mkdir('test')
    os.mkdir('val')

    os.mkdir('train/clock')
    os.mkdir('test/clock')
    os.mkdir('val/clock')

    os.mkdir('train/crocodile')
    os.mkdir('test/crocodile')
    os.mkdir('val/crocodile')

    for file in X_train:
        copyfile(path+file, 'train/' + file)
    for file in X_test:
        copyfile(path+file, 'test/' + file)
    for file in X_val:
        copyfile(path+file, 'val/' + file)
    
    #data augmentation
    dataset = datasets.ImageFolder(root='train/', transform=transforms.ToTensor())
    means, std = get_params(dataset)

    transform_augment = transforms.Compose([
        transforms.RandomRotation([-60, 60]),
        transforms.RandomHorizontalFlip(),
       # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(means, std),])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, std),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, std),])

    train_set = datasets.ImageFolder(root='train/', transform=transform_augment)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    val_set = datasets.ImageFolder(root='val/', transform=transform_val)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

    test_set = datasets.ImageFolder(root='test/', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_loader, val_loader, test_loader