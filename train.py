'''---------- IMPORTS ----------'''

from __future__ import print_function, division

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

import argparse
import copy
import matplotlib.pyplot as plt
import os
import time

plt.ion() # interactive mode 

'''---------- PARSE COMMAND LINE ARGUMENTS ----------'''

# The argument parser will obtain arguments from the command line

parser = argparse.ArgumentParser(description = 'AlexNet Training with ImageNet')

# Checks for the '--data' flag
# This flag directs to the location of the Tiny ImageNet dataset

parser.add_argument("--data", 
        metavar = "DIR", 
        default = "tiny_imagenet/",
        help = "path to Tiny ImageNet dataset")

# Checks for the '--save' flag
# This flag directs to the save location of the trained model

parser.add_argument("--save", 
        metavar = "DIR", 
        default = "trained_model/",
        help = "path where the trained model will be saved")

args = parser.parse_args()


'''---------- ALEX NET MODEL ----------'''

class AlexNet(nn.Module):

    '''
    Model definition inspired from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

    The only difference is the last layer: Feature map goes to 200 classes.
    '''

    # Defines the model

    def __init__(self):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 200), # Difference
        )
    # End of __init__()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    # End of forward()
# End of AlexNet


'''--------- MODEL TRAINING ----------'''

'''
This section implements Transfer Learning; inspired by:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
&
https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

# Organises the validation data found in the Tiny ImageNet dataset

def organise_validation():
    # This leads to the validation folder

    validation_path = os.path.join(args.data, "val/")

    # This leads to the validation images

    validation_images_path = os.path.join(args.data, "val/images/")

    # This leads to a text file with labels annotating the validation data

    validation_annotations_path = os.path.join(args.data, 
                                               "val/val_annotations.txt")

    # Opens the file with the validation labels in read mode

    validation_annotations_file = open(validation_annotations_path, "r")

    # Reads the lines in the validation labels file

    validation_annotations = validation_annotations_file.readlines()

    # The validation_annotiations file has the following format
    # <image_name> <classification> {<bounding_box> x 4}
    # We only care about the image_name and classification for this task

    # Create a dictionary that will hold the image_names and classifications
    # The image_names (unique) will be used as keys
    # The image classifications (not unique) will be used as values

    validation_dictionary = {}

    for line in validation_annotations:
        tokens = line.split("\t")

        image_name = tokens[0]
        image_classification = tokens[1]
        
        validation_dictionary[image_name] = image_classification

    # Now, we can create a folder for each classification and
    # add the corresponding images to those folders

    # This can be done by iterating through the dictionary and checking
    # whether each image's classification is already a folder
    # If so, add the image to the folder
    # If not, create the folder then add the image to the folder

    for image, classification in validation_dictionary.items():
        # Defines the new folder path

        folder_path = os.path.join(validation_images_path, classification)

        # Check if the folder does not exist

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if the image exists in validation_images_path (default)
        # If it does, rename its path so it exists in the new folder

        if os.path.exists(os.path.join(validation_images_path, image)):
            os.rename(os.path.join(validation_images_path, image),
                      os.path.join(folder_path, image))

    # Clean Up

    validation_annotations_file.close()
# End of organise_validation()


########## MAIN ##########

def main():
    # Defines the batch sizes
    # Each class has 500 training images
    # Each class has 50 validation images

    TRAIN_BATCH_SIZE = 50
    VAL_BATCH_SIZE = 50
    
    # Defines the number of epochs

    EPOCHS = 4

    # The tiny-imagenet training data is organised into folders
    # based on its classification.
    # The validation data is not organised, which needs to be done

    organise_validation()

    ####################################
    ### LOADS TINY IMAGENET DATASETS ###
    ####################################

    # Data augmentation and normalisation for training
    # Just normalisation for validation

    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])
            ]),
        
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])
            ]),
        }

    # Apply appropriate agumentation and normalisation 
    # to the training and validation datasets

    image_datasets = {}
    
    image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data, "train"), 
                                                            data_transforms['train'])

    image_datasets['val'] = datasets.ImageFolder(os.path.join(args.data, "val/images"),
                                                            data_transforms['val'])

    #image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x),
    #                                          data_transforms[x])
    #                  for x in ['train', 'val']}

    # Define the batch sizes and whether shuffling is needed

    batch_sizes = {}
    batch_sizes['train'] = TRAIN_BATCH_SIZE
    batch_sizes['val'] = VAL_BATCH_SIZE

#   batch_sizes = {x: [TRAIN_BATCH_SIZE, VAL_BATCH_SIZE]
#                   for x in ['train', 'val']}


    shuffling_needed ={}
    shuffling_needed['train'] = True
    shuffling_needed['val'] = False

#   shuffling_needed = {x : [True, False] for x in ['train', 'val']}

    # Creates the loaders

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                        batch_size = batch_sizes[x],
                                        shuffle = shuffling_needed[x],
                                        num_workers = 4)
                   for x in ['train', 'val']}

    # Defines the dataset sizes

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    #############################################
    ### INITIALISES PRE-TRAINED ALEXNET MODEL ###
    #############################################

    # Retrieves the pretrained model

    pretrained_alexnet = models.alexnet(pretrained = True)

    # Initialises the AlexNet model defined above

    new_alexnet = AlexNet()

    # Transfers the weights of the pretrained model to the new model
    # for every layer except the last layer, which sets feature maps to go
    # from 4096 to 200, instead of 1000.

    new_alexnet.features = pretrained_alexnet.features

    new_alexnet.classifier[1].weight = pretrained_alexnet.classifier[1].weight
    new_alexnet.classifier[1].bias = pretrained_alexnet.classifier[1].bias

    new_alexnet.classifier[4].weight = pretrained_alexnet.classifier[4].weight
    new_alexnet.classifier[4].bias = pretrained_alexnet.classifier[4].bias

    # Freeze all the weights of the new model, except for the last layer
    # This will allow for gradient calculations for the last layer

    # Sets all layers to false, in case some weren't frozen already

    for weight in new_alexnet.parameters():
        weight.requires_grad = False

    # Sets last layer to true

    for weight in new_alexnet.classifier[6].parameters():
        weight.requires_grad = True

    ##########################
    ### TRAINING THE MODEL ###
    ##########################

    # Defines the loss function

    loss_function = nn.CrossEntropyLoss()

    # Defines the optimiser used to update parameters of only the last layer

    optimiser = optim.SGD(new_alexnet.classifier[6].parameters(), lr = 0.001, momentum = 0.9)

    # Decay the learning rate by a factor of 0.1 every 7 seconds

    exp_lr_scheduler = lr_scheduler.StepLR(optimiser, step_size = 7, gamma = 0.1)

    # Start time of model training

    print("\nTRAINING MODEL ... \n")

    new_alexnet.train()

    start_time_epoch = time.time()

    for epoch in range(EPOCHS):

        training_loss = 0

        correct_predictions = 0

        start_time_batch = time.time()

        for batch_i, (data, target) in enumerate(dataloaders['train']):

            # Wraps the inputs & labels as variables

            data = Variable(data)
            target = Variable(target, requires_grad = False)

            # Clears the gradient of optimised torch tensors

            optimiser.zero_grad()

            # Forward propagation

            output = new_alexnet(data)

            # Computes the loss

            batch_loss = loss_function(output, target)

            # Adds the batch's loss to the overall training loss

            training_loss += batch_loss.item()

            # Backward propagation

            batch_loss.backward()

            # Update parameters

            optimiser.step()

            # Checks for correct predictions in the model

            value, index = torch.max(output.data, 1)

            for i in range(TRAIN_BATCH_SIZE):
                if (index[i] == target.data[i]):
                    correct_predictions += 1

            # Batch-specific statistics

            end_time_batch = time.time() - start_time_batch

            # Prints statistics every 50 runs

            if ((batch_i % 50) == 0):
                print("\n----------")
                print("Epoch #" + str(epoch))
                print("Batch #" + str(batch_i))
                print("Batch Loss:", batch_loss.item())
                print("Trained on", ((batch_i + 1) * TRAIN_BATCH_SIZE), "images.")
                print("Time Passed: " + str(end_time_batch) + " seconds.")
                print("----------\n")
            

        # Overall training statistics

        training_accuracy = (correct_predictions / dataset_sizes['train']) * 100
        average_training_loss = training_loss / (dataset_sizes['train'] / TRAIN_BATCH_SIZE)
        end_time_epoch = time.time() - start_time_epoch

        print("\n##########")
        print("Training Accuracy: " + str(training_accuracy) + "%")
        print("Average Training Loss: " + str(average_training_loss))
        print("Total Time Taken: " + str(end_time_epoch) + " seconds.")

    print("\nMODEL TRAINED.\n")

    ######################
    ### VALIDATE MODEL ###
    ######################

    print("\nVALIDATING MODEL...\n")

    new_alexnet.eval()

    correct_classifications = 0
    validation_loss = 0

    start_time_val = time.time()

    for batch_i, (data, target) in enumerate(dataloaders['val']):
        
        data = Variable(data)
        target = Variable(target)

        optimiser.zero_grad()

        output = new_alexnet(data)

        batch_loss = loss_function(output, target)
        validation_loss += batch_loss.item()

        value, index = torch.max(output.data, 1)

        for i in range(VAL_BATCH_SIZE):
            if(index[i] == target.data[i]):
                correct_classifications += 1

        # Batch-specific statistics

        end_time_batch = time.time() - start_time_val

        # Prints statistics every 25 runs

        if ((batch_i % 25) == 0):
            print("\n----------")
            print("Batch #" + str(batch_i))
            print("Batch Loss:", batch_loss.item())
            print("Validated on", ((batch_i + 1) * TRAIN_BATCH_SIZE), "images.")
            print("Time Passed: " + str(end_time_batch) + " seconds.")
            print("----------\n")
        


    validation_accuracy = (correct_classifications / dataset_sizes['val']) * 100
    average_validation_loss = validation_loss / (dataset_sizes['val'] / VAL_BATCH_SIZE)

    print("\n########## NETWORK STATISTICS ##########\n")
    print("Validation Accuracy: " + str(validation_accuracy) +"%")
    print("Average Validation Loss: " + str(average_validation_loss))

    print("\nMODEL VALIDATED\n")

    ##################
    ### SAVE MODEL ###
    ##################

    # Check if the save directory exists

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    save_file = os.path.join(args.save, "model.dat")
    torch.save(new_alexnet, save_file)

# End of main()

#main()
