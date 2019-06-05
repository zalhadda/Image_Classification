'''---------- IMPORTS ----------'''

import matplotlib.pyplot as plt

from train import AlexNet

import torch
from torch.autograd import Variable

from torchvision import datasets, transforms

import argparse
import cv2
import os
import random
import sys


'''---------- PARSE COMMAND LINE ARGUMENTS ----------'''

# The argument parses will obtain arguments from the command line

parser = argparse.ArgumentParser(description = "AlexNet Testing")

# Checks the '-model' flag
# This flag directs to the location of the saved pretrained model

parser.add_argument("--model",
        metavar = "DIR",
        default = "trained_model/",
        help = "path to saved trained model")

args = parser.parse_args()


'''---------- MODEL TESTING ----------'''

TEST_CAM = 0

############
### MAIN ###
############

def main():

    # Load saved model

    model_file = os.path.join(args.model, "model.dat")
    model = torch.load(model_file)
    model.eval()

    # Open the camera

    cam = cv2.VideoCapture(0)

    # Causes a continuous feed

    while(True):

        ret = None
        frame = None

        if TEST_CAM:
            # Tests to check camera is still open

            if not cam.isOpened():
                print("Error: No camera detected.")
                print("Exiting...")
                return -1

            # Reads from camera

            ret, frame = cam.read()
        else:
            test_images_path = "tiny_imagenet/test/images"
            frame = random.choice([
                        x for x in os.listdir(test_images_path)
                        if os.path.isfile(os.path.join(test_images_path, x))])

            #frame = "Zer0.jpg" # This is my friend's pet guinea pig
            #frame = "Catherine's_Dog.jpg"  # This is my friend's dog
            #frame = "Erica.jpg" # This is my friend
            #frame = "Zaid.jpg" # This is me
            #frame = "Omair_In_Blanket.jpg" # This is another friend, wrapped in a blanket
            #frame = "Tori_with_mug.jpg" # Another friend
            #frame = "taxo.jpg" # Everyone wants to be classified now
            #frame = "paige_banana.jpg" # my friend sent a banana
            #frame = "rae.jpg" # friend
            #frame = "wet_zer0.jpg" # the guinea pig is wet now

            print(frame)

            frame = cv2.imread(os.path.join(test_images_path, frame), cv2.IMREAD_COLOR)
    
        # Normalise the frame

        normalise = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                             std = [0.229, 0.224, 0.225])
                        ])

        normalised_frame = normalise(frame)

        # Creates a 4-dimensional weight for the image

        image_tensor = torch.Tensor(1, 3, 224, 224)
        image_tensor[0] = normalised_frame

        # Retrieves the image's label from the pretrained model

        output = model(Variable(image_tensor))
        
        value, label = torch.max(output.data, 1)
        label = label.item()

        # Converts the label to the classification

        training_path = "tiny_imagenet/train/"

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
            ])

        image_datasets = {}
        image_datasets['train'] = datasets.ImageFolder(training_path, train_transform)

        class_names = image_datasets['train'].classes
        num_classes = len(class_names)

#        print(class_names)
 #       print(num_classes)

        labels_path= "tiny_imagenet/words.txt"
        labels_file = open(labels_path, "r")
        labels_lines = labels_file.readlines()

        classes = {}

        for line in labels_lines:
            words = line.split('\t')
            classes[words[0]] = words[1]

        labels_file.close()

        #print(classes)
        #print(label)

        classification = classes[class_names[label]]
        #classification = label

        # Shows the image with the label

        if TEST_CAM:
            # Puts the label on the image

            cv2.putText = (frame,                   # image
                        str(classification),        # text
                        (10, 120),                  # org
                        cv2.FONT_HERSHEY_SIMPLEX,   # font
                        2,                          # fontScale
                        cv2.LINE_AA)                # lineType

            cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Image", frame)
        else:
            image_classification = "Classification: " + str(classification)

            plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
            plt.text(0.05, 0.90, image_classification, fontsize=10, transform=plt.gcf().transFigure)
            #plt.annotate(image_classification, xy = (2,2))
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("Image label:", classification)

            input("Press enter to show the next image...")

            plt.close()


    # Close camera

    cam.release()
    cv2.destroyAllWindows()
# End of main()

main()
