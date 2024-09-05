from Model import TextClassification
from .HelperFunctions.Training import trainModel, testModel
from .HelperFunctions.VisualizationFunctions import createConfusionMatrix
from torchtext import datasets
from torch.utils.data import Dataset, DataLoader
import torch

#find a dataset I want to use from this: https://pytorch.org/text/stable/datasets.html
#dataset: not yet determined

device = "cuda" if torch.cuda.is_available else "cpu"

#import the dataset here





#for this file, I'll be training the files, testing it, and making a confusion matrix to showcase
#the results of the model

