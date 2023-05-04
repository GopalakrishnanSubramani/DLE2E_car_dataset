import os
import sys
from src.components.data_transformation import DataTransformation

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import torch
from skimage.io import imread
from torch.utils.data  import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms, utils



@dataclass
class DataIngestionConfig:
    train_data_path:str = "/home/krish/Documents/PyTorch/End2End_Deep_learning_project_using_segmentation@classification/car_data/train"
    test_data_path:str = "/home/krish/Documents/PyTorch/End2End_Deep_learning_project_using_segmentation@classification/car_data/val"
    BATCH_SIZE:int = 16
    NUM_WORKERS:int = 2


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.train_transform = DataTransformation().get_train_transform()
        self.valid_transform = DataTransformation().get_valid_transform()

    def initiate_data_ingestion(self):
        logging.info("Entering Data Ingestion part")

        """
        Function to prepare the Datasets.
        Returns the training and validation datasets along 
        with the class names.
        """
        data_loader= {}
        dataset_sizes = {}
       
        try:
            logging.info("Datset initialization")
            dataset_train = datasets.ImageFolder(
                self.ingestion_config.train_data_path, 
                transform=self.train_transform
                )
            dataset_valid = datasets.ImageFolder(
                self.ingestion_config.test_data_path, 
                transform=self.valid_transform
            )
            logging.info("Dataset creation complete")

            """
            Prepares the training and validation data loaders.
            :param dataset_train: The training dataset.
            :param dataset_valid: The validation dataset.
            Returns the training and validation data loaders.
            """
            logging.info("Dataloader initialization")
            train_loader = DataLoader(
                dataset_train, batch_size=self.ingestion_config.BATCH_SIZE, 
                shuffle=True, num_workers=self.ingestion_config.NUM_WORKERS)
            
            valid_loader = DataLoader(
                dataset_valid, batch_size=self.ingestion_config.BATCH_SIZE, 
                shuffle=False, num_workers=self.ingestion_config.NUM_WORKERS)

            logging.info("Dataloader complete")

            data_loader['train'], data_loader['val']= train_loader, valid_loader
            dataset_sizes['train'], dataset_sizes['val']= len(dataset_train), len(dataset_valid)
            classes = dataset_train.classes

            return data_loader, dataset_sizes,classes 


        except Exception as e:
            CustomException(e,sys)

if __name__=='__main__':
    dataloader,dataset_sizes ,num_classes = DataIngestion().initiate_data_ingestion()
    # inputs, classes = next(iter(dataloader['val']))
    # out = utils.make_grid(inputs)
    # imshow(out, title=num_classes)
    
    print((dataset_sizes['train']))
    print((dataset_sizes['val']))
    print((num_classes))