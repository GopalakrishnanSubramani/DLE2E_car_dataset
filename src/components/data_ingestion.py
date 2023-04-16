import os
import sys

from DLE2E_car_dataset.src import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import torch
from skimage.io import imread
from torch.utils.data  import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from data_transformation import get_train_transform, get_valid_transform

@dataclass
class DataIngestionConfig:
    train_data_path:str = "car_data/train"
    test_data_path:str = "car_data/test"
    IMAGE_SIZE:int = 224
    BATCH_SIZE:int = 16
    NUM_WORKERS:int = 2


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entering Data Ingestion part")

        """
        Function to prepare the Datasets.
        Returns the training and validation datasets along 
        with the class names.
        """
       
        try:
            logging.info("Datset initialization")
            dataset_train = datasets.ImageFolder(
                DataIngestionConfig().train_data_path, 
                transform=(get_train_transform(DataIngestionConfig().IMAGE_SIZE))
                )
            dataset_valid = datasets.ImageFolder(
                DataIngestionConfig().test_data_path, 
                transform=(get_valid_transform(DataIngestionConfig().IMAGE_SIZE)))
            
            logging.info("Dataset creation complete")

            """
            Prepares the training and validation data loaders.
            :param dataset_train: The training dataset.
            :param dataset_valid: The validation dataset.
            Returns the training and validation data loaders.
            """
            logging.info("Dataloader initialization")
            train_loader = DataLoader(
                dataset_train, batch_size=DataIngestionConfig().BATCH_SIZE, 
                shuffle=True, num_workers=DataIngestionConfig().NUM_WORKERS)
            
            valid_loader = DataLoader(
                dataset_valid, batch_size=DataIngestionConfig().BATCH_SIZE, 
                shuffle=False, num_workers=DataIngestionConfig().NUM_WORKERS)

            logging.info("Dataloader complete")

            return train_loader, valid_loader, dataset_train.classes

        except Exception as e:
            CustomException(e,sys)

if __name__=='__main__':
    class_dict = {}
    train_loader, val_loader, class_names = DataIngestion()
    inputs, classes = next(iter(train_loader))
    # out = utils.make_grid(inputs)
    # imshow(out, title=class_names)
    for idx,name in enumerate(class_names):
        class_dict[idx]=name
    
    print(class_dict)