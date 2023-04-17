import os
import sys

from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import torch
from skimage.io import imread
from torch.utils.data  import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from data_transformation import DataTransformation, get_train_transform,get_valid_transform
from src.utils import imshow


@dataclass
class DataIngestionConfig:
    train_data_path:str = "/home/krish/Documents/PyTorch/End2End_Deep_learning_project_using_segmentation@classification/car_data//train"
    test_data_path:str = "/home/krish/Documents/PyTorch/End2End_Deep_learning_project_using_segmentation@classification/car_data//test"
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

            return train_loader, valid_loader, dataset_train.classes

        except Exception as e:
            CustomException(e,sys)
