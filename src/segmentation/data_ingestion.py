import os
import sys
from src.dirs import dirs
import glob

from src.classification.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from torch.utils.data  import DataLoader
from torchvision import datasets
import torch
from torch.utils.data  import Dataset, DataLoader
from transformations import transforms_training, transforms_validation, pre_transforms
from skimage.io import imread
from tqdm import tqdm


@dataclass
class DataIngestionConfig:
    train_data_path:str = dirs['segmentation_image_path']
    val_data_path:str = dirs['segmentation_mask_path']
    test_data_path:str = dirs['test_data_path']
    random_seed = 42
    train_size = 0.8  # 80:20 split
    BATCH_SIZE:int = 16
    TEST_BATCH_SIZE:int = 1
    NUM_WORKERS:int = 2

class SegmentationDataset(Dataset):
    def __init__(self,inputs,targets,
                 transform= None,
                 use_cache = False,
                 pre_transform = None):
        self.inputs = inputs
        self.targets =  targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching data')
            for i, img_name, tar_name in zip(progressbar ,self.inputs, self.targets):
                img, tar = imread(str(img_name)), imread(str(tar_name))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)
                
                self.cached_data.append((img, tar))  

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index: int):
        
        #use cached data
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            input_id = self.inputs[index]
            targets_id = self.targets[index]

            #Load input and targets
            x,y = imread(str(input_id)), imread(str(targets_id))

        #processing
        if self.transform is not None:
            x,y = self.transform(x,y)
        
        #Typescaping
        x,y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
    
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
        data_loader= {}
        dataset_sizes = {}
       
        try:
            logging.info("Datset initialization")
            logging.info("Datset train_data_path split")
            inputs_train, inputs_valid = train_test_split(glob.glob(os.path.join(self.ingestion_config.train_data_path,'*.png')),
                random_state= self.ingestion_config.random_seed,
                train_size=self.ingestion_config.train_size,
                shuffle=True)
            logging.info("Datset val_data_path split")
            targets_train, targets_valid = train_test_split(glob.glob(os.path.join(self.ingestion_config.val_data_path,'*.png')),
                random_state= self.ingestion_config.random_seed,
                train_size=self.ingestion_config.train_size,
                shuffle=True)

            dataset_train = SegmentationDataset(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache=True,
                                    pre_transform=pre_transforms
                                    )
            
            dataset_valid = SegmentationDataset(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache=True,
                                    pre_transform=pre_transforms)            
 
            logging.info("Dataset creation complete")

            """
            Prepares the training and validation data loaders.
            :param dataset_train: The training dataset.
            :param dataset_valid: The validation dataset.
            Returns the training and validation data loaders.
            """
            logging.info("Dataloader initialization")
            
            dataloader_training = DataLoader(dataset=dataset_train,batch_size=self.ingestion_config.BATCH_SIZE,
                                             shuffle=True, num_workers=self.ingestion_config.NUM_WORKERS)
            
            dataloader_validation = DataLoader(dataset=dataset_valid,batch_size=self.ingestion_config.BATCH_SIZE,
                                             shuffle=True, num_workers=self.ingestion_config.NUM_WORKERS)

            logging.info("Dataloader complete")

            data_loader['train'], data_loader['val']= dataloader_training, dataloader_validation
            dataset_sizes['train'], dataset_sizes['val']= len(dataset_train), len(dataset_valid)

            return data_loader, dataset_sizes 

        except Exception as e:
            CustomException(e,sys)

    def initiate_test_data_ingestion(self):
        logging.info("Entering Data Ingestion part")

        """
        Function to prepare the Datasets.
        Returns the training and validation datasets along 
        with the class names.
        """
     
        try:
            logging.info("Datset initialization")
            dataset_test = datasets.ImageFolder(
                self.ingestion_config.test_data_path, 
                transform=self.valid_transform
                )
            
            logging.info("Dataset creation complete")

            logging.info("Dataloader initialization")

            test_loader = DataLoader(
                dataset_test, batch_size=self.ingestion_config.TEST_BATCH_SIZE, 
                shuffle=False, num_workers=self.ingestion_config.NUM_WORKERS)

            logging.info("Dataloader complete")

            return test_loader 

        except Exception as e:
            CustomException(e,sys)

if __name__=='__main__':
    training_dataloader, dataset_sizes  = DataIngestion().initiate_data_ingestion()

    x,y = next(iter(training_dataloader['train']))

    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

