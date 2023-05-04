import torch.nn as nn
from torchvision import models
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
import sys

@dataclass
class MODELCONFIG:
    """_summary_
    This class provides the configuration for building a model
    """

    LR:float = 0.001
    MOMENTUM:float = 0.9
    STEP_SIZE:float = 0.7
    GAMMA:float = 0.1


class BUILD_MODEL:

    """_summary_
        This class provides the configuration for initializing a model, optimizaters and loss functions 
    """

    def __init__(self):
        self.model_config = MODELCONFIG()

    def init_model(self):
        """_summary_
            This function initializes the different types of model and returns a dictionary
        """
        logging.info(f"Initializing the model dictionary")

        # model = models.resnet18(weights='IMAGENET1K_V1')
        try:
            model_list: dict = {
                        "EFFICIENTNET" : models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1'),
                        "RESNET" : models.resnet18(pretrained=True),
                        "GOOGLENET" : models.googlenet(pretrained=True)}
            
        except Exception as e:
            CustomException(e,sys)

        return model_list
    
    def init_criterion(self):
        """_summary_
            This function initializes the loss function
        """
        logging.info(f"Initializing the loss function")

        return nn.CrossEntropyLoss()

    def init_optimizer(self, model):
        """_summary_
            This function initializes the optimizer
        """
        logging.info(f"Initializing the optimizer")
    
        return optim.SGD(model.fc.parameters(), lr= self.model_config.LR, momentum=self.model_config.MOMENTUM)
    
    def init_scheduler(self, optimizer):
        """_summary_
            This function initializes the scheduler
        """
        logging.info(f"Initializing the scheduler")

        return lr_scheduler.StepLR(optimizer, step_size=self.model_config.STEP_SIZE, gamma=self.model_config.GAMMA)


if __name__ == '__main__':
    from data_ingestion import DataIngestion
    from model_trainer import Model_Training

    model = BUILD_MODEL().init_model()   
    dataloader,dataset_sizes ,num_classes = DataIngestion().initiate_data_ingestion()
    trainer = Model_Training()
    
    for name,model in model.items():

        for params in model.parameters():
            params.requires_grad = True
        
        # Change the final classification head.
        if name == 'EFFICIENTNET':
            num_ftrs = model.classifier[1].in_features
        else:
            num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, out_features=len(num_classes))

        criterion = BUILD_MODEL().init_criterion()
        optimizer = BUILD_MODEL().init_optimizer(model=model)
        exp_lr_scheduler = BUILD_MODEL().init_scheduler(optimizer)

        print(name + "--------- training --------")

        trainer.train_model(name=name,model=model, criterion=criterion, 
                                           optimizer=optimizer,scheduler=exp_lr_scheduler,
                                           dataloader=dataloader,dataset_sizes=dataset_sizes)
        
