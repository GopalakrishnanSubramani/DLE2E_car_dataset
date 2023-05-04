from src.components.data_ingestion import DataIngestion
from src.components.model import BUILD_MODEL
from src.components.model_trainer import Model_Training
import torch.nn as nn
import torch
from dataclasses import dataclass


@dataclass
class TRAINCONFIG:
    model = BUILD_MODEL().init_model()   
    dataloader,dataset_sizes ,num_classes = DataIngestion().initiate_data_ingestion()
    trainer = Model_Training()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TRAINING():
    def __init__(self):
        self.train_config = TRAINCONFIG()

    def training(self):    
        for name,model in self.train_config.model.items():

            for params in model.parameters():
                params.requires_grad = True
            
            # Change the final classification head.
            if name == 'EFFICIENTNET':
                num_ftrs = model.classifier[1].in_features
            else:
                num_ftrs = model.fc.in_features

            model.fc = nn.Linear(num_ftrs, out_features=len(self.train_config.num_classes))
            model = model.to(self.train_config.device)

            criterion = BUILD_MODEL().init_criterion()
            optimizer = BUILD_MODEL().init_optimizer(model=model)
            exp_lr_scheduler = BUILD_MODEL().init_scheduler(optimizer)

            print(name + "--------- training --------")

            self.train_config.trainer.train_model(name=name,model=model, criterion=criterion, 
                                                optimizer=optimizer,scheduler=exp_lr_scheduler,
                                                dataloader=self.train_config.dataloader,dataset_sizes=self.train_config.dataset_sizes)

if __name__ == '__main__':
    training = TRAINING().training()
