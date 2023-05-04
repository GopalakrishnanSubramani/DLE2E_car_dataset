import torch.nn as nn
from torchvision import models
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from data_ingestion import DataIngestion
from model_trainer import Model_Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_model(pretrained=True, fine_tune=False, num_classes=196):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')


    # model = models.resnet18(weights='IMAGENET1K_V1')
    models = {  "RESNET" : models.resnet18(pretrained=pretrained),
                "GOOGLENET" : models.googlenet(pretrained=pretrained),
                "MOBILENET" : models.mobilenet_v3_small(pretrained=pretrained)}

    num_ftrs = model.fc.in_features

    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False

    # Change the final classification head.
    model.fc = nn.Linear(num_ftrs, out_features=num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    return model, criterion,optimizer,exp_lr_scheduler

if __name__ == '__main__':
    model, criterion,optimizer,exp_lr_scheduler = build_model()
    dataloader,dataset_sizes ,num_classes = DataIngestion().initiate_data_ingestion()
    trainer = Model_Training()
    trainer.train_model(model=model, criterion=criterion, 
                                           optimizer=optimizer,scheduler=exp_lr_scheduler,
                                           dataloader=dataloader,dataset_sizes=dataset_sizes)