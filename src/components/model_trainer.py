import time
import copy
import torch
from dataclasses import dataclass
from model import build_model
from src.utils import Utils
from tqdm.auto import tqdm
import os

@dataclass
class TrainConfig:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model_Training:
    def __init__(self):
        self.utils = Utils()

    def train_model(self,model, criterion, optimizer, scheduler,dataloader,device,dataset_sizes,num_epochs=2):
        
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        accuracy = {'train':[0],'val':[0]}
        loss_epoch = {'train':[0],'val':[0]}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i,(inputs, labels) in tqdm(enumerate(dataloader[phase]), total=len(dataloader[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                accuracy[phase].append(epoch_acc)
                loss_epoch[phase].append(epoch_loss)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)

        # Save the trained model weights.
        self.utils.save_model(num_epochs, model, optimizer, criterion)

            # Save the loss and accuracy plots.
        self.utils.save_plots(accuracy['train'], accuracy['val'], loss_epoch['train'], loss_epoch['val'])

#         return model

