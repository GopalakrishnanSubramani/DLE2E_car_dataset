import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from dataclasses import dataclass
from src.dirs import dirs
import cv2 
matplotlib.style.use('ggplot')


@dataclass
class UtilsConfig:

    model_dir = os.path.join(dirs['model_dir'])
    plot_dir = os.path.join(dirs['plot_dir'])

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)


class Utils:
    def __init__(self):
        self.config = UtilsConfig()


    def save_model(self,name,epochs, model, optimizer, criterion):
        """
        Function to save the trained model to disk.
        """     

        torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, os.path.join(self.config.model_dir,f"{name}_model.pth"))
        
    def save_plots(self,name,train_acc, valid_acc, train_loss, valid_loss):
        """
        Function to save the loss and accuracy plots to disk.
        """
        # Accuracy plots.
        
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-', 
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-', 
            label='validataion accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.config.plot_dir,f"{name}_accuracy.png"))
        
        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-', 
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(UtilsConfig().plot_dir,f"{name}_loss.png"))

    def imshow(self,inp, title=None):
        "Imshow for tensor"
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.figure()
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.show()

    def cv_batch_prediction(self,inp,ground_truth:str,prediction:str,win:str="Car Classification"):
        inp = inp[0].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean

        font, fontscale = cv2.FONT_HERSHEY_SIMPLEX,1
        ground_truth_color,prediction_color = (0,255,0),(0,0,255)
        thickness = 1

        inp=cv2.resize(inp,(1000,500))
        inp = cv2.putText(inp,ground_truth,(0,30),font,fontscale,ground_truth_color,thickness,cv2.LINE_4)
        inp = cv2.putText(inp,prediction,(250,470),font,fontscale,prediction_color,thickness,cv2.LINE_4)
        cv2.imshow(win,inp)
        cv2.waitKey(0)
    
    def cv_prediction(self,inp,ground_truth:str,prediction:str,win:str="Car Classification"):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean

        font, fontscale = cv2.FONT_HERSHEY_SIMPLEX,1
        ground_truth_color,prediction_color = (0,255,0),(0,0,255)
        thickness = 1

        inp=cv2.resize(inp,(1000,500))
        inp = cv2.putText(inp,ground_truth,(0,30),font,fontscale,ground_truth_color,thickness,cv2.LINE_4)
        inp = cv2.putText(inp,prediction,(250,470),font,fontscale,prediction_color,thickness,cv2.LINE_4)
        cv2.imshow(win,inp)
        cv2.waitKey(0)