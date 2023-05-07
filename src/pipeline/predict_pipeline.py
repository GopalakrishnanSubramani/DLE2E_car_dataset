import torch
from src.components.data_ingestion import DataIngestion
from src.components.model import BUILD_MODEL
import torchvision.transforms as T
from PIL import Image
from src.dirs import class_mapping, dirs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from dataclasses import dataclass
from src.utils import Utils
matplotlib.style.use('ggplot')
import cv2

@dataclass
class INFERENCE_CONFIG:
    model_weights = dirs["saved_model"]
    single_img_path = dirs["single_img_path"]
    dataloader = DataIngestion().initiate_test_data_ingestion()

class INFERENCE:

    def __init__(self):
        self.class_mapping = class_mapping
        self.config = INFERENCE_CONFIG()

    
    def transform_image(self,image):
        my_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        ])

        return my_transforms(image) #.unsqueeze(0)

    def prepare_model(self):
            # load back the model
        model = BUILD_MODEL().init_model(pretrained=False)
        model = model['EFFICIENTNET'].to('cpu')
        state_dict = torch.load(self.config.model_weights, torch.device('cpu'))["model_state_dict"]
        num_ftrs = model.classifier[1].in_features
        model.fc = torch.nn.Linear(num_ftrs, out_features=len(self.class_mapping))   
        model.load_state_dict(state_dict)
        model.eval()
        
        return model

    def get_prediction(self):
        image = Image.open(self.config.single_img_path)
        tensor = self.transform_image(image=image)
        model = self.prepare_model()
        outputs = model(tensor[None, ...])
        _, predicted = torch.max(outputs, 1)

        Utils().cv_prediction(tensor,"car",str(self.class_mapping[predicted]))

    def batch_prediction(self):
        #check Ground Truth class names
        model = self.prepare_model()
        for i,(inputs,classes) in enumerate(self.config.dataloader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            Utils().cv_batch_prediction(inputs,str(self.class_mapping[classes]),str(self.class_mapping[predicted]))
            # break


if __name__=='__main__':
    INFERENCE().get_prediction()