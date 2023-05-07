import torch
from src.components.data_ingestion import DataIngestion
from src.components.model import BUILD_MODEL
import torchvision.transforms as T
from PIL import Image
from src.dirs import class_mapping, dirs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import cv2


def transform_image(image):
    my_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])

    return my_transforms(image) #.unsqueeze(0)

def get_prediction(model,tensor,class_mapping):
    # tensor = transform_image(image=image)
    outputs = model(tensor[None, ...])
    _, predicted = torch.max(outputs, 1)

    return class_mapping[predicted]

def batch_prediction(model,tensor,class_mapping):
    # tensor = transform_image(image=image)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)

    return class_mapping[predicted]

def imshow(inp, title=None):
    inp = inp[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    cv2.imshow("CV prediction",cv2.resize(inp,(1000,500)))
    cv2.waitKey(0)
    print(inp.shape)
    

if __name__=='__main__':
    from torchvision import utils
    from src.utils import Utils

    model_weights = dirs["saved_model"]

    single_img_path = "/home/krish/Documents/PyTorch/End2End_Deep_learning_project_using_segmentation@classification/car_data_full/train/Audi R8 Coupe 2012/01877.jpg"
    single_img = Image.open(single_img_path)

    dataloader = DataIngestion().initiate_test_data_ingestion()
    inputs, classes = next(iter(dataloader))
    # image = Image.open(image)

    # load back the model
    model = BUILD_MODEL().init_model(pretrained=False)
    model = model['EFFICIENTNET'].to('cpu')
    # state_dict = torch.load(path, map_location=torch.device('cpu'))
    num_ftrs = model.classifier[1].in_features
    model.fc = torch.nn.Linear(num_ftrs, out_features=len(class_mapping))
    model.load_state_dict(torch.load(model_weights,map_location=torch.device('cpu'))["model_state_dict"])    
    model.eval()

    # get a sample from the validation dataset for inference

    for i,(inputs, classes) in enumerate(dataloader):
        # predicted = get_prediction(inputs)
        predicted = batch_prediction(model,inputs,class_mapping)

        # out = utils.make_grid(inputs)
        # Utils().imshow(out, title=predicted+f"---- {i}\n\naaaaaaaaa")

        imshow(inputs)

   
        break

    # trans_img = transform_image(single_img)
    # predicted = get_prediction(model,trans_img, class_mapping)
    # print(predicted)

