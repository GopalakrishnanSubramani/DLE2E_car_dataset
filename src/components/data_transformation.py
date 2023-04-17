from torchvision import transforms
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys


@dataclass
class DataTransformConfig:
    image_size:int = 224

class DataTransformation:
    def __init__(self):
        self.image_size = DataTransformConfig()

    def get_train_transform(self):
        """_summary_
            This function returns the augumentations for the validation dataset

        """
        try:
            logging.info("Entering Train transform")

            train_transform = transforms.Compose([
                transforms.Resize((self.image_size.image_size, self.image_size.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(35),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomPosterize(bits=2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
    ])
            logging.info("Train transform Completed successfully")
            return train_transform
        
        except Exception as e:
            logging.info("Error Train transform")
            raise CustomException(e,sys)
    
    def get_valid_transform(self):
        """_summary_
            This function returns the augumentations for the validation dataset

        """
        try:            
            logging.info("Entering Test transform")

            valid_transform = transforms.Compose([
                transforms.Resize((self.image_size.image_size, self.image_size.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )
            ])

            logging.info("Test transform Completed successfully")

            return valid_transform
        
        except Exception as e:
            logging.info("Error valid transform")
            raise CustomException(e,sys)