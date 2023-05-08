from src.dirs import dirs
import glob
import os

train_data_path:str = glob.glob(os.path.join(dirs['segmentation_image_path'],'*.png'))
val_data_path:str = dirs['segmentation_mask_path'] 

print((train_data_path[0]))

