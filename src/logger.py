import logging
import os
from datetime import datetime   

path = "/home/krish/Documents/PyTorch/End2End_Deep_learning_project_using_segmentation@classification/"

LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y %H-%M-%S')}.log"
log_path = os.path.join(path, "logs", LOG_FILE)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH =os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
