import logging
import os
from datetime import datetime   
from src.dirs import dirs

LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y %H-%M-%S')}.log"
log_path = os.path.join(dirs['logging_path'], "logs", LOG_FILE)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH =os.path.join(log_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
