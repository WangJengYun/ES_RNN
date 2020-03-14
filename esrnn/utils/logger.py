import logging
import os 
from datetime import datetime

logs_path = os.path.join(os.getcwd(),'logs')

if not os.path.exists('logs'):
    os.makedirs(logs_path)

filename = 'ES_RNN_{:%Y-%m-%d_%H-%M-%S}.log'.format(datetime.now())

def create_logger(logger_name):

    my_logger= logging.getLogger(logger_name)
    my_logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s %(message)s','%Y-%m-%d %H:%M:%S')    

    fileHander = logging.FileHandler(os.path.join(logs_path,filename),mode = 'a',encoding = 'utf-8')
    fileHander.setLevel(logging.INFO)
    fileHander.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    my_logger.addHandler(fileHander)
    my_logger.addHandler(console)
    
    return(my_logger)
    