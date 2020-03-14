import pandas as pd 
import numpy as np 
from esrnn.utils.logger import create_logger

logger = create_logger('read_data')

# file_path = test_path
def read_ts_data(file_path):
    series = []
    ids = []

    with open(file_path,'r') as file:
        data = file.read().split('\n')
    
    for i in range(1,len(data)): 
        row = data[i].replace('"','').split(',')
        series.append(np.array([float(j) for j in row[1:] if j!='']))
        ids.append(row[0])
    
    series = np.array(series)

    return series

def create_dataset(train_path,test_path,output_size):
    
    train_data = read_ts_data(train_path)
    logger.info("train_data = {}".format(train_data.shape))
    
    test_data = read_ts_data(test_path)
    logger.info("test_data = {}".format(test_data.shape))

    val = []
    for i in range(len(train_data)):
        val.append(train_data[i][-output_size:])
        train_data[i] = train_data[i][:-output_size]
    
    val_data = np.array(val)
    logger.info("val_data = {}".format(val_data.shape))

    return train_data , test_data ,val_data
