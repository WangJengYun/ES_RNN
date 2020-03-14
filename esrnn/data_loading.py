import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import Dataset
from esrnn.utils.logger import create_logger

logger = create_logger('data_loading')

# chop_value = 200
def chop_series(train,chop_value):
    # train_len_before = [len(i) for i in train]
    logger.info('train_berfor_chop = {}'.format(len(train)))
    train_len_mask = [True if len(i)>= chop_value else False for i in train]
    train = [train[i][-chop_value:] for i in range(len(train)) if train_len_mask[i]]
    logger.info('train_after_chop = {}'.format(len(train)))
    # train_chop_info = pd.DataFrame({'train_len_before':train_len_before,
    #                         'train_len_mask':train_len_mask})
    
    return train,train_len_mask


# dataTrain, dataVal, dataTest = train, val, test
# variable = 'Daily' ; chop_value = 200;devic = 'cuda'
class SeriesDataset(Dataset):

    def __init__(self,dataTrain, dataVal, dataTest, info_table, variable, chop_value, device):
        dataTrain,mask = chop_series(dataTrain,chop_value)

        dataInfoCatOHE = pd.get_dummies(info_table[info_table['SP'] == variable]['category'])
        self.dataInfoCatHeader = np.array([i for i in dataInfoCatOHE.columns.values])
        self.dataInfoCat = torch.from_numpy(dataInfoCatOHE[mask].values).float()

        self.dataTrain = [torch.tensor(dataTrain[i]) for i in range(len(dataTrain))]
        self.dataVal = [torch.tensor(dataVal[i]) for i in range(len(dataVal)) if mask[i]]
        self.dataTest = [torch.tensor(dataTest[i]) for i in range(len(dataTest)) if mask[i]]
        self.device = device

    def __len__(self):
        return len(self.dataTrain)

    def __getitem__(self,idx):
        return self.dataTrain[idx].to(self.device),\
               self.dataVal[idx].to(self.device),\
               self.dataTest[idx].to(self.device),\
               self.dataInfoCat[idx].to(self.device),\
               idx
            


