import os 
import time 
import numpy as np 
import torch 
import torch.nn as nn 

class ESRNNTrainer(object):

    def __init__(self,model,dataloader,run_id,config,one_headers):
        
        # data setting 
        self.dl = dataloader 
        self.ohe_headers = ohe_headers

        # model setting 
        self.model = model.to(config['device'])
        self.config = config
        
        # training setting 
        self.run_id = str(run_id)
        self.epochs = 0 
        self.max_epochs = config['num_of_train_epochs']

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = config['learning_rate'])
        
        


        # other_setting
        self.epochs = 0 
     
