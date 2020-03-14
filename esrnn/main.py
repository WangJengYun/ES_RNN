import pandas as pd 
import time 
from esrnn.config import get_config
from esrnn.utils.logger import create_logger
from esrnn.data_reading import create_dataset
from esrnn.data_loading import SeriesDataset
from torch.utils.data import DataLoader
from esrnn.model import ESRNN
from torchsummary import summary
import time 

logger = create_logger('main')
config = get_config('Daily')
train_path = './data/Train/%s-train.csv'%(config['variable'])
test_path = './data/Test/%s-test.csv'%(config['variable'])

### reading info data 
logger.info("Start : reading info data \n")
info_table = pd.read_csv('./data/info.csv')
train, val, test = create_dataset(train_path,test_path,config['output_size'])
logger.info("End : reading info data \n")


### input data for pytorch model 
logger.info("Start : input data for pytorch model  \n")
dataset = SeriesDataset(train, val,test, info_table,\
                        config['variable'], config['chop_val'], config['device'])
dataloader = DataLoader(dataset,batch_size=config['batch_size'], shuffle=True)
logger.info("End :input data for pytorch model \n")

logger.info("building the empty  esrnn model  \n")
model  = ESRNN(num_series = len(dataset),config = config)

logger.info("Start : training model \n")
run_id = str(int(time.time()))
logger.info("CurrentID = run_id\n")


