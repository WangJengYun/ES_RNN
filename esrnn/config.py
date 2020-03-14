import torch 
import numpy as np 

def get_config(interval):
    config = {
        'prod': True,
        'device': ("cuda" if torch.cuda.is_available() else "cpu"),
        'percentile': 50,
        'training_percentile': 45,
        'add_nl_layer': True,
        'rnn_cell_type': 'LSTM',
        'learning_rate': 1e-3,
        'learning_rates': ((10, 1e-4)),
        'num_of_train_epochs': 15,
        'num_of_categories': 6,  # in data provided
        'batch_size': 1024,
        'gradient_clipping': 20,
        'c_state_penalty': 0,
        'min_learning_rate': 0.0001,
        'lr_ratio': np.sqrt(10),
        'lr_tolerance_multip': 1.005,
        'min_epochs_before_changing_lrate': 2,
        'print_train_batch_every': 5,
        'print_output_stats': 3,
        'lr_anneal_rate': 0.5,
        'lr_anneal_step': 5
    }

    if interval == 'Daily':
        config.update({
            #     RUNTIME PARAMETERS
            'chop_val': 200,
            'variable': "Daily",
            'dilations': ((1, 7), (14, 28)),
            'state_hsize': 50,
            'seasonality': 7,
            'input_size': 7,
            'output_size': 14,
            'level_variability_penalty': 50
        })
    
    return config

