"""
In this file we configure which model to run and what to to with it
We also select the dataset which we want to operate on
"""
config = dict()

"""
Select which model to run 

Available modes: 'train', 'test', 'inference' 
"""
config['model'] = 'CNN'  # select which model to run
config['mode'] = 'train'


"""
Dataset related settings 
"""
config['dataset'] = 'EEG'
config['batch_size'] = 32
config['shuffle'] = True

config['train_dataset'] = ['data-1996-06-09-01-1.nc']
config['val_dataset'] = ['data-1996-06-09-01-1.nc']
config['test_dataset'] = ['data-1996-06-09-01-1.nc']

'''
Feature List
'''
config['feature_list'] = ['T500']

"""
Training related settings
"""
config['gpu_ids'] = [0]  # only use one gpu

"""
Evaluation related settings 
"""
#TODO:implement