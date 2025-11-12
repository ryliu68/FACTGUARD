import logging
import os
import json
import random
import torch
import numpy as np
# Ablation study can be completed by here.
from models.factguard import Trainer as FACTGUARDTrainer
from models.factguardd import Trainer as FACTGUARDDTrainer

def setup_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def frange(x, y, jump):
    """
    Generate a range of floating point numbers.
    """
    while x < y:
        x = round(x, 8)
        yield x
        x += jump

class Run():
    """
    Main class for running grid search and training.
    """
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        """
        Create a file logger for logging training process.
        """
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(level = logging.INFO)
            handler = logging.FileHandler(log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        """
        Convert configinfo to a dictionary.
        """
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        """
        Main function for grid search over hyperparameters and training.
        """
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + self.config['data_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)

        train_param = { 'lr': [self.config['lr']] }  # Hyperparameter grid

        print(train_param)
        param = train_param
        best_param = []

        json_dir = os.path.join(
            './logs/json/',
            self.config['model_name'] + '_' + self.config['data_name']
        )
        json_path = os.path.join(
            json_dir,
            'month_' + str(self.config['month']) + '.json'
        )
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

        json_result = []
        for p, vs in param.items():
            setup_seed(self.config['seed'])
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                self.config['lr'] = v

                # Select trainer based on model name
                if self.config['model_name'] == 'FACTGUARD':
                    trainer = FACTGUARDTrainer(self.config, self.writer)
                elif self.config['model_name'] == 'FACTGUARD-D':
                    trainer = FACTGUARDDTrainer(self.config, self.writer)
                else:
                    raise ValueError('model_name is not supported')

                # Train and evaluate model
                metrics, model_path, train_epochs = trainer.train(logger)
                json_result.append({
                    'lr': self.config['lr'],
                    'metric': metrics,
                    'train_epochs': train_epochs,
                })

                # Update best metric and model path
                if metrics['metric'] > best_metric['metric']:
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best macro f1:", best_metric['metric'])
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('==================================================\n\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)

        return best_metric
