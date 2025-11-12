import logging
import os
import json
import random
import torch
import numpy as np

from models.factguard import Trainer as FACTGUARDTrainer
from models.factguardd import Trainer as FACTGUARDDTrainer

def setup_seed(seed):
    """
    Set random seed for reproducibility across random, numpy, and torch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def frange(x, y, jump):
    """
    Generate a range of floating point numbers from x (inclusive) to y (exclusive) with step size jump.

    Args:
        x (float): Start value.
        y (float): End value (exclusive).
        jump (float): Step size.

    Yields:
        float: Next value in the sequence.
    """
    while x < y:
        x = round(x, 8)  # Avoid floating point precision issues
        yield x
        x += jump

class Run():
    """
    Run class for managing model training, logging, and hyperparameter search.
    """

    def __init__(self, config, writer):
        """
        Initialize with config and TensorBoard writer.

        Args:
            config (dict): Configuration dictionary.
            writer: TensorBoard writer instance.
        """
        self.config = config
        self.writer = writer

    def getFileLogger(self, log_file):
        """
        Get a file logger for logging training process.

        Args:
            log_file (str): Path to the log file.

        Returns:
            Logger: Configured logger instance.
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
        Convert configinfo attribute to a standard dictionary.

        Returns:
            dict: Configuration dictionary.
        """
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        """
        Main execution function for model training, logging, and hyperparameter management.

        Returns:
            dict: Best metric achieved in this run.
        """
        # Create parameter log directory if not exists
        param_log_dir = self.config['param_log_dir']
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.config['model_name'] + '_' + self.config['data_name'] +'_'+ 'param.txt')
        logger = self.getFileLogger(param_log_file)

        # Define hyperparameter grid (currently only learning rate)
        train_param = { 'lr': [self.config['lr']] }

        print(train_param)
        param = train_param
        best_param = []

        # Prepare directory for storing results as JSON
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

                # Choose model trainer based on model_name in config
                if self.config['model_name'] == 'FACTGUARD':
                    trainer = FACTGUARDTrainer(self.config, self.writer)
                elif self.config['model_name'] == 'FACTGUARD-D':
                    trainer = FACTGUARDDTrainer(self.config, self.writer)
                else:
                    raise ValueError('model_name is not supported')

                # Train the model and get metrics, model path, and number of epochs
                metrics, model_path, train_epochs = trainer.train(logger)
                json_result.append({
                    'lr': self.config['lr'],
                    'metric': metrics,
                    'train_epochs': train_epochs,
                })

                # Update best metric and best model if necessary
                if metrics['metric'] > best_metric['metric']:
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best macro f1:", best_metric['metric'])
            print("best metric:", best_metric)
            logger.info("best model path:" + str(best_model_path))
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('==================================================\n\n')
        # Save results to JSON file
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)

        return best_metric