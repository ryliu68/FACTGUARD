from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime as dt 
from tensorboardX import SummaryWriter
import os

import json
import pandas as pd

class Recorder():
    """
    Early stopping and model checkpoint recorder.
    Tracks the best metric and decides when to save or stop training.
    """
    def __init__(self, early_step):
        self.max = {'metric': 0}      # Best metric so far
        self.cur = {'metric': 0}      # Current metric
        self.maxindex = 0             # Epoch index of best metric
        self.curindex = 0             # Current epoch index
        self.early_step = early_step  # Early stopping patience

    def add(self, x):
        """
        Add new metric and judge if model should be saved or stopped.
        """
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        """
        Decide whether to save model, continue, or stop early.
        """
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        """
        Print the best metric so far.
        """
        print("Max", self.max)

class NpEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def metrics(y_true, y_pred):
    """
    Calculate various classification metrics.
    Returns a dictionary of metrics.
    """
    all_metrics = {}

    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        all_metrics['auc'] = -1
    try:
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except ValueError:
        all_metrics['spauc'] = -1
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    try:
        all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['f1_real'], all_metrics['f1_fake'] = -1, -1
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    try:
        all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['recall_real'], all_metrics['recall_fake'] = -1, -1
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    try:
        all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['precision_real'], all_metrics['precision_fake']= -1, -1
    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    
    return all_metrics

def data2gpu(batch, use_cuda, data_type):
    """
    Move batch data to GPU if use_cuda is True.
    Supports 'rationale' data_type.
    """
    if use_cuda:
        if data_type == 'rationale':
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'FTR_2_pred': batch[2].cuda(),
                'FTR_2_acc': batch[3].cuda(),
                'FTR_3_pred': batch[4].cuda(),
                'FTR_3_acc': batch[5].cuda(),
                'FTR_2': batch[6].cuda(),
                'FTR_2_masks': batch[7].cuda(),
                'FTR_3': batch[8].cuda(),
                'FTR_3_masks': batch[9].cuda(),
                'label': batch[10].cuda(),
                'id': batch[11].cuda(),
            }
        else:
            print('error data type!')
            exit()
    else:
        if data_type == 'rationale':
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'FTR_2_pred': batch[2],
                'FTR_2_acc': batch[3],
                'FTR_3_pred': batch[4],
                'FTR_3_acc': batch[5],
                'FTR_2': batch[6],
                'FTR_2_masks': batch[7],
                'FTR_3': batch[8],
                'FTR_3_masks': batch[9],
                'label': batch[10],
                'id': batch[11],
            }
        else:
            print('error data type!')
            exit()
    return batch_data

class Averager():
    """
    Utility class to compute running average.
    """
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        """
        Add a new value to the average.
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        """
        Get the current average value.
        """
        return self.v

def get_monthly_path(data_type, root_path, month, data_name):
    """
    Get the file path for a specific month and data type.
    """
    if data_type == 'rationale':
        file_path = os.path.join(root_path, data_name)
        return file_path
    else:
        print('No match data type!')
        exit()

def get_tensorboard_writer(config):
    """
    Create a tensorboardX SummaryWriter for logging.
    """
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    writer_dir = os.path.join(config['tensorboard_dir'], config['model_name'] + '_' + config['data_name'], TIMESTAMP)
    writer = SummaryWriter(logdir=writer_dir, flush_secs=5)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    return writer

def process_test_results(test_file_path, test_res_path, label, pred, id, ae, acc):
    """
    Save test results to a JSON file, including predictions and metrics for each sample.
    Args:
        test_file_path: Path to the test data file.
        test_res_path: Path to save the results.
        label: True labels.
        pred: Predicted values.
        id: Sample IDs.
        ae: Absolute error for each sample.
        acc: Accuracy for each sample.
    """
    test_result = []
    test_df = pd.read_json(test_file_path)
    for index in range(len(label)):
        cur_res = {}
        cur_id = id[index]
        cur_data = test_df[test_df['id'] == int(cur_id)].iloc[0]
        for (key, val) in cur_data.iteritems(): 
            cur_res[key] = val
        cur_res['pred'] = pred[index]
        cur_res['ae'] = ae[index]
        cur_res['acc'] = acc[index]

        test_result.append(cur_res)

    json_str = json.dumps(test_result, indent=4, ensure_ascii=False, cls=NpEncoder)

    with open(test_res_path, 'w') as f:
        f.write(json_str)
    return
