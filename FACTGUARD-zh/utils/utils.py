from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime as dt 
from tensorboardX import SummaryWriter
import os
import json
import pandas as pd

class Recorder():
    """
    Recorder class for early stopping and tracking the best metric during training.
    """

    def __init__(self, early_step):
        self.max = {'metric': 0}      # Store the best metric values observed so far
        self.cur = {'metric': 0}      # Store the current metric values
        self.maxindex = 0             # Index (epoch/step) at which best metric was achieved
        self.curindex = 0             # Current index (epoch/step)
        self.early_step = early_step  # Early stopping patience

    def add(self, x):
        """
        Add a new metric record and decide if early stopping or model saving is needed.
        Args:
            x (dict): Current metric dictionary.
        Returns:
            str: 'save', 'esc', or 'continue' for model saving, early stopping, or continuation.
        """
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        """
        Judge whether to save the model, stop early, or continue.
        Returns:
            str: Decision signal.
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
    Custom JSON Encoder for numpy data types.
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
    Compute various classification metrics.
    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels or scores.
    Returns:
        dict: Various evaluation metrics.
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
    Move batch data to GPU if use_cuda is True, otherwise keep on CPU.
    Args:
        batch (tuple): Batch of tensors.
        use_cuda (bool): Whether to use CUDA.
        data_type (str): Data type flag (should be 'rationale').
    Returns:
        dict: Batch data dictionary on the target device.
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
    Class for tracking running average of a value.
    """

    def __init__(self):
        self.n = 0  # Number of values added
        self.v = 0  # Current average value

    def add(self, x):
        """
        Add a new value and update the running average.
        Args:
            x (float): Value to add.
        """
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        """
        Get the current average value.
        Returns:
            float: Current average.
        """
        return self.v

def get_monthly_path(data_type, root_path, month, data_name):
    """
    Get file path for a monthly dataset.
    Args:
        data_type (str): Data type (should be 'rationale').
        root_path (str): Root directory.
        month (str): Month info (unused here).
        data_name (str): Data file name.
    Returns:
        str: Full file path.
    """
    if data_type == 'rationale':
        file_path = os.path.join(root_path, data_name)
        return file_path
    else:
        print('No match data type!')
        exit()

def get_tensorboard_writer(config):
    """
    Create and return a TensorBoard SummaryWriter for logging.
    Args:
        config (dict): Configuration containing tensorboard_dir, model_name, and data_name.
    Returns:
        SummaryWriter: TensorBoard writer object.
    """
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    writer_dir = os.path.join(config['tensorboard_dir'], config['model_name'] + '_' + config['data_name'], TIMESTAMP)
    writer = SummaryWriter(logdir=writer_dir, flush_secs=5)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    return writer

def process_test_results(test_file_path, test_res_path, label, pred, id, ae, acc):
    """
    Save test results to a JSON file, including predictions and extra metrics.
    Args:
        test_file_path (str): Path to the original test set JSON file.
        test_res_path (str): Path where results will be saved.
        label (list): True labels.
        pred (list): Model predictions.
        id (list): Data IDs.
        ae (list): Additional metric (e.g., absolute error).
        acc (list): Accuracy per sample.
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