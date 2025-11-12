import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Label mapping dictionary for main classification
label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1
}

# Label mapping dictionary for feature prediction, including an 'other' class
label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2
}

def word2input(texts, max_len, tokenizer):
    """
    Tokenizes and encodes a list of texts, returning both input IDs and attention masks.

    Args:
        texts (list): List of input text strings.
        max_len (int): Maximum sequence length.
        tokenizer: HuggingFace tokenizer.

    Returns:
        tuple: (token_ids, masks), both as torch tensors.
    """
    token_ids = []
    for i, text in enumerate(texts):
        # Encode each text using the tokenizer, add special tokens, pad and truncate as needed
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        # Create attention mask (1 for tokens, 0 for padding)
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language):
    """
    Loads data from the given path, processes it, and returns a DataLoader for training/evaluation.

    Args:
        path (str): Path to the data file.
        max_len (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.
        bert_path (str): Path or name of the pretrained BERT model.
        data_type (str): Type of data to load ('rationale' expected).
        language (str): Language of the dataset (not used in this function).

    Returns:
        DataLoader: PyTorch DataLoader ready for model training or evaluation.
    """
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    # tokenizer =  BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    if data_type == 'rationale':
        # Read the input JSON data
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        # Initialize DataFrame with required columns
        df_data = pd.DataFrame(columns=('content', 'label'))
        for item in data_list:
            tmp_data = {}
            tmp_data['classify'] = item.get('classify', None)  # Default to None if key not present
            # Extract content and label information
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']

            # Rationales and predictions/accuracies for features
            tmp_data['FTR_2'] = item['td_rationale']
            tmp_data['FTR_3'] = item['cs_rationale']
            tmp_data['FTR_2_pred'] = item['td_pred']
            tmp_data['FTR_3_pred'] = item['cs_pred']
            tmp_data['FTR_2_acc'] = item['td_acc']
            tmp_data['FTR_3_acc'] = item['cs_acc']

            # Append to DataFrame
            df_data = df_data._append(tmp_data, ignore_index=True)

        # Convert columns to numpy arrays or tensors as needed
        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        id = torch.tensor(df_data['id'].to_numpy())

        FTR_2_pred = torch.tensor(df_data['FTR_2_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())
        FTR_3_pred = torch.tensor(df_data['FTR_3_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())

        FTR_2_acc = torch.tensor(df_data['FTR_2_acc'].astype(float).to_numpy())
        FTR_3_acc = torch.tensor(df_data['FTR_3_acc'].astype(float).to_numpy())

        FTR_2 = df_data['FTR_2'].to_numpy()
        FTR_3 = df_data['FTR_3'].to_numpy()

        # Tokenize and encode all required text fields
        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
        FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

        # Build the TensorDataset with all relevant fields
        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                FTR_2_pred,
                                FTR_2_acc,
                                FTR_3_pred,
                                FTR_3_acc,
                                FTR_2_token_ids,
                                FTR_2_masks,
                                FTR_3_token_ids,
                                FTR_3_masks,
                                label,
                                id,
                                )
        # Create and return the DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    else:
        print('No match data type!')
        exit()