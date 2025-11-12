import os
import torch
import tqdm
import time
from .layers import *
from sklearn.metrics import *
from transformers import BertModel,RobertaModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader
from utils.utils import get_monthly_path, get_tensorboard_writer, process_test_results

class FACTGUARDModel(torch.nn.Module):
    """
    FACTGUARD model using Bert backbone and attention/MLP heads for classification.
    Only uses content and FTR_2 features.
    """
    def __init__(self, config):
        super(FACTGUARDModel, self).__init__()

        # Load pretrained Bert models for content and FTR features
        self.bert_content = BertModel.from_pretrained(config['bert_path'],ignore_mismatched_sizes=True).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config['bert_path'],ignore_mismatched_sizes=True).requires_grad_(False)
       
        # Only train the last encoder layer
        for name, param in self.bert_content.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in self.bert_FTR.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Aggregator and MLP for final prediction
        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])

        # Hard rationale head for FTR_2
        self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_2 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 1),
            nn.Sigmoid()
        )
        self.score_mapper_ftr_2 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['model']['mlp']['dims'][-1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Hard rationale head for FTR_3
        self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_3 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 1),
            nn.Sigmoid()
        )
        self.score_mapper_ftr_3 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['model']['mlp']['dims'][-1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Simple rationale heads for FTR_2 and FTR_3
        self.simple_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_2 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 2)
        )
        self.simple_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_3 = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Linear(config['model']['mlp']['dims'][-1], 3)
        )

        # Content attention
        self.content_attention = MaskAttention(config['emb_dim'])    

        # Co-attention modules
        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        # Cross attention modules
        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

    
    def forward(self, **kwargs):
        """
        Forward pass for FACTGUARD model.
        """
        content, content_masks = kwargs['content'], kwargs['content_masks']
        FTR_2, FTR_2_masks = kwargs['FTR_2'], kwargs['FTR_2_masks']

        # Extract features using Bert backbone
        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        content_feature_1 = content_feature
        
        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask = FTR_2_masks)[0]
        content_feature_2 = FTR_2_feature

        # Content attention (average of two passes)
        attn_content1_1, _ = self.content_attention(content_feature_1, mask=content_masks)
        attn_content1_2, _ = self.content_attention(content_feature_1, mask=content_masks)
        attn_content1 = (attn_content1_1 + attn_content1_2) / 2
        attn_content2_1, _ = self.content_attention(content_feature_2, mask=content_masks)
        attn_content2_2, _ = self.content_attention(content_feature_2, mask=content_masks)
        attn_content2 = (attn_content2_1 + attn_content2_2) / 2
        attn_content = (attn_content1 + attn_content2) / 2

        # Aggregate features and predict
        all_feature = attn_content.unsqueeze(1)
        final_feature, _ = self.aggregator(all_feature)
        label_pred = self.mlp(final_feature)

        # Collect outputs
        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'final_feature': final_feature,
            'content_feature': attn_content,          
        }

        return res


class Trainer():
    """
    Trainer class for FACTGUARD model, including training, validation, and testing.
    """
    def __init__(self,
                 config,
                 writer
                 ):
        self.config = config
        self.writer = writer
        self.num_expert = 2
        
        # Set save path for model parameters
        self.save_path = os.path.join(
            self.config['save_param_dir'],
            self.config['model_name']+'_'+self.config['data_name'],
            str(self.config['month']))
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        

    def train(self, logger = None):
        """
        Training loop for FACTGUARD model.
        """
        st_tm = time.time()
        writer = self.writer

        if(logger):
            logger.info('start training......')
        print('\n\n')
        print('==================== start training ====================')

        self.model = FACTGUARDModel(self.config)

        if self.config['use_cuda']:
            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'])

        # Load training, validation, and test data
        train_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'train.json')
        train_loader = get_dataloader(
            train_path, 
            self.config['max_len'], 
            self.config['batchsize'], 
            shuffle=True, 
            bert_path=self.config['bert_path'], 
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        val_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'val.json')
        val_loader = get_dataloader(
            val_path, 
            self.config['max_len'], 
            self.config['batchsize'], 
            shuffle=False, 
            bert_path=self.config['bert_path'], 
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        test_path = get_monthly_path(self.config['data_type'], self.config['root_path'], self.config['month'], 'test.json')
        test_future_loader = get_dataloader(
            test_path, 
            self.config['max_len'], 
            self.config['batchsize'], 
            shuffle=False, 
            bert_path=self.config['bert_path'],
            data_type=self.config['data_type'],
            language=self.config['language']
        )

        ed_tm = time.time()
        print('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
    
        for epoch in range(self.config['epoch']):
            print('---------- epoch {} ----------'.format(epoch))
            self.model.train()
            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss_classify = Averager()

            for step_n, batch in enumerate(train_data_iter):
                # Move batch data to GPU if configured
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                label = batch_data['label']

                batch_input_data = {**self.config, **batch_data}

                # Forward pass and loss calculation
                res = self.model(**batch_input_data)
                loss_classify = loss_fn(res['classify_pred'], label.float())

                loss = loss_classify

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss_classify.add(loss_classify.item())

            print('----- in val progress... -----')
            results, val_aux_info = self.test(val_loader)
            mark = recorder.add(results)
            print()

            # Tensorboard logging
            writer.add_scalar('month_'+str(self.config['month'])+'/train_loss', avg_loss_classify.item(), global_step=epoch)
            writer.add_scalars('month_'+str(self.config['month'])+'/test', results, global_step=epoch)

            # Logger output
            if(logger):
                logger.info('---------- epoch {} ----------'.format(epoch))
                logger.info('train loss classify: {}'.format(avg_loss_classify.item()))
                logger.info('\n')

                logger.info('val loss classify: {}'.format(val_aux_info['val_avg_loss_classify'].item()))

                logger.info('val result: {}'.format(results))
                logger.info('\n')

            if mark == 'save':
                torch.save(self.model.state_dict(),
                    os.path.join(self.save_path, 'parameter_bert.pkl'))
                print("save：",results)

            if mark == 'esc':
                break
            else:
                continue
        

        # Save test results
        test_dir = os.path.join(
            './logs/test/',
            self.config['model_name'] + '_' + self.config['data_name']
        )
        os.makedirs(test_dir, exist_ok=True)
        test_res_path = os.path.join(
            test_dir,
            'month_' + str(self.config['month']) + '.json'
        )

        # Load best model and run prediction on test set
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bert.pkl')))
        future_results, label, pred, id, ae, acc = self.predict(test_future_loader)

        writer.add_scalars('month_'+str(self.config['month'])+'/test', future_results)
        
        if(logger):
            logger.info("start testing......")
            logger.info("test score: {}.".format(future_results))
            logger.info("lr: {}, avg test score: {}.\n\n".format(self.config['lr'], future_results['metric']))
        print('test results:', future_results)
        return future_results, os.path.join(self.save_path, 'parameter_bert.pkl'), epoch

 

    def test(self, dataloader):
        """
        Validation/testing loop for FACTGUARD model.
        """
        loss_fn = torch.nn.BCELoss()
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        avg_loss_classify = Averager()
        
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                batch_label = batch_data['label']

                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)

                loss_classify = loss_fn(res['classify_pred'], batch_label.float())
            
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(res['classify_pred'].detach().cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item())

        aux_info = {
            'val_avg_loss_classify': avg_loss_classify
        }

        return metrics(label, pred), aux_info


    def predict(self, dataloader):
        """
        Prediction loop for FACTGUARD model.
        """
        if self.config['eval_mode']:
            self.model = FACTGUARDModel(self.config)
            if self.config['use_cuda']:
                self.model = self.model.cuda()
            print('========== in test process ==========')
            print('now load in test model...')
            self.model.load_state_dict(torch.load(self.config['eval_model_path']))
        pred = []
        label = []
        id = []
        ae = []
        accuracy = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(
                    batch, 
                    self.config['use_cuda'], 
                    data_type=self.config['data_type']
                )
                batch_label = batch_data['label']
                batch_input_data = {**self.config, **batch_data}
                res = self.model(**batch_input_data)
                batch_pred = res['classify_pred']

                cur_labels = batch_label.detach().cpu().numpy().tolist()
                cur_preds = batch_pred.detach().cpu().numpy().tolist()
                label.extend(cur_labels)
                pred.extend(cur_preds)
                ae_list = []
                for index in range(len(cur_labels)):
                    ae_list.append(abs(cur_preds[index] - cur_labels[index]))
                accuracy_list = [1 if ae<0.5 else 0 for ae in ae_list]
                ae.extend(ae_list)
                accuracy.extend(accuracy_list)
        
        return metrics(label, pred), label, pred, id, ae, accuracy
