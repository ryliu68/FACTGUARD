
# FACTGUARD
python main.py \
    --seed 3759 \
    --gpu 1 \
    --lr 2e-5 \
    --model_name FACTGUARD \
    --language ch \
    --root_path ./data/zh_enhanced_loss_0/ \
    --bert_path ./pretrain_model/bert-base-chinese/\
    --data_name zh-factguard \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 0.4 \
    --llm_text_encoder_weight 0.15912392468192324 \
#rationale_usefulness_evaluator_weight is alpha,llm_text_encoder_weight is beta

# FACTGUARD-D
python main.py \
    --seed 3759 \
    --gpu 0 \
    --lr 5e-4 \
    --model_name FACTGUARD-D \
    --language ch \
    --root_path ./data/zh_enhanced_loss_0/  \
    --bert_path ./pretrain_model/bert-base-chinese \
    --data_name zh-factguardd \
    --data_type rationale \
    --kd_loss_weight 8 \
    --teacher_path ./param_model/FACTGUARD_zh-factguard/1/parameter_bert.pkl \
    --rationale_usefulness_evaluator_weight 0.4 \
    --llm_text_encoder_weight 0.15912392468192324 \
        #kd_loss_weight is lambda