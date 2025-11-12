# FACTGUARD
python main.py \
    --seed 3759 \
    --gpu 0 \
    --lr 5e-5 \
    --model_name FACTGUARD \
    --language en \
    --root_path ./data/en_enhanced_loss_0\
    --bert_path ./pretrain_model/roberta/ \
    --data_name en-factguard \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 0.5 \
    --llm_text_encoder_weight 0.579938945331448
#rationale_usefulness_evaluator_weight is alpha,llm_text_encoder_weight is beta


    
    
# # FACTGUARD-D
python main.py \
    --seed 3759 \
    --gpu 0 \
    --lr 5e-5 \
    --model_name FACTGUARD-D \
    --language en \
    --root_path ./data/en_enhanced_loss_0 \
    --bert_path ./pretrain_model/roberta \
    --data_name en-factguardd \
    --data_type rationale \
    --rationale_usefulness_evaluator_weight 0.5 \
    --llm_text_encoder_weight 0.579938945331448 \
    --teacher_path ./param_model/FACTGUARD_en-factguard/1/parameter_bert.pkl \
    --kd_loss_weight 8 \
 #kd_loss_weight is lambda