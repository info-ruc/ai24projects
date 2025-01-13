#!/bin/bash
#model="/mnt/d/AI/model/chinese/chinese-roberta-wwm-ext"
model="hfl/chinese-roberta-wwm-ext" #-large # princeton-nlp/sup-simcse-roberta-base # large #或者换成本地模型路径 
model_type=v1   #这里选择需要训练的模型
transform=True
objective=classification # contrast # rank # focal #在报告中提到了不同模型对应的任务，在使用时请进行二次确认
measure=dot_product #cosine # sigmoid #cosine在v1模型下表现也挺好
use_weight=False
batch_size=8
lr=1e-5
wd=0.1
seed=42

num_train_epochs=10

train_file=data # data/sft_retrieval_data_fkt.json       #这些数据来自不同的源
eval_file=data # data/sft_retrieval_data_lawyerllama.json
test_file=data/sft_retrieval_data_lawyerllama.json

output_dir=output
config="encoder_${model_type}__meas_${measure}__obj_${objective}__bsz_${batch_size}__lr_${lr}__wd_${wd}__s_${seed}__weight_${use_weight}"


python3 run_sts.py \
  --output_dir "${output_dir}/${model//\//__}/${config}" \
  --model_name_or_path ${model} \
  --model_type ${model_type} \
  --objective ${objective} \
  --measure ${measure} \
  --pooler_type cls \
  --freeze_encoder False \
  --transform ${transform} \
  --max_seq_length 512 \
  --train_file ${train_file} \
  --validation_file ${eval_file} \
  --test_file ${test_file} \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --per_device_train_batch_size ${batch_size} \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing False \
  --learning_rate ${lr} \
  --weight_decay ${wd} \
  --fp16 True \
  --max_grad_norm 0.0 \
  --num_train_epochs ${num_train_epochs} \
  --lr_scheduler_type linear \
  --warmup_ratio 0.1 \
  --log_level info \
  --disable_tqdm True \
  --save_strategy epoch \
  --save_total_limit 2 \
  --seed ${seed} \
  --data_seed ${seed} \
  --log_time_interval 15 \
  --overwrite_output_dir True \
  --train_test_split 0.1 \
  --num_show_example 8 \
  --show_verbosity 1 \
  --embedding_init False \
  --use_weight ${use_weight} \
  #--max_train_samples 16 \
  #--max_eval_samples 8
  

  