#! /bin/bash

. ./finetuning_constants.sh

python ./run_glue_edit.py \
    --model_name_or_path=${MODEL_NAME} \
    --do_train \
    --do_eval \
    --save_steps=${STEPS}\
    --save_strategy=steps \
    --save_total_limit=8 \
    --early_stopping_step=${EARLY_STOPPING_STEP} \
    --max_seq_length=${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --use_fast_tokenizer \
    --learning_rate=${LEARNING_RATE} \
    --num_train_epochs=${NUM_TRAIN_EPOCHS} \
    --output_dir=${OUTPUT_DIR} \
    --train_file_task1=${TRAIN_PATH_TASK1} \
    --train_file_task2=${TRAIN_PATH_TASK2} \
    --validation_file_task1=${EVAL_PATH_TASK1} \
    --validation_file_task2=${EVAL_PATH_TASK2} \
    --run_name ${PROJECT_NAME} \
    --logging_dir ${LOG_DIR} \
    --logging_strategy steps \
    --logging_steps=${STEPS} \
    --evaluation_strategy steps \
    --load_best_model_at_end\
    --eval_steps=${STEPS} \
    --metric_for_best_model eval_loss 
    
#     --overwrite_output_dir \