unset SETUPTOOLS_USE_DISTUTILS
export OMP_NUM_THREADS=4
autoalign-cli dmpo \
    --model_name_or_path ${MODEL_PATH:?"MODEL_PATH is required"} \
    --ref_model_name_or_path ${REF_MODEL_PATH:-${MODEL_PATH}} \
    --data_path ${DATA_PATH:?"DATA_PATH is required"} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR:?"OUTPUT_DIR is required"} \
    --num_train_epochs ${EPOCH:-"3"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"1"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"4"} \
    --gradient_accumulation_steps ${GA:-"1"} \
    --evaluation_strategy ${EVAL_STRATEGY:-"no"} \
    --save_strategy ${SAVE_STRATEGY:-"no"} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-"5"} \
    --learning_rate ${LR:-"1e-6"} \
    --beta ${BETA:-"0.1"} \
    --gamma ${GAMMA:-"0.7"} \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type ${LR_SCHEDULE:-"constant_with_warmup"} \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --logging_dir ${OUTPUT_DIR} \
    --logging_steps ${LOGGING_STEPS:-"5"} \
    --model_max_length ${MODEL_MAX_LENGTH:-"4096"} \
    --max_prompt_length ${MAX_PROMPT_LENGTH:-"512"} \
    --max_target_length ${MAX_TARGET_LENGTH:-"3072"} \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG:-"configs/zero3.json"}
