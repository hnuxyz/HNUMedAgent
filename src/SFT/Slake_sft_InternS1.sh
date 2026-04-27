#!/bin/bash
# 创建日志目录
LOG_DIR="./SFT_output"
mkdir -p $LOG_DIR

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/[SFT]internS1_mini_${TIMESTAMP}.log"

# 设置环境变量
# export ENABLE_AUDIO_OUTPUT=False
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# 设置随机端口号，避免端口冲突
export MASTER_PORT=$((10000 + RANDOM % 50000))

# 先打印启动信息
echo "Starting training..."
echo "Log file: $LOG_FILE"
echo "Using port: $MASTER_PORT"

# 没有指定 model_type
# 启动训练并获取PID
swift sft \
    --model '/root/share/new_models/Intern-S1-mini' \
    --dataset '/root/VLM_Med/datasets/slake_data/train.jsonl' \
    --val_dataset '/root/VLM_Med/datasets/slake_data/val.jsonl' \
    --eval_steps 17 \
    --train_type lora \
    --lora_rank 16 \
    --lora_dropout 0.05 \
    --lorap_lr_ratio 16\
    --lora_alpha 32 \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 4 \
    --save_steps 85 \
    --save_total_limit 20 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir '/root/VLM_Med/SFT_output/InternS1_mini'\
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --metric acc \
    --freeze_vit true \
    > "$LOG_FILE" 2>&1 &

# 获取PID并等待一下确保进程启动
TRAIN_PID=$!
sleep 2

# 检查进程是否还在运行
if kill -0 $TRAIN_PID 2>/dev/null; then
    echo "Training started successfully with PID $TRAIN_PID"
    echo "To view logs in real-time, use:"
    echo "tail -f $LOG_FILE"
    echo ""
    echo "To stop training, use:"
    echo "kill -9 $TRAIN_PID"
else
    echo "Failed to start training process"
    echo "Check log file for errors: $LOG_FILE"
fi