export DATA_DIR=outputs
ACTION=${1:-none}

if [ "$ACTION" = "train" ]; then
    dataset=$2
    exp_id=$3

    exp_prefix="exps/${dataset}_${exp_id}/"

    mkdir ${exp_prefix}
    cp run_nli.sh "${exp_prefix}run_nli.sh"

    if [ "$dataset" = "mnli" ]; then
        CUDA_VISIBLE_DEVICES=1 \
        python -u run_nli.py \
            --model_type roberta \
            --model_name_or_path roberta-base \
            --dataset $dataset \
            --do_train \
            --do_eval \
            --disable_tqdm \
            --train_file $DATA_DIR/train-mrpc_${dataset}.jsonl \
            --predict_file $DATA_DIR/mrpc_${dataset}.jsonl \
            --learning_rate 2e-5 \
            --evaluate_during_training \
            --num_train_epochs 10 \
            --overwrite_output_dir \
            --max_seq_length 128 \
            --logging_steps 1000 \
            --eval_steps 6000 \
            --save_steps 12000 \
            --warmup_steps 60 \
            --output_dir "${exp_prefix}output" \
            --per_gpu_train_batch_size 32 \
            --per_gpu_eval_batch_size 32 2>&1 | tee "${exp_prefix}log.txt"
    else
        echo "invalid dataset"
    fi

elif [ "$ACTION" = "eval" ]; then
    dataset=$2
    split=${3:-dev}
    devices=${4:-0}
    CUDA_VISIBLE_DEVICES=${devices} \
    python -u run_nli.py \
        --model_type roberta \
        --model_name_or_path checkpoints/${dataset}_roberta-base \
        --dataset $dataset \
        --do_eval \
        --predict_file $DATA_DIR/${split}_${dataset}.jsonl \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --output_dir  predictions/${dataset} \
        --per_gpu_eval_batch_size 100
else
    echo "train or eval"
fi