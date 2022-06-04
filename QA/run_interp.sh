export DATA_DIR=outputs
DATASET=${1:-none}
METHOD=${2:-none}
ACTION=${3:-run}
SPLIT=${4:-dev}

MAX_SEQ_LENGTH=512
if [ "$DATASET" = "squad" ]; then
    MAX_SEQ_LENGTH=512
elif [ "$DATASET" = "trivia" ]; then
    MAX_SEQ_LENGTH=512
elif [ "$DATASET" = "hotpot" ]; then
    MAX_SEQ_LENGTH=512
else
  echo "Invalid dataset ${DATASET}"
  exit 1
fi

MODEL_TYPE="roberta-base"
if [ "$ACTION" = "run" ]; then
  if [ "$METHOD" = "lime" -o "$METHOD" = "shap" ]; then
    echo "Run "${METHOD}
    python -u run_${METHOD}.py \
      --model_type roberta \
      --model_name_or_path checkpoints/squad_roberta-base \
      --dataset ${DATASET} \
      --predict_file $DATA_DIR/${SPLIT}_${DATASET}.json \
      --overwrite_output_dir \
      --max_seq_length ${MAX_SEQ_LENGTH} \
      --output_dir pred_output \
      --interp_dir interpretations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE} 2>&1
  else
    echo "No such method" $METHOD
  fi
elif [ "$ACTION" = "vis" ]; then
  if [ "$METHOD" = "lime" -o "$METHOD" = "shap" ]; then
    echo "Vis ${METHOD}"
    python -u run_${METHOD}.py \
      --model_type roberta \
      --tokenizer_name $MODEL_TYPE \
      --model_name_or_path $MODEL_TYPE \
      --dataset ${DATASET} \
      --do_vis \
      --output_dir pred_output \
      --predict_file $DATA_DIR/${SPLIT}_${DATASET}.json \
      --interp_dir interpretations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE} \
      --visual_dir visualizations/${METHOD}/${DATASET}_${SPLIT}_${MODEL_TYPE} 2>&1
  else
    echo "No such method"
  fi
else
  echo "run or vis"
fi
