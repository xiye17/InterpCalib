RUN_SQUAD_EXP () {
    DATASET="squad"
    TRAIN_SIZE=${1:-500}
    echo "${DATASET}-${TRAIN_SIZE}"
    echo "METHOD, AUC, ACC, F1@25, F1@50, F1@75"
    CONF_RES=$(python calib_exp/run_exp.py --method shap --train_size ${TRAIN_SIZE} --dataset ${DATASET} --do_maxprob 2>/dev/null | tail -1)
    echo 'Conf,'${CONF_RES}
    KAM_RES=$(python calib_exp/run_exp.py --method shap --train_size ${TRAIN_SIZE} --dataset ${DATASET} --do_baseline --arg_n_tree 300 --arg_max_depth 6  2>/dev/null  | tail -1)
    echo 'KAMATH,'${KAM_RES}
    BOW_RES=$(python calib_exp/run_exp.py --method shap --train_size ${TRAIN_SIZE} --dataset ${DATASET} --do_bow --arg_n_tree 200 --arg_max_depth 20 2>/dev/null  | tail -1)
    echo 'BOW,'${BOW_RES}
    EXPL_RES=$(python calib_exp/run_exp.py --method shap --train_size ${TRAIN_SIZE} --dataset ${DATASET} --arg_n_tree 300 --arg_max_depth 20 2>/dev/null | tail -1)
    echo 'ShapCal,'${EXPL_RES}
}

RUN_SQUAD_EXP 500

