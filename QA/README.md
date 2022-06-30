# Calibrating Black-Box QA Models

## Dependency and Project Structure
1. First make sure to install the dependency (as specified in `requirement.txt`) in the parent directory.
2. Please download spacy `en_core_web_sm` resource file via `python -m spacy download en_core_web_sm`.
3. Prepare some necessary directories: `mkdir -p cached data outputs checkpoints misc ` 

## Preparing Model and Data
We include a trained SQuAD RoBERTa-based model here
https://utexas.box.com/s/zdee0cuz28c6xgj2hz765uy1tb9syafc. Unzip and put the model as `checkpoints/squad_roberta-base`.

We converted the SQuAD-Adv, TriviaQA, and HotpotQA data (https://utexas.box.com/s/ngsf9bcalxwjlm8y5kftlzicjkq3wk1j) into the SQuAD format. Unzip and put them under `outputs` directory.

## Explanaining using Lime/Shap

**Running inference**

`CUDA_VISIBLE_DEVICES=0 sh run_qa.sh eval squad addsent-dev # or triva/hotpot`

Please copy the nbest predictions at `predictions/squad/nbest_predictions_.json` to `misc/addsent-dev_squad_predictions.json` (`misc/dev_triva_predictions.json` or `misc/dev_hotpot_predictions.json`) accordingly. This generates be the predictions we are going to calibrate.

**Generating interpretations**. This will generate interprations (bin files) under `interpretations` directory.

`CUDA_VISIBLE_DEVICES=0 sh run_interp.sh squad shap run addsent-dev # or triva/hotpot`

**Visualizing interpretations**. We provide a simple visualization tool for you to view the explanations.

`sh run_interp.sh squad shap vis addsent-dev # or triva/hotpot`


## Calibrating

**Running NER tagger**

`python calib_exp/run_tagger.py --dataset squad # --dataset hotpot or --dataset trivia`


**Extracting features**

`python calib_exp/make_calib_dataset.py --dataset squad # --dataset hotpot or --dataset trivia`

**Running calibrators**

See examples in `exp_scripts/run_main_exp.sh`.

`sh exp_scripts/run_main_exp.sh`
