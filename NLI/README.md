# Calibrating Black-Box NLI Models

## Dependency and Project Structure
1. First make sure to install the dependency (as specified in `requirement.txt`) in the parent directory.
2. Please download spacy `en_core_web_sm` resource file via `python -m spacy download en_core_web_sm`.
3. Prepare some necessary directories: `mkdir -p cached data outputs checkpoints misc ` 

## Preparing Model and Data
Download the data and model from [https://utexas.box.com/s/out3mneykgjacrc60epw5im1f5pymz2c](https://utexas.box.com/s/out3mneykgjacrc60epw5im1f5pymz2c). Put the model as `checkpoints/mnli_roberta-base`. Put the data files under `outputs` directory.

## Explanaining using Lime/Shap

**Running inference**

`CUDA_VISIBLE_DEVICES=0 sh run_nli.sh eval mnli mrpc # or qnli`

You should see an accuracy of 57.3 for MRPC (50.5 for QNLI).

Please copy the predictions at `predictions/mnli/predictions.json` to `misc/mnli_mrpc_predictions.json` (`misc/mnli_qnli_predictions.json`). This generates be the predictions we are going to calibrate.

**Generating interpretations**. This will generate interprations (bin files) under `interpretations` directory.

`CUDA_VISIBLE_DEVICES=0 sh run_interp.sh mnli shap run mrpc # or qnli`

**Visualizing interpretations**. We provide a simple visualization tool for you to view the explanations.

`sh run_interp.sh squad shap vis mrpc # qnli`


## Calibrating

**Running NER tagger**

`python calib_exp/run_tagger.py --split mrpc # or qnli`


**Extracting features**

`python calib_exp/make_calib_dataset.py --method shap --split mrpc # or lime for method, or qnli for split`

**Running calibrators**

See examples in `exp_scripts/run_main_exp.sh`.

`sh exp_scripts/run_main_exp.sh`
