# Research Notebook Collection

This folder contains two standalone Jupyter notebooks:

1. `Task 1_LLM_Finetuning_MedicalQA.ipynb`
2. `Task 2_RL_ChefsHat_RobustnessGeneralisation Code.ipynb`

Both notebooks are written as end-to-end experiments and are intended to be run in a Python notebook environment such as Google Colab or JupyterLab.

## Contents

- `Task 1_LLM_Finetuning_MedicalQA.ipynb`
  - Fine-tunes `google/flan-t5-base` on the `pubmed_qa` dataset.
  - Compares three approaches:
    - baseline pretrained model
    - full fine-tuning
    - LoRA fine-tuning
  - Evaluates with ROUGE and BLEU.
  - Saves comparison outputs such as `results.json` and training-loss plots.

- `Task 2_RL_ChefsHat_RobustnessGeneralisation Code.ipynb`
  - Trains a dueling DQN agent for the Chef’s Hat card game using `ChefsHatGym`.
  - Compares performance across:
    - multiple random seeds
    - different opponent types
    - epsilon-decay hyperparameters
  - Produces plots and training logs under the `outputs/` directory.

## Requirements

The notebooks install most dependencies themselves, but you will generally need:

- Python 3.10+
- Jupyter Notebook or Google Colab
- PyTorch
- TensorFlow
- `transformers`
- `datasets`
- `peft`
- `evaluate`
- `sacrebleu`
- `rouge_score`
- `accelerate`
- `bitsandbytes`
- `sentencepiece`
- `tqdm`
- `ChefsHatGym`
- `gym`
- `numpy`
- `pandas`
- `matplotlib`

## How to Run

### Task 1: Medical QA Fine-Tuning

1. Open `Task 1_LLM_Finetuning_MedicalQA.ipynb`.
2. Run the setup cell to install dependencies.
3. Execute the notebook top to bottom.
4. Review:
   - baseline evaluation
   - full fine-tuning results
   - LoRA results
   - comparison tables and charts

### Task 2: Chef’s Hat RL Robustness Study

1. Open `Task 2_RL_ChefsHat_RobustnessGeneralisation Code.ipynb`.
2. Run the installation cell.
3. Execute the notebook top to bottom.
4. Review:
   - baseline DQN training
   - seed robustness experiment
   - generalisation across opponent types
   - epsilon-decay sensitivity experiment

## Expected Outputs

### Task 1

- `loss_curves.png`
- `results.json`
- model checkpoints and trainer outputs created by Hugging Face Transformers

### Task 2

- `outputs/`
  - baseline training artifacts
  - seed experiment artifacts
  - generalisation experiment artifacts
  - hyperparameter sweep artifacts

## Notes

- Both notebooks assume internet access for package installation and dataset/model downloads.
- Task 1 is GPU-friendly and may be slow on CPU.
- Task 2 depends on the `ChefsHatGym` environment and its package structure.
- The notebooks are written as experiments, so some cells are long and include plotting, logging, and serialization steps.

## Project Summary

This repository combines two machine learning workflows:

- NLP fine-tuning for medical question answering
- Reinforcement learning robustness and generalisation analysis for a card-game agent

If you want, I can also add:

1. a shorter `README.md` for quick submission use
2. a `requirements.txt` generated from both notebooks
3. a cleaned notebook summary section for each file
