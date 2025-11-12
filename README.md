# FACTGUARD: Event-Centric and Commonsense-Guided Fake News Detection.

## Dataset

The experimental datasets should be placed in the data folder after your application is approved.
The original datasets are from the paper: ["**Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection**"](https://arxiv.org/abs/2309.12247), which has been accepted by AAAI 2024. We cannot provide the original datasets because they were not collected by us and we are not authorized to distribute them. We only provide how we extract Topic and Content. Note that you can download the datasets only after an ["Application to Use the Datasets from ARG for Fake News Detection"](https://forms.office.com/r/DfVwbsbVyM) has been submitted. Please visit the links provided to obtain the original datasets and cite the ARG paper.

## Code

### Requirements

- python==3.8.20
- CUDA: 11.7
- Python Packages:
  ```
  pip install -r requirements.txt
  ```

### Pretrained Models

You can download pretrained models ([roberta-base](https://huggingface.co/FacebookAI/roberta-base) and [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)) and change paths (`bert_path`) in the corresponding scripts.

### Run

You can run this model through `run_zh.sh` for FACTGUARD-zh and `run_en.sh` for FACTGUARD-en. 

### Data preprocessing

After you apply for the datasets, please replace the content of the "cs_rationale" field in the dataset with the Topic and Content extracted by an LLM using the prompt engineering in the paper, and change all the "cs_acc" fields to 0. The FACTGUARD training process uses six fields: `content`, `label`, `td_rationale`, `cs_rationale`, `cs_pred`, and `cs_acc`, which correspond to n, y, c, r, y_llm, and 0 in the paper.

### Code introduction

All models and training codes used are stored under the `models` folder:
- `factguard.py` is the main model code;
- `factguardd.py` is the distillation model code;
- `layers.py` contains layers and modules used in the model training process;
- `factguard1.py` only retains the ablation model of the original news;
- `factguard2.py` only retains the ablation model of the news topic and content extracted by LLM;
- `factguard3.py` only retains the ablation model of LLM's commonsense reasoning on the news;
- `factguard1_2.py` removes the ablation model of LLM's commonsense reasoning;
- `factguard1_3.py` removes the ablation model of the Topic and Content of the original news extracted by LLM;
- `factguard2_3.py` removes the ablation model of the original news;
- `factguard_without_llm_judge.py` does not use the ablation model of the large-model usability judgment;
- `roberta.py` is the implementation code for one of the baseline experiments.
