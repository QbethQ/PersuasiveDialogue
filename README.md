# PersuasiveDialogue
Code of LREC-COLING 2024 long paper "Would You Like to Make a Donation? A Dialogue System to Persuade You to Donate"

## Data
The PersuasionForGood dataset is from ***Wang et al., 2019. Persuasion for Good: Towards a Personalized Persuasive Dialogue System for Social Good.***. You can refer to it for more details about the dataset.

## Run
1. Run `preprocess.py` to preprocess the dataset
2. Run `main_bert.py` / `multilabel.py` to train and test a Persuasion Strategy Selection Module.
3. Run `train_gpt.py` to train a Natural Language Generation Module
4. Run `persuasiveness.py` to train and test a Persuasiveness Prediction Model
5. Run `eval.py` to evaluate the Natural Language Generation Module
6. Run `interact.py` to combine the Persuasion Strategy Selection Module and the Natural Language Generation Module into a persuasive dialogue system, and then you can talk to it!

## Citation
If you use any source codes included in this repository in your work, please cite our paper:
```
@inproceedings{song2024would,
    title={Would You Like to Make a Donation? A Dialogue System to Persuade You to Donate},
    author={Song, Yuhan and Wang, Houfeng},
    booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
    pages={17707--17717},
    year={2024}
}
```