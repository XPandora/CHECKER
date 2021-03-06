# CHECKER
Codebase and dataset for *CHECKER: Detecting Clickbait Thumbnails with Weak Supervision and Co-Teaching*.

## Setups

The basic environment is as follows
- snorkel==0.9.3
- pytorch==1.5.0
- torchvision==0.6.0
- gensim==4.0.0b0
- block.bootstrap.pytorch==0.1.6

The used GloVe word vectors can be directly downloaded from: https://drive.google.com/file/d/1R-K4PoPll7GCJ2d92dWcIixQtLJcx7At/view?usp=sharing

## Data
- Dataset with feature engineering results is provided in Data/Dataset_with_feature.csv. Use the following code you can have a brief view:

```python
import pandas as pd
df = pd.read_csv('Data/Dataset_with_feature.csv', lineterminator='\n')
print(df.head())
```
- Thumbnails are available at: https://drive.google.com/file/d/1jaJW3atiflkSx_IRhOojh8MefJ9UjyJU/view?usp=sharing


## Generate Labels:
You can generate labels using the following command:
```shell
python label_generator.py
```
The generated labels would be available at `Data/unlabeled_data/train_generated_label.csv`

## Model Training:
You can train model using the following command:
```shell
python main.py --train_path Data/labeled_data/train.csv --vocab_path {path_to_GloVe} --prob_data Data/unlabeled_data/train_generated_label.csv --test_path Data/labeled_data/test.csv --text_model_type glove --fusion_layer Mutan --val_index 0 --batch_size 32 --lr 1e-4 --exp_name Mutan_glove_val_0_with_prob_data_forget_0 --result_dir ./results/ --forget_rate 0 --num_gradual 10 --incongruent_rate 0 --n_epoch 20 --thumbnail_folder {path_to_thumbnail_folder}
```

Note that for evaluting a model, we train it with `val_index=0,1,2,3,4` respectively and use the averaged result as the final perfomrance.