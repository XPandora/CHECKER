from snorkel.labeling import labeling_function, LFAnalysis, PandasLFApplier, LabelingFunction, filter_unlabeled_dataframe
from snorkel.analysis import get_label_buckets
from snorkel.preprocess import preprocessor
from snorkel.labeling import LabelModel, MajorityLabelVoter
from snorkel.utils import probs_to_preds
from textblob import TextBlob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re
import argparse

ABSTAIN = -1
NONCLICKBAIT = 0
CLICKBAIT = 1

parser = argparse.ArgumentParser()
parser.add_argument('--train_ws_path', type=str, default='Data/unlabeled_data/unlabel_data_ws.csv')
parser.add_argument('--test_ws_path', type=str, default='Data/labeled_data/labeled_data_ws.csv')
parser.add_argument('--feature_path', type=str, default='Data/Dataset_with_feature.csv')
parser.add_argument('--output_path', type=str, default='Data/unlabeled_data/train_generated_label.csv')
args = parser.parse_args()

df_train = pd.read_csv(args.train_ws_path, lineterminator='\n')
df_val = pd.read_csv(args.test_ws_path, lineterminator='\n')
df_train["thumbnail_text"] = df_train["thumbnail_text"].apply(str)
df_val["thumbnail_text"] = df_val["thumbnail_text"].apply(str)
df_val["label"] = df_val["label"].apply(lambda x: 1 if x == "clickbait" else 0)
Y_val = df_val.label.values


@preprocessor(memoize=True)
def textblob_title_sentiment(x):
	scores = TextBlob(x.title)
	x.title_polarity = scores.sentiment.polarity
	x.title_subjectivity = scores.sentiment.subjectivity
	return x

@preprocessor(memoize=True)
def textblob_tb_text_sentiment(x):
	tb_text= x.thumbnail_text
	if tb_text == 0:
		x.tb_text_polarity = 'none'
		x.tb_text_subjectivity = 'none'
	else:
        	scores = TextBlob(tb_text)
        	x.tb_text_polarity = scores.sentiment.polarity
        	x.tb_text_subjectivity = scores.sentiment.subjectivity
	return x

@labeling_function()
def lf_channel_nonclickbait_label(x):
	return NONCLICKBAIT if x.channel_label_y == "nonclickbait" else ABSTAIN

@labeling_function()
def lf1_channel_clickbait_label(x):
	return CLICKBAIT if x.channel_label_y == "clickbait" and x.text_area > 0.01  else ABSTAIN

@labeling_function(pre=[textblob_title_sentiment])
def lf2_channel_clickbait_label(x):
	return CLICKBAIT if x.channel_label_y == "clickbait" and x.viewCount / (x.commentCount**0.5)  > 41000  else ABSTAIN 


@labeling_function()
def lf_thumbnail_label(x):
	return NONCLICKBAIT if x.thumbnail_label == "nonclickbait" else ABSTAIN

@labeling_function()
def lf_description_contains_nolink(x):
	return NONCLICKBAIT if "http" not in str(x.description).lower() else ABSTAIN

@labeling_function(pre=[textblob_title_sentiment])
def lf_title_subjectivity(x):
	return NONCLICKBAIT if x.title_subjectivity < 0.05 else ABSTAIN


lfs = [lf1_channel_clickbait_label,lf2_channel_clickbait_label,lf_channel_nonclickbait_label,lf_title_subjectivity,lf_thumbnail_label,lf_description_contains_nolink]


applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df = df_train)
L_val = applier.apply(df=df_val)

print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())
print(LFAnalysis(L=L_val, lfs=lfs).lf_summary(Y=Y_val))

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, lr = 0.001, log_freq=20, seed=321)
majority_model = MajorityLabelVoter()

label_model_acc = label_model.score(L=L_val, Y=Y_val)["accuracy"]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
label_model_metric = label_model.score(L=L_val, Y=Y_val, metrics=["f1","precision", "recall"])
print(f"{'Label Model F1 score:':<25} {label_model_metric['f1'] * 100:.1f}%")
print(f"{'Label Model Precision:':<25} {label_model_metric['precision'] * 100:.1f}%")
print(f"{'Label Model Recall:':<25} {label_model_metric['recall'] * 100:.1f}%")

majority_acc = majority_model.score(L=L_val, Y=Y_val)["accuracy"]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")
majority_metric = majority_model.score(L=L_val, Y=Y_val, metrics=["f1", "precision", "recall"])
print(f"{'Majority Vote F1 score:':<25} {majority_metric['f1'] * 100:.1f}%")
print(f"{'Majority Vote Precision:':<25} {majority_metric['precision'] * 100:.1f}%")
print(f"{'Majority Vote Recall:':<25} {majority_metric['recall'] * 100:.1f}%")

probs_train = majority_model.predict_proba(L = L_train)
probs_val = majority_model.predict_proba(L = L_val)

df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)

df_val_filtered, probs_val_filtered = filter_unlabeled_dataframe(
	X = df_val, y = probs_val, L = L_val
)

preds_val = probs_to_preds(probs=probs_val)

df_train_filtered["soft_label"] = [i[1] for i in probs_train_filtered]
df_train_filtered["hard_label"] = df_train_filtered["soft_label"].apply(lambda x: "clickbait" if x>=0.5 else "nonclickbait" if x< 0.5 else "unclear")
feature = pd.read_csv(args.feature_path, lineterminator='\n')[['ID', 'title']]

hard_train_set = df_train_filtered[["ID","hard_label"]]
hard_train_set.columns = ["ID", "label"]
hard_train_set = hard_train_set.merge(feature,on="ID",how="inner")
hard_train_set.to_csv(args.output_path, index=False)