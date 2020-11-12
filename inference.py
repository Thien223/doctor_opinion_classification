from preprocessing.text_preprocessing import hangul_preprocessing
from train import get_pred_label, load_labels, load_checkpoint
from hparams import get_hparams
hparams=get_hparams()
from model import Classifier
import os
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import torch, numpy as np
import pandas as pd
import argparse

def get_args():
	args = argparse.ArgumentParser()
	args.add_argument('--opinion', type=str, default=None, help="text of opinion to classification")
	args.add_argument('--opinions_path',type=str,default=None, help="path to opinions file")
	args.add_argument('--tokenizer',type=str, default='models/tokenizer.pickle', help="path to tokenizer model")
	args.add_argument('--checkpoint',type=str, default='checkpoint/15000_loss_0.045780032873153687', help="path to classification model")
	return args.parse_args()



def inference(model, opinion, tokenizer):
	model.eval()
	labels_text, label_class_sequences = load_labels()
	cleaned_opinions = [hangul_preprocessing(doctor_opinions=opinion.replace('\n','').strip())]
	opinions_sequences = tokenizer.texts_to_sequences(cleaned_opinions)
	max_input_sequence_length = hparams.opinions_max_length
	opinions_sequences = pad_sequences(opinions_sequences, maxlen=max_input_sequence_length, padding='post')
	opinions_sequences = torch.from_numpy(opinions_sequences).long()
	with torch.no_grad():
		pred_outputs = model(opinions_sequences)
		pred_label = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[0])
		print(f'=========')
		print(f'입력 소견: {opinion}')
		print(f'예측 레이블: {pred_label}')
		print(f'=========\n')
	return pred_label


if __name__=='__main__':
	#
	args = get_args()
	if args.opinions_path is None:
		assert args.opinion is not None, "you must pass either path to opinions file or opinion text to classify.."

	opinion=args.opinion
	opinions_path=args.opinions_path
	token_path = args.tokenizer
	model_path = args.checkpoint
	assert os.path.isfile(token_path), "model checkpoint must be a file"
	with open(token_path, 'rb') as handle:
		tokenizer = pickle.load(handle)


	if args.opinions_path is not None:

		#### classifying multi opinions from file
		model = Classifier(hparams=hparams, words_count=len(tokenizer.word_counts))
		assert os.path.isfile(model_path), "model checkpoint must be a file"
		model, optimizer, learning_rate, iteration = load_checkpoint(model_path, model)

		opinions_df = pd.read_excel(opinions_path)
		opinions = list(opinions_df['소견'])
		results = [[]]
		for i,opinion in enumerate(opinions):
			pred_label = inference(model=model, opinion=opinion, tokenizer=tokenizer)
			results.append([opinion,pred_label])
		result_df = pd.DataFrame(results, columns={'예측 레이블','소견'})
		result_df.to_excel("result.xlsx")

	else:
		# ####### 1 opinion classification at a time
		model = Classifier(hparams=hparams, words_count=len(tokenizer.word_counts))
		assert os.path.isfile(model_path),"model checkpoint must be a file"
		model, optimizer, learning_rate, iteration = load_checkpoint(model_path, model)
		inference(model=model, opinion=opinion, tokenizer=tokenizer)

