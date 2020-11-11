from preprocessing.text_preprocessing import hangul_preprocessing
from train import get_pred_label, load_labels, load_checkpoint
from hparams import get_hparams
hparams=get_hparams()
from model import Classifier
import os
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import torch, numpy as np


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
		print(f'predicted label: {pred_label}')



if __name__=='__main__':

	opinion='''좌측 신장에 2.1`cm 크기의 단순 낭종(simple cyst)이 1개 있으며
	임상적 의미가 없는 병변임.
	담도 확장 없으며 간, 담낭, 우측 신장, 췌장, 비장에 특이소견 안보임.
	 
	결론 : 좌측 신장 낭종.
	
	'''
	token_path='models/tokenizer.pickle'
	model_path='checkpoint/15000_loss_0.045780032873153687'

	with open(token_path, 'rb') as handle:
		tokenizer = pickle.load(handle)

	model = Classifier(hparams=hparams, words_count=len(tokenizer.word_counts))
	assert os.path.isfile(model_path),"model checkpoint must be a file"
	model, optimizer, learning_rate, iteration = load_checkpoint(model_path, model)
	inference(model_path=model_path, opinion=opinion, tokenizer=tokenizer)