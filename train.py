from model import Classifier
import torch
from hparams import get_hparams
hparams = get_hparams()
from preprocessing.text_preprocessing import load_train_data, load_val_data
import os
import pickle
import json
import numpy as np
import argparse

def load_labels():
	f = open('models/labels.txt', 'r')
	labels= f.read().replace('\'','\"')
	labels=json.loads(labels)
	f.close()
	label_texts = []
	label_sequence=[]
	for k,v in labels.items():
		label_texts.append(k)
		label_sequence.append(v)
	return label_texts, label_sequence

def validation(valloader,model_path, words_count):
	labels_text, label_class_sequences = load_labels()
	model = Classifier(hparams=hparams, words_count=words_count)
	assert os.path.isfile(model_path),"model checkpoint must be a file"
	model, optimizer, learning_rate, iteration = load_checkpoint(model_path, model)
	model.eval()
	count=0
	with torch.no_grad():
		for i,(inputs,labels) in enumerate(valloader):
			pred_outputs = model(inputs)
			pred_label = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[0])
			true_label = get_pred_label(labels_text, np.asarray(label_class_sequences), labels[0])
			if pred_label == true_label:
				count+=1
			print(f'--- val: pred: {pred_label} -- true: {true_label}---')

	print(f'--- validation accuracy: {(count/(i+1))*100}%')


def train(dataloader,words_count,model_path=None):
	labels_text, label_class_sequences = load_labels()
	from random import randint
	model = Classifier(hparams=hparams, words_count=words_count)
	if model_path is None:
		optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
		iteration = 1
	else:
		model, optimizer, learning_rate, iteration = load_checkpoint(model_path, model)
	criterion = torch.nn.MSELoss()

	model.train()
	for epoch in range(hparams.epochs):
		print(f'------ Epoch: {epoch} -----')
		for inputs,labels in dataloader:
			model.zero_grad()
			pred_outputs = model(inputs)
			loss = criterion(pred_outputs, labels)
			loss.backward()
			optimizer.step()
			print(f'loss --- {loss}')
			rand = randint(0,50)
			if rand==5:
				pred_label = get_pred_label(labels_text, np.asarray(label_class_sequences), pred_outputs[0])
				true_label = get_pred_label(labels_text, np.asarray(label_class_sequences), labels[0])
				print(f'pred_label {pred_label} ---- true_label {true_label}')

			if iteration % 5000 == 0:
				checkpoint_path='checkpoint'
				os.makedirs(checkpoint_path, exist_ok=True)
				filepath=os.path.join(checkpoint_path,f'{iteration}_loss_{loss}')
				torch.save({'iteration': iteration,
							'state_dict': model.state_dict(),
							'optimizer': optimizer.state_dict(),
							'learning_rate': 0.0003}, filepath)

			iteration += 1

def get_pred_label(labels_text, label_class_sequences, pred_label):
	from torch.nn import functional as F
	import numpy as np
	## convert to torch tensor to use with F.mse_loss
	label_class_sequences = torch.from_numpy(label_class_sequences)
	mse_losses = []
	### find the loss of pred sequence with each class sequence
	for i, label in enumerate(label_class_sequences):
		mse_loss = F.mse_loss(label, pred_label)
		mse_losses.append(mse_loss)
	### take the minimum mse loss as predicted sequence
	matched_sequence = label_class_sequences[mse_losses.index(min(mse_losses))]
	### take the equivalent labels text as predicted label
	matched_label = labels_text[int(np.where(np.all(label_class_sequences.data.cpu().numpy() == matched_sequence.data.cpu().numpy(), axis=1))[0])]
	return matched_label

def load_checkpoint(checkpoint_path, model):
	assert os.path.isfile(checkpoint_path)
	checkpoint_dict = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint_dict['state_dict'])
	learning_rate = checkpoint_dict['learning_rate']
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	optimizer.load_state_dict(checkpoint_dict['optimizer'])
	iteration = checkpoint_dict['iteration']
	print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
	return model, optimizer, learning_rate, iteration




def get_args():
	args = argparse.ArgumentParser()
	args.add_argument('--mode', type=str, default="train", choices=['train','val','validation'], help="set mode for running")
	args.add_argument('--checkpoint', type=str, default=None, help="checkpoint to resume training/validating")
	args.add_argument('--val_file', type=str, default=None, help="validation data path")
	return args.parse_args()



if __name__=='__main__':
	args = get_args()
	model_path=args.checkpoint
	val_path = f'dataset/val.xlsx' if args.val_file is None else args.val_file
	if args.mode=='train':
		### training
		dataloader, words_count = load_train_data(filepath=f'dataset/train.xlsx', tokenizer=None)
		train(dataloader=dataloader,words_count=words_count,model_path=model_path)
	else:
		# loading tokenizer
		with open('models/tokenizer.pickle', 'rb') as handle:
			tokenizer = pickle.load(handle)
		assert model_path is not None, "checkpoint path is require when running validation model"
		assert model_path is not None, "checkpoint path is require when running validation model"
		words_count=len(tokenizer.word_counts)
		#### validation
		valloader, _ = load_val_data(filepath=val_path, tokenizer=tokenizer)
		validation(valloader, model_path=model_path, words_count=words_count)

#
# import torch
# from torch.nn import functional as F
# a = torch.tensor([20., 25.,  0.,  0.,  0.,  0.])
# b = torch.tensor([ 2.0272e+01,  2.5123e+01,  1.6699e+01, -1.5177e-01, -2.6956e-01,-5.0150e-04])
#
#

# score = F.mse_loss(a,b)
# print(score)
