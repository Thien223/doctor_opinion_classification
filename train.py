from model import Classifier
import torch
from hparams import get_hparams
hparams = get_hparams()
from preprocessing.text_preprocessing import load_data
from torch.nn import functional as F
import os
import pickle


def train(dataloader,words_count,labels_text, label_class_sequences,model_path=None):
	from random import randint
	model = Classifier(hparams=hparams, words_count=words_count)
	if model_path is None:
		optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
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
				pred_label = get_pred_label(labels_text, label_class_sequences, pred_outputs[0])
				true_label = get_pred_label(labels_text, label_class_sequences, labels[0])
				print(f'pred_label {pred_label} ---- true_label {true_label}')

			if iteration % 5000 == 0:
				filepath=f'checkpoint/{iteration}_loss_{loss}'
				torch.save({'iteration': iteration,
							'state_dict': model.state_dict(),
							'optimizer': optimizer.state_dict(),
							'learning_rate': 0.003}, filepath)

			iteration += 1

def get_pred_label(labels_text, label_class_sequences, pred_label):
	from torch.nn import functional as F
	import numpy as np
	label_class_sequences = torch.from_numpy(label_class_sequences)
	mse_losses = []
	for i, label in enumerate(label_class_sequences):
		mse_loss = F.mse_loss(label, pred_label)
		mse_losses.append(mse_loss)
	matched_sequence = label_class_sequences[mse_losses.index(min(mse_losses))]
	matched_label = labels_text[int(np.where(np.all(label_class_sequences.data.cpu().numpy() == matched_sequence.data.cpu().numpy(), axis=1))[0])]
	return matched_label

def load_checkpoint(checkpoint_path, model):
	assert os.path.isfile(checkpoint_path)
	print("Loading checkpoint '{}'".format(checkpoint_path))
	checkpoint_dict = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint_dict['state_dict'])
	learning_rate = checkpoint_dict['learning_rate']
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	optimizer.load_state_dict(checkpoint_dict['optimizer'])
	iteration = checkpoint_dict['iteration']
	print("Loaded checkpoint '{}' from iteration {}".format(checkpoint_path, iteration))
	return model, optimizer, learning_rate, iteration

if __name__=='__main__':
	# loading
	with open('models/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	dataloader, words_count, labels_text, label_class_sequences = load_data(filepath=f'dataset/opinions.xlsx', tokenizer=None)

	train(dataloader, words_count, labels_text =labels_text, label_class_sequences=label_class_sequences, model_path=f'checkpoint/10000_loss_0.07650874555110931')

#
# import torch
# from torch.nn import functional as F
# a = torch.tensor([20., 25.,  0.,  0.,  0.,  0.])
# b = torch.tensor([ 2.0272e+01,  2.5123e+01,  1.6699e+01, -1.5177e-01, -2.6956e-01,-5.0150e-04])
#
#

# score = F.mse_loss(a,b)
# print(score)
