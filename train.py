from model import Classifier
import torch
from hparams import get_hparams
hparams = get_hparams()
from preprocessing.text_preprocessing import load_data





def train():
	dataloader, words_count = load_data()
	model = Classifier(hparams=hparams, words_count=len(words_count))
	opimizer = torch.optim.Adam(model.parameters(), lr=0.003)
	criterion = torch.nn.CrossEntropyLoss()
	for epoch in range(hparams.epochs):
		print(f'------ Epoch: {epoch} -----')
		for inputs,outputs in dataloader:
			model.zero_grad()
			pred_outputs = model(inputs)
			print(pred_outputs.shape)
			loss = criterion(pred_outputs, outputs)
			loss.backward()
			opimizer.step()
			print(f'loss --- {loss}')

if __name__=='__main__':
	train()