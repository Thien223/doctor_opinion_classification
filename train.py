from model import Classifier
import torch
from hparams import get_hparams
hparams = get_hparams()
from preprocessing.text_preprocessing import load_data
from torch.nn import functional as F




def train(dataloader,words_count):
	model = Classifier(hparams=hparams, words_count=len(words_count))
	opimizer = torch.optim.Adam(model.parameters(), lr=0.003)
	criterion = torch.nn.MSELoss()
	for epoch in range(hparams.epochs):
		print(f'------ Epoch: {epoch} -----')
		for inputs,outputs in dataloader:
			model.zero_grad()
			pred_outputs = model(inputs)
			loss = criterion(pred_outputs, outputs)
			loss.backward()
			opimizer.step()
			print(f'loss --- {loss}')
		print(pred_outputs)



def val(valloader)


if __name__=='__main__':

	dataloader, words_count = load_data()


	for inputs, outputs in dataloader:
		print(outputs)
		break
	train(dataloader,words_count)
