from model import Classifier
import torch
from hparams import get_hparams
hparams = get_hparams()
from preprocessing.text_preprocessing import load_data
from torch.nn import functional as F




def train(dataloader,words_count):
	from random import randint
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
			rand = randint(0,50)
			if rand==5:
				print(f'output[[0]] {outputs[0]} ---- pred[0] {pred_outputs[0]}')



if __name__=='__main__':

	dataloader, words_count = load_data()


	train(dataloader,words_count)


import torch
from torch.nn import functional as F
a = torch.tensor([20., 25.,  0.,  0.,  0.,  0.])
b = torch.tensor([ 2.0272e+01,  2.5123e+01,  1.6699e+01, -1.5177e-01, -2.6956e-01,-5.0150e-04])



score = F.mse_loss(a,b)
print(score)
