import torch
from torch.nn import functional as F
class LinearNorm(torch.nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
		super(LinearNorm, self).__init__()
		self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

		torch.nn.init.xavier_uniform_(
			self.linear_layer.weight,
			gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, x):
		return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
				 padding=None, dilation=1, bias=True, w_init_gain='linear'):
		super(ConvNorm, self).__init__()
		if padding is None:
			assert(kernel_size % 2 == 1)
			padding = int(dilation * (kernel_size - 1) / 2)

		self.conv = torch.nn.Conv1d(in_channels, out_channels,
									kernel_size=kernel_size, stride=stride,
									padding=padding, dilation=dilation,
									bias=bias)

		torch.nn.init.xavier_uniform_(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

	def forward(self, signal):
		conv_signal = self.conv(signal)
		return conv_signal



class Inception(torch.nn.Module):
	def __init__(self, hparams):
		super(Inception, self).__init__()
		self.conv1d_10 = ConvNorm(in_channels=32, out_channels=hparams.linear_layer_input_dim, kernel_size=5, stride=1)
		self.conv1d_20 = ConvNorm(in_channels=32, out_channels=hparams.linear_layer_input_dim, kernel_size=11, stride=1)
		self.conv1d_40 = ConvNorm(in_channels=32, out_channels=hparams.linear_layer_input_dim, kernel_size=21, stride=1)
		#### residual_conv and bottleneck convolution must match the inputs shape
		self.bottleneck = ConvNorm(in_channels=8, out_channels=32, kernel_size=1, stride=1)
		self.residual_conv = ConvNorm(in_channels=8, out_channels=hparams.linear_layer_input_dim, kernel_size=1, stride=1)
		self.max_pooling = torch.nn.MaxPool1d(kernel_size=3, stride=1)

	def forward(self, inputs):
		print(f'inception input shape: {inputs.shape}')
		pool_out = self.max_pooling(inputs)

		residual_out = self.residual_conv(pool_out)
		bottleneck_output = self.bottleneck(inputs)

		conv_10_out = self.conv1d_10(bottleneck_output)
		conv_20_out = self.conv1d_20(bottleneck_output)
		conv_40_out = self.conv1d_40(bottleneck_output)
		conv_outouts = torch.cat((residual_out,conv_10_out,conv_20_out,conv_40_out), dim=2)
		print(f'inception conv_outouts shape: {conv_outouts.shape}')
		return conv_outouts


class Classifier(torch.nn.Module):
	def __init__(self, hparams, words_count):
		super(Classifier, self).__init__()
		self.hparams = hparams
		self.inception = Inception(hparams)
		self.linear = LinearNorm(in_dim=65456, out_dim=6)
		self.conv1d = ConvNorm(in_channels=100, out_channels=8, kernel_size=3, stride=1, w_init_gain='relu')
		self.softmax = torch.nn.Softmax(dim=1)
		self.embedding = torch.nn.Embedding(embedding_dim=512, num_embeddings=words_count+10)
		self.batch_norm = torch.nn.BatchNorm1d(hparams.linear_layer_input_dim)

	def forward(self, inputs):
		try:
			embedded = self.embedding(inputs)
		except Exception as e:
			print(e)
			print(inputs.shape)
			print(self.embedding)
		conv1d_out = F.dropout(F.relu(self.batch_norm(self.conv1d(embedded)), inplace=True),0.5)

		incept_1_out = F.dropout(F.relu(self.batch_norm(self.inception(conv1d_out)), inplace=True),0.5)

		incept_2_out = F.dropout(F.relu(self.batch_norm(self.inception(incept_1_out)), inplace=True),0.5)
		incept_2_out = incept_2_out.view(incept_2_out.size(0), -1)

		linear_out = self.linear(incept_2_out)

		# output = self.softmax(linear_out)
		return linear_out

# inputs = torch.zeros(torch.Size((32,100)), dtype=torch.long)
# from hparams import get_hparams
# hparams = get_hparams()
# classifier = Classifier(hparams=hparams, words_count=932)
# classifier(inputs)