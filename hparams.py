class Hparams():
	opinions_max_length=100
	label_max_length=6
	linear_layer_input_dim=8
	batch_size=8
	epochs=50
def get_hparams():
	hparams = Hparams()
	return hparams