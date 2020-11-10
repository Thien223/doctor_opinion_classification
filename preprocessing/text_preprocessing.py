
import torch
import re
from tqdm import tqdm
import pandas as pd
from hparams import get_hparams
hparams = get_hparams()
def data_generator(opinions_sequences, labels_sequences, batch_size=8):
	from torch.utils.data import TensorDataset, DataLoader
	input_sequence = torch.tensor(opinions_sequences, dtype=torch.long)
	output_sequence = torch.tensor(labels_sequences, dtype=torch.float)
	dataset = TensorDataset(input_sequence, output_sequence)
	dataloader = DataLoader(dataset, batch_size=batch_size)
	return dataloader

def hangul_preprocessing(doctor_opinions, remove_stopwords=False, stopwords=None):
	'''
	process hangul to list of words
	:param doctor_opinions: documents to analysis
	:param remove_stopwords: whether remove the stopwords or not
	:param stopwords: list of stopwords to remove
	:return: list of words in document
	'''
	doctor_opinions_cleaned=None
	try:
		if str(doctor_opinions)!='nan':
			stop_words = {'은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한'}
			from konlpy.tag import Okt
			## remove all non-hangul and space chars
			doctor_opinions_cleaned = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", doctor_opinions)
			open_korean_text = Okt()
			## extract words
			doctor_opinions_cleaned = open_korean_text.morphs(doctor_opinions_cleaned, stem=True)
			### remove stopwords in words list
			if remove_stopwords:
				assert stopwords is not None, 'when remove_stopwords is True, require stopwords list'
				doctor_opinions_cleaned = [token for token in doctor_opinions_cleaned if not token in stop_words]
	except:
		print(doctor_opinions)
	return doctor_opinions_cleaned


def words_transform(opinionslist, labelslist):
	'''
	transform words in  to sequences
	:param wordslist: list of all processed opinions
	:return: sequences lists of each opinions
	'''
	from tensorflow.python.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras.preprocessing.text import Tokenizer
	max_input_sequence_length = hparams.opinions_max_length
	max_output_sequence_length = hparams.label_max_length
	wordslist = opinionslist + labelslist
	tokenizer = Tokenizer()
	### train tokenizer
	tokenizer.fit_on_texts(wordslist)
	opinions_sequences = tokenizer.texts_to_sequences(opinionslist)
	labels_sequences = tokenizer.texts_to_sequences(labelslist)
	word_counts = tokenizer.word_counts

	opinions_sequences = pad_sequences(opinions_sequences, maxlen=max_input_sequence_length, padding='post')
	labels_sequences = pad_sequences(labels_sequences, maxlen=max_output_sequence_length, padding='post')

	return opinions_sequences, labels_sequences, word_counts


def load_data(filepath=f'dataset/opinions.xlsx'):
	doctor_opinions = pd.read_excel(filepath)
	doctor_opinions.drop_duplicates(keep="first", inplace=True)

	### make dataset balanced
	df_1 = doctor_opinions.loc[doctor_opinions['처방']=='상복부초음파']
	df_2 = doctor_opinions.loc[doctor_opinions['처방']=='흉부촬영(PA)']
	df_3 = doctor_opinions.loc[doctor_opinions['처방']=='위내시경']

	max_count = min(len(df_1), len(df_2), len(df_3))
	df_1=df_1.iloc[:max_count]
	df_2=df_2.iloc[:max_count]
	df_3=df_3.iloc[:max_count]

	doctor_opinions_df = df_1.append(df_2).append(df_3)

	### random shufling
	doctor_opinions_df = doctor_opinions_df.sample(frac=1)
	opinions = list(doctor_opinions_df['소견'])
	labels = list(doctor_opinions_df['처방'])


	cleaned_opinions = []
	cleaned_labels = []
	for opinion, label in tqdm(zip(opinions, labels)):
		cleaned_opinion = hangul_preprocessing(doctor_opinions=opinion.replace('\n','').strip())
		cleaned_label = hangul_preprocessing(doctor_opinions=label.strip())
		cleaned_opinions.append(cleaned_opinion)
		cleaned_labels.append(cleaned_label)
	opinions_sequences, labels_sequences, words_count = words_transform(cleaned_opinions, cleaned_labels)
	dataloader = data_generator(opinions_sequences, labels_sequences, batch_size=hparams.batch_size)
	return dataloader, words_count