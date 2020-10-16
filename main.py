
import fasttext
model = fasttext.load_model('pretrained_words_vectors/fasttext/cc.ko.300.bin') ## because the model is huge, loading it before calling the function to save loading time

from nltk.translate.bleu_score import sentence_bleu


def similarity_calculate(index_word, to_compare_word, model):
	'''
	compare similarity between 색인어 and keyword (from 소견)
	:param index_word: 저장 된 색인어
	:param to_compare_word: keyword from 소견
	:param model: fasttext model (because model loading takes time, we load it into memory and call from function)
	:return: similarity score, less is better
	'''
	import scipy.spatial.distance as distance
	index_word_vec = model.get_word_vector(index_word)
	to_compare_word_vec = model.get_word_vector(to_compare_word)
	return distance.cosine(index_word_vec, to_compare_word_vec)


def doctor_opinion_split(doctoral_opinion):
	'''
	split 의사소견 to 질환 소견
	:param doctoral_opinion: 검진소견
	:return: list of 소견 by 질환
	'''
	dease_opinions = doctoral_opinion.replace('\n','').split('*')
	return dease_opinions[1:]

#dease_opinion=dease_opinions[2]
def dease_index_words_extract(dease_opinion):
	'''
	Get the splitted 소견 and extract the keywords
	:param dease_opinion: 질환에 대한 의사 소견
	:return: list of keywords
	'''
	from konlpy.tag import Okt
	open_korean_text = Okt()
	### get the complex nouns
	# nouns = open_korean_text.nouns(phrase='위내시경 검사결과 역류성 식도염 소견입니다. - 역류성 식도염은 흡연, 음주, 커피, 기름진 음식, 야식 등이 주된 원인입니다. 치료여부는 식도염 정도와 증상에 따라 달라지므로 내과 전문의 상담 권합니다.')
	# print(nouns)

	### get the complex nouns and nouns
	keywords = open_korean_text.phrases(phrase=dease_opinion)
	print(keywords)
	### filter out the nouns to keep only complex nouns

	# keywords = list(set(phrases) - set(nouns)) ### remove 명사, keep only 숙어
	return keywords

def get_index_words():
	'''
	get 색인어 from file
	:return: list of 색인어
	'''
	index_words = []
	with open('index_words.txt', 'r') as f:
		for line in f.readlines():
			index_words.append(line.strip())
	return index_words


def extract_index_words(keywords):
	'''
	calculate similarity between filtered_phrase (keywords) and 저장된 색인어, return best matched 색인어
	:param filtered_phrases: keywords extracted from 소견
	:return: best matched 색인어
	'''
	import numpy as np
	matched_index_words=[]
	### get 색인어
	index_words = get_index_words()
	similar_scores = np.zeros(shape=(len(index_words),len(keywords)))


	for j, keyword in enumerate(keywords):
		for i, index_word in enumerate(index_words):
			### calculate similarity using bleu score
			# similar_scores[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)
			## calculate similarity using cosine similarity
			similar_scores[i,j] = similarity_calculate(index_word.replace(' ',''), keyword.replace(' ',''), model=model) if str(similarity_calculate(index_word, keyword, model=model))!='nan' else 1.0

	for idx in np.argwhere(similar_scores == similar_scores.max()):
		matched_index_words.append(index_words[idx[0]])
		print(f'색인어 --- {index_words[idx[0]]}')
		print(f'키워드 --- {keywords[idx[1]]}')
		print('############### \n')
	return list(set(matched_index_words))








import pandas as pd
opinions = pd.read_excel('dataset/종합검진소견.xlsx')
dease_info = pd.read_excel('dataset/질환정보.xlsx')


for row in dease_info.itertuples():
	if row[7] =='Y':
		try:
			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
		except IndexError as e:
			print(e)
			continue
		keywords = ['1일', '하루']
		for word in keywords:
			if word in opinion:
				print(row[1])
				file = open(f'{row[1]}_{word}.txt','w')
				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 1일\n\n\n소견에서 찾아되는 시간: {word}')
	if row[8] =='Y':
		try:
			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
		except IndexError as e:
			print(e)
			continue
		keywords = ['1개월', '1달', '일개월', '한달','1 개월', '1 개 월', '1개 월']
		for word in keywords:
			if word in opinion:
				print(row[1])
				file = open(f'{row[1]}_{word}.txt','w')
				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 1개월\n\n\n소견에서 찾아되는 시간: {word}')
	if row[9] =='Y':
		try:
			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
		except IndexError as e:
			print(e)
			continue
		keywords = ['3개월', '3달', '삼개월', '세달','3 개월', '3 개 월', '3개 월']
		for word in keywords:
			if word in opinion:
				print(row[1])
				file = open(f'{row[1]}_{word}.txt','w')
				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 3개월\n\n\n소견에서 찾아되는 시간: {word}')
	if row[10] =='Y':
		try:
			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
		except IndexError as e:
			print(e)
			continue
		keywords = ['6개월', '6달', '육개월', '여섯달','6 개월', '6 개 월', '6개 월']
		for word in keywords:
			if word in opinion:
				print(row[1])
				file = open(f'{row[1]}_{word}.txt','w')
				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 6개월\n\n\n소견에서 찾아되는 시간: {word}')
	if row[11] =='Y':
		try:
			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
		except IndexError as e:
			print(e)
			continue
		keywords = ['1년']
		for word in keywords:
			if word in opinion:
				print(row[1])
				file = open(f'{row[1]}_{word}.txt','w')
				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 1년\n\n\n소견에서 찾아되는 시간: {word}')

