
from nltk.translate.bleu_score import sentence_bleu


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


class Detector():
	def __init__(self):
		self.index_words = get_index_words()

	def similarity_calculate(self, index_word, to_compare_word, model):
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


	def doctor_opinion_split(self, doctoral_opinion):
		'''
		split 의사소견 to 질환 소견
		:param doctoral_opinion: 검진소견
		:return: list of 소견 by 질환
		'''
		dease_opinions = doctoral_opinion.split('*')
		return dease_opinions[1:]


	#dease_opinion=dease_opinions[2]
	def dease_index_words_extract(self, dease_opinion):
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
		### filter out the nouns to keep only complex nouns

		# keywords = list(set(phrases) - set(nouns)) ### remove 명사, keep only 숙어
		return keywords



	def extract_index_words(self, keywords):
		'''
		calculate similarity between filtered_phrase (keywords) and 저장된 색인어, return best matched 색인어
		:param filtered_phrases: keywords extracted from 소견
		:return: best matched 색인어
		'''
		import numpy as np
		matched_index_words=[]
		### get 색인어
		similar_scores = np.zeros(shape=(len(self.index_words),len(keywords)))


		for j, keyword in enumerate(keywords):
			for i, index_word in enumerate(self.index_words):
				### calculate similarity using bleu score
				similar_scores[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)
				## calculate similarity using cosine similarity
				# similar_scores[i,j] = similarity_calculate(index_word.replace(' ',''), keyword.replace(' ',''), model=model) if str(similarity_calculate(index_word, keyword, model=model))!='nan' else 1.0

		for idx in np.argwhere(similar_scores == similar_scores.max()):
			matched_index_words.append(self.index_words[idx[0]])
		return matched_index_words[-1]



	def need_recheck(self, opinion):
		time_keywords = ['1일'] +['1개월', '1달', '일개월', '한달','1 개월', '1 개 월', '1개 월']+['3개월', '3달', '삼개월', '세달','3 개월', '3 개 월', '3개 월']+['6개월', '6달', '육개월', '여섯달','6 개월', '6 개 월', '6개 월']+['1년', '2년']
		needed=False
		for time in time_keywords:
			if time in opinion:
				needed = True
		return needed

	def __call__(self, doctor_opinions):
		splitted_opinions = self.doctor_opinion_split(doctoral_opinion=doctor_opinions)
		index_words = []
		for opinion in splitted_opinions:
			sub_opinions = opinion.split('\n\n\n')
			for sub_opinion in sub_opinions:
				if self.need_recheck(sub_opinion):
					keywords = self.dease_index_words_extract(sub_opinion.strip())
					index_words.append(self.extract_index_words(keywords=keywords))
		return index_words


if __name__=='__main__':
	doctoral_opinion='''"* 갑상선 초음파 검사결과 우측 갑상선 결절 소견입니다.



  양성 질환으로 판단되며, 갑상선결절에 대한 치료 및 추가검사는 필요하지 않습니다.

  결절의 크기나 모양에 변화가 생기는지 1년 후 추적검사를 권합니다.





 * 유방 초음파 검사결과 양측 유방의 석회화 및 우측 결절 소견입니다.



  석회화란 석회 등이 침착되어 조직이 딱딱해지는 경우를 말합니다.

  석회화에는 원인이 여러가지 있는데, 염증 후에 생기는 경우도 있지만,

  반면에 원인을 모르는 경우도 있습니다.



  치료는 불필요하며, 크기가 커지는지 주위에 다른 종괴는 없는지 등의 정기적인

  검진으로 추적관찰 하시기 바랍니다.



  유방 결절은 양성질환으로 생각되며, 이에 대한 현 치료는 필요하지 않습니다.

  크기나 모양의 변화를 관찰하기 위하여 6개월 후 추적검사를 권합니다."
'''
	detector = Detector()
	extracted_words = detector(doctoral_opinion)
	print(extracted_words)
# 	splitted_opinions=doctor_opinion_split(doctoral_opinion=doctoral_opinion)
# 	index_words = []
# 	for opinion in splitted_opinions:
# 		sub_opinions = opinion.split('\n\n\n')
# 		for sub_opinion in sub_opinions:
# 			if need_recheck(sub_opinion):
# 				keywords = dease_index_words_extract(sub_opinion.strip())
# 				index_words.append(extract_index_words(keywords=keywords))


#### extract time keywords from opinions text
# import pandas as pd
# opinions = pd.read_excel('dataset/종합검진소견.xlsx')
# dease_info = pd.read_excel('dataset/질환정보.xlsx')
#
#
# for row in dease_info.itertuples():
# 	if row[7] =='Y':
# 		try:
# 			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
# 		except IndexError as e:
# 			print(e)
# 			continue
# 		keywords = ['1일', '하루']
# 		for word in keywords:
# 			if word in opinion:
# 				print(row[1])
# 				file = open(f'{row[1]}_{word}.txt','w')
# 				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 1일\n\n\n소견에서 찾아되는 시간: {word}')
# 	if row[8] =='Y':
# 		try:
# 			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
# 		except IndexError as e:
# 			print(e)
# 			continue
# 		keywords = ['1개월', '1달', '일개월', '한달','1 개월', '1 개 월', '1개 월']
# 		for word in keywords:
# 			if word in opinion:
# 				print(row[1])
# 				file = open(f'{row[1]}_{word}.txt','w')
# 				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 1개월\n\n\n소견에서 찾아되는 시간: {word}')
# 	if row[9] =='Y':
# 		try:
# 			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
# 		except IndexError as e:
# 			print(e)
# 			continue
# 		keywords = ['3개월', '3달', '삼개월', '세달','3 개월', '3 개 월', '3개 월']
# 		for word in keywords:
# 			if word in opinion:
# 				print(row[1])
# 				file = open(f'{row[1]}_{word}.txt','w')
# 				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 3개월\n\n\n소견에서 찾아되는 시간: {word}')
# 	if row[10] =='Y':
# 		try:
# 			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
# 		except IndexError as e:
# 			print(e)
# 			continue
# 		keywords = ['6개월', '6달', '육개월', '여섯달','6 개월', '6 개 월', '6개 월']
# 		for word in keywords:
# 			if word in opinion:
# 				print(row[1])
# 				file = open(f'{row[1]}_{word}.txt','w')
# 				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 6개월\n\n\n소견에서 찾아되는 시간: {word}')
# 	if row[11] =='Y':
# 		try:
# 			opinion = str(opinions['종합검진소견'].loc[opinions['접수번호']==row[1]].values[0])
# 		except IndexError as e:
# 			print(e)
# 			continue
# 		keywords = ['1년']
# 		for word in keywords:
# 			if word in opinion:
# 				print(row[1])
# 				file = open(f'{row[1]}_{word}.txt','w')
# 				file.write(f'내원번호: {row[1]}\n\n\n검진소견:{opinion}\n\n\n질환명: {row[4]}\n\n\n실제 다시검진 기한: 1년\n\n\n소견에서 찾아되는 시간: {word}')
#
