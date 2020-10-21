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

	# dease_opinion=dease_opinions[2]
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
		matched_index_words = []
		corresponding_keywords = []
		### get 색인어
		similar_scores = np.zeros(shape=(len(self.index_words), len(keywords)))

		for j, keyword in enumerate(keywords):
			for i, index_word in enumerate(self.index_words):
				### calculate similarity using bleu score
				similar_scores[i, j] = sentence_bleu([list(index_word.replace(' ', ''))], list(keyword.replace(' ', '')), auto_reweigh=True)
		## calculate similarity using cosine similarity
		# similar_scores[i,j] = similarity_calculate(index_word.replace(' ',''), keyword.replace(' ',''), model=model) if str(similarity_calculate(index_word, keyword, model=model))!='nan' else 1.0

		for idx in np.argwhere(similar_scores == similar_scores.max()):
			matched_index_words.append(self.index_words[idx[0]])
			corresponding_keywords.append(keywords[idx[1]])

		# print(f'색인어 --- {self.index_words[idx[0]]}')
		# print(f'keyword --- {keywords[idx[1]]}\n')
		return matched_index_words, corresponding_keywords

	def need_recheck(self, opinion):
		time_keywords = ['1일'] + ['1개월', '1달', '일개월', '한달', '1 개월', '1 개 월', '1개 월'] + ['3개월', '3달', '삼개월', '세달', '3 개월', '3 개 월', '3개 월'] + ['6개월', '6달', '육개월', '여섯달', '6 개월', '6 개 월', '6개 월'] + ['1년', '2년']
		needed = False
		time_keywords_idx = -1
		for time in time_keywords:
			if time in opinion:
				needed = True
				time_keywords_idx = opinion.index(time)
		return needed, time_keywords_idx

	def __call__(self, doctor_opinions):
		splitted_opinions = self.doctor_opinion_split(doctoral_opinion=doctor_opinions)
		index_words = []
		for opinion in splitted_opinions:
			temp = []
			# print(f'opinion: -------- {opinion}')
			sub_opinions = opinion.split('-')
			for sub_opinion in sub_opinions:
				# print(f'sub_opinion: -------- {sub_opinion}')
				# print(f'opinion --- {opinion}')
				needed, time_keyword_idx = self.need_recheck(sub_opinion)
				# print(f'time_keyword_idx --- {time_keyword_idx}')

				if True:
					matched_idx = 0
					matched_keyword = ''
					keywords = self.dease_index_words_extract(sub_opinion.strip())
					# print(f'keywords: -------- {keywords}')
					matched_index_words, corresponding_keywords = self.extract_index_words(keywords=keywords)
					# print(f'corresponding_keywords --- {corresponding_keywords}')

					for matched_word, keyword in zip(matched_index_words, corresponding_keywords):
						keyword_idx = sub_opinion.index(keyword)
						if keyword_idx < time_keyword_idx:
							# print(f'sub_opinion --- {sub_opinion}')

							# print(f'keyword_idx --- {keyword_idx}')
							# print(matched_keyword)
							if keyword_idx > matched_idx:
								matched_idx = keyword_idx
								matched_keyword = keyword
					if matched_keyword!='':
						index_words.append(matched_index_words[corresponding_keywords.index(matched_keyword)])
				break
		return index_words


if __name__ == '__main__':
	doctoral_opinions = '''"* 부인과 초음파 검사결과 자궁근종 소견 입니다.



 - 자궁근종은 자궁의 대부분을 이루고 있는 평활근(smooth muscle)에 생기는 종양이며 

 증상이 있는 경우(생리통, 월경과다, 빈혈)나 크기가 커지는 경우 



 산부인과 진료가 필요합니다.



* 위내시경 검사결과 만성 위축성 위염 소견입니다.



 - 위축성위염은 만성적인 염증과 노화에 의하여 위점막이 얇아진 상태입니다. 

 추후 정기 검진 권하며,



 흡연하고 있으시다면 꼭 금연하시고, 과음을 삼가십시오.

 위에 자극이 되는 짠 음식과 탄 음식을 삼가시는 것이 위암 예방에 도움이 됩니다.



* 갑상선 초음파 검사결과 양측 갑상선 결절 소견입니다. 



 - 갑상선 결절에 대한 현치료는 필요하지 않으며, 결절의 변화가 있는지

 6개월 후 추적검사를 권합니다.



* 청력 검사 결과 좌측 질환의심 소견 입니다.



 - 이비인후과 진료 보시기 바랍니다.



* 혈액 검사결과 백혈구 수치 감소 소견입니다. 



 - 이른 시간 채혈로 일시적인 백혈구 감소 보일 수 있으나 지속적인 백혈구 감소는 

 혈액 질환 가능성 배제할 수 없어 추적 검사 및 내과 진료 권합니다.



* 혈액 검사결과 amylase(아밀라제) 상승 소견입니다.



 - 아밀라제 상승은 급성 췌장염등 췌장질환, 이하선염, 만성간염 등에서 상승할 수 있으며, 

 구강 통증 및 복부 통증 등의 증상이 있으시면 내과진료를 권합니다.



* 체성분 검사결과 표준입니다.



 - 규칙적인 운동, 식이조절을 통해 지방량을 4.5kg 감량하고, 근육량을 4.1kg 증가시켜

 적정체중을 유지하시기 바랍니다.



* B형간염 검사결과 항체가 형성되어 면역보유자로 예방접종이 필요하지 않습니다.

"'''
	detector = Detector()
# print(extracted_words)
#
	extracted_words = detector(doctoral_opinions.strip())
	import pandas as pd
	df = pd.read_excel('dataset/질환정보 -색인어추가.xlsx')

	output=[]
	i=0
	for row in df.itertuples():
		if str(row[13])!='nan':
			print(row)
			opinions = row[13]
			extracted_words = detector(opinions.strip())
			output.append([opinions, extracted_words])
			# print(f'접수 번호: {row[1]} -- 질화 코드: {row[3]} -- 색인어: {row[5]} -- 추출된 keyword: {extracted_words}')
			i+=1
		if i>=30:
			break
	output_df = pd.DataFrame(output, columns=['종합소견', '추출 키워드'])
	output_df.to_excel("result.xlsx")


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


#### 테스트했음:

# 1901030007 종합검진소견과 색인어 (질환정보에) 안 맞음
# 1901040001 ok
# 1901020018 소결은 종합검진소견과 색인어 (질환정보에) 안 맞음 (소결 부족)
# 1901040006 소결은 종합검진소견과 색인어 (질환정보에) 안 맞음
# 1901040024