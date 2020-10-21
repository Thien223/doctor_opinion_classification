# import fasttext
# model=fasttext.load_model('pretrained_words_vectors/fasttext/cc.ko.300.bin')
from nltk.translate.bleu_score import sentence_bleu


def get_index_words(file='index_words.txt'):
	'''
	get 색인어 from file
	:return: list of 색인어
	'''
	index_words = []
	with open(file, 'r') as f:
		for line in f.readlines():
			index_words.append(line.strip())
	return index_words


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
	dease_opinions = doctoral_opinion.split('*')
	return dease_opinions[1:]


#dease_opinion=dease_opinions[1]
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
	### filter out the nouns to keep only complex nouns

	# keywords = list(set(phrases) - set(nouns)) ### remove 명사, keep only 숙어
	return keywords



def extract_index_words(keywords):
	'''
	calculate similarity between filtered_phrase (keywords) and 저장된 색인어, return best matched 색인어
	:param filtered_phrases: keywords extracted from 소견
	:return: best matched 색인어
	'''
	import numpy as np
	matched_index_words=[]
	# matched_name_words=[]
	corresponding_keywords_index=[]
	# corresponding_keywords_name=[]
	### get 색인어
	index_words_1 = get_index_words('index_words.txt')
	index_words_2 = get_index_words('index_words_.txt')
	bleu_similar_scores_1 = np.zeros(shape=(len(index_words_1),len(keywords)))
	# bleu_similar_scores_2 = np.zeros(shape=(len(index_words_2),len(keywords)))
	# cosine_similar_scores = np.zeros(shape=(len(index_words),len(keywords)))


	for j, keyword in enumerate(keywords):
		for i, index_word in enumerate(index_words_1):
			### calculate similarity using bleu score
			bleu_similar_scores_1[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)

	# for j, keyword in enumerate(keywords):
	# 	for i, index_word in enumerate(index_words_2):
	# 		### calculate similarity using bleu score
	# 		bleu_similar_scores_2[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)
	# 		## calculate similarity using cosine similarity
	# 		# cosine_similar_scores[i,j] = similarity_calculate(index_word.replace(' ',''), keyword.replace(' ',''), model=model) if str(similarity_calculate(index_word, keyword, model=model))!='nan' else 1.0
	# # similar_scores = np.sqrt(bleu_similar_scores**2 + cosine_similar_scores**2)

	for idx in np.argwhere(bleu_similar_scores_1 == bleu_similar_scores_1.max()):
		matched_index_words.append(index_words_1[idx[0]])
		corresponding_keywords_index.append(keywords[idx[1]])
		# print(f'색인어 --- {index_words_1[idx[0]]}')
		# print(f'keyword --- {keywords[idx[1]]}\n')
	# for idx in np.argwhere(bleu_similar_scores_2 == bleu_similar_scores_2.max()):
	# 	matched_name_words.append(index_words_2[idx[0]])
	# 	corresponding_keywords_name.append(keywords[idx[1]])
	# 	print(f'질환명 --- {index_words_2[idx[0]]}')
	# 	print(f'keyword --- {keywords[idx[1]]}\n')
	return [matched_index_words, corresponding_keywords_index]#, [matched_name_words, corresponding_keywords_name],


def need_recheck(opinion):
	time_keywords = ['1일']+['1개월', '1달', '일개월', '한달','1 개월', '1 개 월', '1개 월']+['12개월', '12달', '십이개월', '12 개월', '12 개 월', '12개 월']+['3개월', '3달', '삼개월', '세달','3 개월', '3 개 월', '3개 월']+['6개월', '6달', '육개월', '여섯달','6 개월', '6 개 월', '6개 월']+['1년', '2년']
	needed=False
	time_keywords_idx=-1
	for time in time_keywords:
		if time in opinion:
			needed = True
			time_keywords_idx = opinion.index(time)
	return needed, time_keywords_idx

def detect(doctor_opinions):
	import pandas as pd
	index_name_code = pd.read_excel('index_name_code_mapping.xlsx')

	splitted_opinions = doctor_opinion_split(doctoral_opinion=doctor_opinions)
	index_words = []
	name_words=[]
	code_words=[]
	opinions_sub_opinions=[]
	sub_opinions_keywords=[]
	for idx, opinion in enumerate(splitted_opinions):
		# print(f'opinion: -------- {opinion}')
		sub_opinions = opinion.split('-')
		# sub_opinion=sub_opinions[0]
		for idx_,sub_opinion in enumerate(sub_opinions):
			# print(f'sub_opinion: -------- {sub_opinion}')
			# print(f'opinion --- {opinion}')
			needed, time_keyword_idx = need_recheck(sub_opinion)
			# print(f'time_keyword_idx --- {time_keyword_idx}')

			if needed:
				matched_idx = 0
				matched_keyword=''
				keywords = dease_index_words_extract(sub_opinion.strip())
				# print(f'keywords: -------- {keywords}')
				[matched_index_words, corresponding_keywords_index]= extract_index_words(keywords=keywords)
				# print(f'corresponding_keywords --- {corresponding_keywords}')

				for matched_word, keyword in zip(matched_index_words, corresponding_keywords_index):
					keyword_idx = sub_opinion.index(keyword)
					if keyword_idx < time_keyword_idx:
						# print(f'sub_opinion --- {sub_opinion}')

						# print(f'keyword_idx --- {keyword_idx}')
						# print(matched_keyword)
						if keyword_idx >= matched_idx:
							matched_idx = keyword_idx
							matched_keyword=keyword
				if matched_keyword != '':
					index_word = matched_index_words[corresponding_keywords_index.index(matched_keyword)]
					name_word = str(index_name_code.loc[index_name_code['색인어']==index_word]['질환명'].values[0])
					code_word = str(index_name_code.loc[index_name_code['색인어']==index_word]['질환코드'].values[0])
					index_words.append(index_word)
					name_words.append(name_word)
					code_words.append(code_word)
				#
				# matched_idx = 0
				# matched_keyword = ''
				# for matched_word, keyword in zip(matched_name_words, corresponding_keywords_name):
				# 	keyword_idx = sub_opinion.index(keyword)
				# 	if keyword_idx < time_keyword_idx:
				# 		# print(f'sub_opinion --- {sub_opinion}')
				#
				# 		# print(f'keyword_idx --- {keyword_idx}')
				# 		# print(matched_keyword)
				# 		if keyword_idx >= matched_idx:
				# 			matched_idx = keyword_idx
				# 			matched_keyword=keyword
				# name_words.append(matched_name_words[corresponding_keywords_name.index(matched_keyword)])
				opinions_sub_opinions.append(sub_opinion)
				sub_opinions_keywords.append(keywords)
	return index_words, name_words,code_words, opinions_sub_opinions, sub_opinions_keywords

if __name__=='__main__':
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
	index_words, name_words,code_words, opinions_sub_opinions, sub_opinions_keywords = detect(doctoral_opinions)
	print(index_words)

	#### detect 소션 from file
	#========================================
	# import pandas as pd
	# df = pd.read_excel('dataset/질환정보 -색인어추가.xlsx')
	# from konlpy.tag import Okt
	#
	# open_korean_text = Okt()
	# output=[]
	# i=0
	# for row in df.itertuples():
	# 	if str(row[13])!='nan':
	# 		opinions = row[13]
	# 		keywords = open_korean_text.phrases(opinions)
	# 		output.append([opinions, keywords])
	# 		# print(f'접수 번호: {row[1]} -- 질화 코드: {row[3]} -- 색인어: {row[5]} -- 추출된 keyword: {extracted_words}')
	# 		i+=1
	# 	if i>=30000:
	# 		break
	# output_df = pd.DataFrame(output, columns=['종합소견', '추출 키워드 (숙어)'])
	# output_df.to_excel("result.xlsx")
