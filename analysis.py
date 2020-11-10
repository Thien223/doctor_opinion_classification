import fasttext
model=fasttext.load_model('pretrained_words_vectors/fasttext/cc.ko.300.bin')
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
	cosine_similar_scores = np.zeros(shape=(len(index_words_1),len(keywords)))


	for j, keyword in enumerate(keywords):
		for i, index_word in enumerate(index_words_1):
			### calculate similarity using bleu score
			bleu_similar_scores_1[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)

	for j, keyword in enumerate(keywords):
		for i, index_word in enumerate(index_words_1):
			### calculate similarity using bleu score
			# bleu_similar_scores_2[i,j] = sentence_bleu([list(index_word.replace(' ',''))], list(keyword.replace(' ','')), auto_reweigh=True)
			## calculate similarity using cosine similarity
			cosine_similar_scores[i,j] = similarity_calculate(index_word.replace(' ',''), keyword.replace(' ',''), model=model) if str(similarity_calculate(index_word, keyword, model=model))!='nan' else 1.0
	# # similar_scores = np.sqrt(bleu_similar_scores**2 + cosine_similar_scores**2)
			# bleu_similar_scores_2[i,j] = sentence_bleu([list('밥을 먹었어요?')], list('밥을 먹었어?'), auto_reweigh=True)

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
	doctoral_opinion = '''"* 흉부 CT 검사결과 간유리양 음영 폐결절, 관상동맥에 석회화, 흉부 대동맥 죽상경화증 소견입니다.



 - 간유리양 음영 결절은 염증, 감염, 알레르기 병변(호산구성 폐렴 등), 

 출혈과 같은 일시적인 병변 가능성 높으나 드물게

 악성 결절의 가능성도 배제할 수 없습니다.

  

 임상적 증상 없으시면 1~3개월 뒤 추적 검사 권합니다.



 - 관상동맥은 심장에 혈액을 공급하는 혈관으로 관상동맥 석회화에 의해 

 관상동맥 질환이 있거나 향후 발생 가능성이 비교적 높아 보입니다.

 심혈관 질환 진행을 예방하기 위해 반드시 금연하시고, 싱거운 식습관, 

 적정 혈압 및 정상 체중 유지, 규칙적인 유산소 운동을 권합니다.



 흉통이나 숨이차는 등의 증상이 있거나 증상이 없더라도 

 향후 심혈관질환의 위험인자 관리 및 필요시 추가적인 정밀검사 평가를 위해 

 심장 내과 전문의 상담 권합니다.



 - 죽상경화증이란 동맥의 탄력이 떨어지고 동맥에 혈전등이 생기거나 하는등

 기타의 이유로 동맥이 좁아지는 질병입니다.

 진행을 예방하는 치료에는 건강한 혈관을 유지하기 위한 혈압관리, 당뇨병관리,

 금연, 규칙적인 유산소 운동과 체중관리 및 혈액의 콜레스테롤 함량을

 개선시키기위한 생활습관 관리와 약물치료(지질강하제) 등이 있습니다.



 추후 추적검사 및 전문의 상담 권합니다.



* 자궁경부암 세포진 검사상 반응성 세포변화 소견입니다.



 - 반응성 세포 변화란 여러 가지 자극에 의한 세포 모양의 변화가 생긴 것입니다. 

 질 분비물의 증가와 가려움 등의 증상이 동반되었을 경우는 산부인과 진료 권하며,

 특별한 증상이 동반되지 않은 경우 6개월 후 산부인과 진료 권합니다.



* 유방 촬영 검사결과 치밀 유방 및 비대칭 음영으로 인한 판정유보 소견이나 

  유방 초음파 검사상 특이소견 없습니다.



 - 유방 검진상 특이소견 없으며 정기적인 추적 관찰 권합니다.



* 복부 초음파 검사결과 담낭 결석 소견을 보입니다.



 - 담낭 결석에 대하여 우상복부 통증, 발열 등의 증상이 있는 경우

  담도계 외과 전문의 진료 권합니다.



* 경동맥 초음파 검사결과 우측 경동맥 비후 및 협착 소견입니다.



 - 경동맥 유소견에 대해 혈관 질환에 대해 생활습관 개선 및 약물치료 필요할 수 있으니 

 내과/신경외과 전문의 상담 권합니다.



 건강한 혈관을 유지하기 위해서는  혈압 관리, 당뇨병 관리, 

 금연, 규칙적인 유산소 운동과 체중 관리 및 혈액의 콜레스테롤 함량을 

 개선시키기 위한 생활습관 관리가 필요합니다.



* 갑상선 초음파 검사결과 좌측 갑상선 결절(Indeterminate nodule)및 낭종,

 우측 갑상선 결절 소견입니다.



 - 상기 소견에 대해 추가 검사 필요할 수 있어 갑상선 외과 전문의 진료 권합니다.



* 골밀도 검사상 골다공증 소견 입니다.



 - 골다공증에 대해 약물 치료가 필요하여 산부인과 / 내분비 내과 전문의 진료 권합니다.



* 동맥경화도 검사결과 혈관벽이 약간 단단합니다.



 - 동맥 경화도 검사는 동맥의 경직도를 재어(맥파속도로 확인) 동맥경화정도를 측정하고 

 발목 상완 지수(ABI)를 통해 말초 동맥의 협착이나 폐쇄 정도를 확인하는 검사 입니다. 

 환자분의 경우 맥파속도(PWV)가 증가되어 동일연령군에 비해 혈관 탄력정도가 떨어져 

 혈관질환의 위험성이 증가 되어있습니다.



 동맥경화증을 예방하기 위해 생활습관 교정(체중 감소, 운동, 금연), 및 

 혈관질환 위험인자(당뇨, 혈압, 이상지질혈증 동반된 경우)에 대한 

 철저한 관리가 필요하며 추적 검사 권합니다.



* 혈액검사상 혈당 수치 상승으로 공복혈당장애 소견 입니다.



 - 공복혈당장애는 공복혈당이 100 ~ 125 mg/dl의 범위에 있을 나타나는 소견 입니다.

 당뇨병 전단계로 고려되지만 당뇨로 진행될 수 있으므로 예방을 위해

 관리하여야 합니다. 



 비만에 유의하시고, 심한 폭음, 과식, 불규칙한 식사, 과로 등을 피하고 

 적당한 운동 하면서 공복시 반복적인 혈당 측정 권합니다.



* 혈액 검사결과 불현성 갑상선 기능 항진증 소견입니다.



 - 불현성 갑상선 기능항진증(갑상선 자극 호르몬 감소 및 갑상선 호르몬 정상)은 

 특이증상은 없으나 심장질환 및 골밀도 감소의 가능성이 있기 때문에 치료가 필요한 

 경우가 있습니다. 



 3개월 뒤 내분비 내과로 추적 검사 권합니다.



* 검진결과 고혈압 전단계 소견입니다.  

  

 - 고혈압 전단계란 수축기 혈압이 120~140mmHg  사이를 의미하며, 고혈압으로  

 진행할 가능성이 있습니다. 

 1-2주 후 흡연, 커피를 삼가하고, 충분한 휴식 후에 재검사 받으시길 바랍니다.



* 체성분 검사결과 비만입니다.



 - 규칙적인 운동, 식이조절을 통해 지방량을 8.8kg 감량하고, 근육량을 1.6kg 증가시켜

 적정체중을 유지하시기 바랍니다.



* B형간염 검사결과 항체가 형성되어 면역보유자로 예방접종이 필요하지 않습니다.



* 혈액 검사결과 NK 세포 활성도 저하(250-500 pg/mL 사이)-관심 구간 의심 소견입니다.



 - NK세포 활성이 약간 떨어지는 경계 상태로 NK 세포 활성을 저하시키는 질환이나 

 약물 복용 수면 부족 및 극심한 스트레스 등을 의심 할 수 있습니다. 

 NK 세포 활성도를 높여 주는 방법은 삼림욕, 많이 웃기, 명상, 숙면, 버섯이나 현미

 섭취 등이 있습니다. 

 면역력 증진을 위한 꾸준히 활동 후 추적 검사 권합니다.

"

'''
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
