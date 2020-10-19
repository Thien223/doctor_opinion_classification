
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
			print(f'색인어 --- {self.index_words[idx[0]]}')
			print(f'keyword --- {keywords[idx[1]]}\n')
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
			print(f'opinion: -------- {opinion}')
			# sub_opinions = opinion.split('\n\n\n')
			# for sub_opinion in sub_opinions:
			# 	print(f'sub_opinion: -------- {sub_opinion}')
			sub_opinion=opinion
			if self.need_recheck(sub_opinion):
				keywords = self.dease_index_words_extract(sub_opinion.strip())
				print(f'keywords: -------- {keywords}')
				index_words.append(self.extract_index_words(keywords=keywords))
		return index_words


if __name__=='__main__':
	doctoral_opinion='''"* 위내시경 검사결과 역류성 식도염 및 만성 위축성 위염, 점막하 종양 의심 소견입니다.



 - 역류성 식도염은 흡연, 음주, 커피, 기름진 음식, 야식 등이 주된 원인입니다.

 치료여부는 식도염 정도와 증상에 따라 달라지므로 내과 전문의 상담 권합니다.



 - 위축성위염은 만성적인 염증과 노화에 의하여 위점막이 얇아진 상태입니다. 

 정기적인 추적 검사 및 내과 전문의 상담 권합니다.



 흡연하고 있으시다면 꼭 금연하시고, 과음을 삼가십시오.

 위에 자극이 되는 짠 음식과 탄 음식을 삼가시는 것이 위암 예방에 도움이 됩니다.

 

 - 위 점막하 종양에 대해서 크기 및 모양 변화를 살펴보기 위해 

 정기적인 추적 검사가 필요합니다.



 - 위 조직 검사결과 만성 위염으로 결과 나왔습니다.

  상기소견에 대해 내과 전문의 상담 권합니다.



* 대장내시경 검사상 대장 용종(3mm*2/조직 검사 및 제거 시행) 소견입니다.



 - 조직검사상 과형성 용종으로 결과 나왔습니다.



 과형성 용종이란 비종양성 용종으로는 가장 많은 발생빈도를 보입니다.

 대장내시경 검사 시행 환자의 11%에서 발견할 수 있습니다.

 과형성 용종은 암으로 발전하지 않기 때문에 문제가 되지 않습니다.



 내과 전문의 상담 및 추적 검사 권합니다.



* 복부 초음파 검사결과 간낭종 및 간석회화, 양측 신낭종 소견입니다. 



 - 간낭종에 대해 현재 특별한 치료는 필요하지 않으며 낭종의 변화가 있는지 

 정기검진을 통하여 추적관찰 하시기 바랍니다.



 - 간석회화가 단순 실질 간석회화 인지 다른 병변에 연관된 석회화 인지 감별을 위해

 크기가 커지는지 주위에 다른 종괴는 없는지 등의 정기적인 검진으로

 추적관찰 하시기 바랍니다.



 - 신낭종에 대해 현재 특별한 치료는 필요하지 않으며 낭종의 변화가 있는지 

 정기검진을 통하여 추적관찰 하시기 바랍니다.



* 골밀도 검사상 골감소증 소견 입니다.



 - 골감소증은 골다공증의 전단계 소견으로 골밀도 감소 보입니다.

 1주 3회, 1회 30분 이상 체중 부하운동(조깅, 걷기, 자전거 타기 등)이나 

 근력강화 운동을 통해 적절한 골량을 유지하시고 절주, 금연을 권합니다. 

 비타민 D 합성을 위해 하루 30분 정도 햇볕을 쬐어주는 것도 좋습니다. 



 1000mg이상의 칼슘 및 800IU이상 비타민 D3복용을 권하며		

 추후 추적 검사 권합니다.



* 혈액 검사결과 신기능 저하 의심(크레아티닌 수치 상승 및 추정 사구체 여과율 감소) 소견입니다.	



 - 크레아티닌 상승은 탈수되거나 신장질환으로 인해 수치가 높게 나올 수 있습니다.

 사구체 여과율은 신기능 저하 여부를 확인하는 검사이나 나이, 환자분의 체구,

 일부 약물 등에 의해 영향을 받아 결과가 부정확할 수 있습니다.



 크레아티닌 수치 상승 및 추정 사구체 여과율 60 mL/min/1.73m2 미만으로 

 정확한 신기능 평가 위해 신장 내과 전문의 진료가 필요합니다.



* 혈액 검사결과 불현성 갑상선 기능 항진증 소견입니다.



 - 불현성 갑상선 기능항진증(갑상선 자극 호르몬 감소 및 갑상선 호르몬 정상)은 

 특이증상은 없으나 심장질환 및 골밀도 감소의 가능성이 있기 때문에 치료가 필요한 

 경우가 있습니다. 



 특이적인 임상증상 없으면 3개월 후 추적검사 권합니다.



* 혈액 검사결과 NK 세포 활성도 정상(500 pg/mL 이상) 소견입니다.



 - NK세포 활성이 정상인 상태로 면역 기능이 정상이며 현재 암이 발병했을 가능성은 

 낮습니다.

 다만 지속적인 스트레스 노출, 이상지질혈증, 면역 억제제 같은 약물 장기간 복용, 

 노화, 환경오염에 노출되는 상태라면 

 주기적인 항암면역 기능 상태 확인 위해 정기적인 추적 검사 권합니다.



* 체성분 검사결과 표준입니다.

 

 - 규칙적인 운동, 식이조절을 통해 지방량을 3.3kg 감량하고, 근육량을 2.6kg 증가시켜

 적정체중을 유지하시기 바랍니다.



* B형간염 검사결과 항체가 형성되어 면역보유자로 예방접종이 필요하지 않습니다.







 

"
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


#### 테스트했음:

# 1901030007 종합검진소견과 색인어 (질환정보에) 안 맞음
# 1901040001 ok
# 1901020018 소결은 종합검진소견과 색인어 (질환정보에) 안 맞음 (소결 부족)
# 1901040006 소결은 종합검진소견과 색인어 (질환정보에) 안 맞음
