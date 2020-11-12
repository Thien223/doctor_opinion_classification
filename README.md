# 의사소견 분류 모델

## 목표

- 검진한 후 의사의 소견을 분석하여 그 소견에대한 질환명 또는 처방 예측 하는 목표임
- 예측 모델 사용함으로서 처방 작동으로 나오기 때문에 사람 인력 절감 
### 데이터 sample:
```
    opinion='''좌측 신장에 2.1'cm 크기의 단순 낭종(simple cyst)이 1개 있으며
            임상적 의미가 없는 병변임.
            담도 확장 없으며 간, 담낭, 우측 신장, 췌장, 비장에 특이소견 안보임.
        
            결론 : 좌측 신장 낭종.'''
```
## 모델
- CNN
- Inception module
- Fully connected layer
![모델 구조](모델구조.png?raw=true "소견 분류 모델 구조")

## 성능
- 처방 3개 (위내시경','상복부초음파','흉부촬영(PA)') 모델 개발하여 테스트 해보니까 성능은 90% 정확도 나옴


## 사용 법:
#### 가상 환경 설치
`pip install -r requirements.txt`

#### 예측 1 소견:
- 모델 checkpoint 및 tokenizer 다운로드 받음
- 다음 명령 실행
`python inference.py --opinion="<소견>" --tokenizer="<tokenizer_path>" --checkpoint="<checkpoint_path>"`
    - <소션> : 분류할 소견
    - <tokenizer_path> : 다운로드 받은 tokenizer 파일 path
    - <checkpoint_path> : 다운로드 받은 checkpoint 파일 path
    
#### 예측 많은 소견 한꺼번:
- 모델 checkpoint 및 tokenizer 다운로드 받음
- 먼저 분류 할 소견을 excel 파일 "소견" 칼럼에 놓음
- 프로젝트 root folder에서 다음 명령 실행
`python inference.py --opinions_path="<소견_파일_경로>" --tokenizer="<tokenizer_path>" --checkpoint="<checkpoint_path>"`
    - <소견_파일_경로> : 분류할 소견 파일 path
    - <tokenizer_path> : 다운로드 받은 tokenizer 파일 path
    - <checkpoint_path> : 다운로드 받은 checkpoint 파일 path
- 실행 끝나면 `result.xlsx` 파일 나오고 양식은: [소견 - 예측된 label] 임
