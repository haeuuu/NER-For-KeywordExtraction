# NER-For-Keyword Extraction

### Named Entity Recognition for Keyword extraction from Online lecture

"온라인 학습도우미 PLUS" 의 두 번째 기능인 '키워드 추출'을 위한 repo입니다.
  
  
  
#### Keyword

* Named Entity Recognition
* BERT
* Conditional Random Field
* TF-IDF
  
  
  
## :peach: Train Data / Test data

##### NER 대회용 데이터와 한국민족문화대백과사전에서 스크래핑한 데이터를 섞어 train/test
  
  
  
### 1. Naver/창원대 NLP Challenge

> 네이버와 창원대에서 개최한 NLP challenge의 train set

##### 	아래와 같은 총 14개의 개체명 태그가 존재한다.

1. `PERSON`	PER	실존, 가상 등 인물명에 해당 하는 것
2. `FIELD`	FLD	학문 분야 및 이론, 법칙, 기술 등
3. `ARTIFACTS_WORKS`	AFW	인공물로 사람에 의해 창조된 대상물
4. `ORGANIZATION`	ORG	기관 및 단체와 회의/회담을 모두 포함
5. `LOCATION`	LOC	지역명칭과 행정구역 명칭 등
6. `CIVILIZATION`	CVL	문명 및 문화에 관련된 용어
7. `DATE`	DAT	날짜
8. `TIME`	TIM	시간
9. `NUMBER`	NUM	숫자
10. `EVENT`	EVT	특정 사건 및 사고 명칭과 행사 등
11. `ANIMAL`	ANM	동물
12. `PLANT`	PLT	식물
13. `MATERIAL`	MAT	금속, 암석, 화학물질 등
14. `TERM`	TRM	의학 용어, IT곤련 용어 등 일반 용어를 총칭
  
  
  
#### NOTE

* `index`는 새로운 문장이 시작될 때마다 1로 초기화된다.
* `tag`의 앞부분은 개체명의 의미를, 뒷부분은 BIO tagging을 뜻한다.
* B는 개체명의 시작 어절, I는 끝 어절, -는 개체명이 아닌 어절을 뜻한다.
* 두 개체명이 조합된 경우, 앞에 등장하는 개체명을 따라 태그를 부여한다. 
  ex ) 포항공과대학교(LOC_B) 컴퓨터공학과(ORG_B) => LOC로 부여
  
  
  
### 2. 한국민족문화대백과사전

* 인물, 지명, 문화재, 유물, 단체 등의 카테고리를 이용하여 true tag를 생성

  1. 각 카테고리에 접근한다.

  2. 단체 카테고리에 속하는 단어들은 모두 `ORG `를 true tag로 지정한다.

  3. NER 학습을 위해서는 문장이 필요하다. 해당 단어가 포함된 설명을 스크래핑한다.

     <img src="C:\Users\haeyu\AppData\Roaming\Typora\typora-user-images\image-20200803171703617.png" alt="image-20200803171703617" style="zoom:67%;" />

     <img src="C:\Users\haeyu\AppData\Roaming\Typora\typora-user-images\image-20200803171946450.png" alt="image-20200803171946450" style="zoom: 67%;" />

     > `교민` : `-` , `중국` : `-` , `관헌도` : `-` ,  `간민회` : `ORG_B`

  4. true tag가 달리지 않은 `교민`, `중국`, `관헌도 `등은 **기존의 model(acc 97%)를 이용하여 약한 정답**을 생성한다.

     > **기존 모델에 의한 정답**  `교민` : `PER` , `중국` : `LOC` , `관헌도` : `LOC`  ,`간민회` : `-`
     >
     > **스크래핑으로 생성한 정답**  `교민` : `-` , `중국` : `-` , `관헌도` : `-` ,  `간민회` : `ORG_B`
     >
     > **=> 최종 모델에 대한 정답 ** `교민` : `PER` , `중국` : `LOC` , `관헌도` : `LOC`  ,`간민회` : `ORG_B`
  
  
  
  
#### NOTE

* 전체 카테고리 중 "유물","유적","작품","제도","지명","문헌","단체","문화재" 를 이용
  
  
  
  

## :peach: DistilBert + CRF

#### Model

* Knowledge distillation을 이용하여 경량화된 KoBERT를 이용

  > [DistilKoBERT](https://github.com/monologg/DistilKoBERT)

  * 12 layer를 3개의 layer로
  * 한국어 위키, 나무위키, 뉴스 등의 10GB 데이터로 3 epoch학습
  
  
  
  
* Accuracy : 0.9069
* 태그별 precision, recall, f1-score 체크

<img src="C:\Users\haeyu\AppData\Roaming\Typora\typora-user-images\image-20200803191525711.png" alt="image-20200803191525711" style="zoom:80%;" />

* 한국사에서 중요한 태그라고 생각되는 `PER`, `ORG`, `CVL`, `EVT`에 대해 f1-score는 위와같다. 

* 주로 I tag를 잘 예측하지 못하는데, 이는 한국사 용어 특성상 여러가지 개체명이 결합되어있기 때문에 발생한 현상으로 보인다.

  ##### 특히 LOC_I의 score가 낮은 이유는 다음과 같이 추측해볼 수 있다.

  * 앞 글자가 LOC일 경우, 뒷 글자까지 LOC일 것으로 기대하지만 실제로는 아닌 경우가 많다. 이는 한국사 단어 특성상 여러 개체명이 뭉쳐있기 때문에 발생한다.
  * ex : `강릉농악`에서 `강릉`은 `LOC`이지만 `강릉농악`은 `AFW`
  * 위와 같은 이유로 LOC_I에 대한 규칙이 모호해지면서 예측력이 하락했을 것이다.
