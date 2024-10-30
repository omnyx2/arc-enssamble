# arc-enssamble
arc-prize를 많이 돌아다녀도 바로 즉시 쓸만한 모델이 얼마없다  
즉시 쓸만한 형태로 만들기 위해서 다음의 조건을 만족해야한다.

+ 셋업이 완료가 되었다고 한다면, 함수 하나 아크 문제 하나로 호출이 가능해야 한다.
+ 셋업이 완료가 되지 않았다고 하면 함수 호출에 있어서 셋업(학습, C++의 경우는 컴파일)을 자동으로 해줘야한다.

덧붙여 데이터 로더의 호환성이 전부 다들 서로 다르기 때문에 아래의 조건을 계속 잡아주어야한다.

1. 데이터 호출 형태가 ARC-PRIZE에 맞춰져 있는가, 

train, test | array
array | dict
dict | {input,output}

2. 위의 데이터 외에도 호출시 즉시 배열 형태로 나타나는가 ?

현재 위의 조건을 만족하는 형태로 소스코드를 MODELS에 작성하고 한번에 쓰기 쉽게끔 만들어 COOKED MODELS에 집어넣어서 사용하면 된다.


