# About
앙상블을 하거나 대회에 조금 더 편하게 프로그램을 제출하기 위해 만든 공간입니다.
본 소스 코드를 돌리기 위해서는 jupyter-notebook과 linux환경이 필요합니다.

현재 제공하는 모델은 아래와 같습니다.
1. Sklearn Tree(Desicion Tree Classifier)
2. Symetric Repairing
3. ice-cube(결과만)
4. arc-dsl
5. Different Solvers (diff arc solvers)
6. 

# 구조
전체 프로젝트의 폴더는 데이터, 데이터 로더, model, cooked moded, setting, visualization 으로 구성되어 있습니다.
데이터 폴더에는 데이터를 생성한 프로젝트 또는 기여자를 바탕으로 만들어주시면 됩니다. 자세한 사항은 아래를 참고 하실 수 있습니다.
데이터 로더 폴더에는 데이터를 쉽게 불러올 수 있도록 합니다. 자세한 사항은 아래를 참고 하실 수 있습니다.
model 폴더에는 가장 핵심이 되는 로직을 정리해서 넣으시면 됩니다. 모델의 원칙은 이미 앙상블이어여서는 안된다는 것 입니다. 자세한 사항은 아래를 참고 하실 수 있습니다.
cooked model은 메인 함수에서 측시 끌어다 쓸수 있어야 하며, 이는 데이터를 기록할 store, 그리고 가장 핵심적인 task 그리고 하나의 인풋을 넣어주어야 합니다. 자세한 사항은 아래를 참고 하실 수 있습니다.
setting은 경로의 자동화를 위해서 경로에 대한 명시 파일이며 데이터를 추가시 반드시 여기에 경로를 추가해 주어야 합니다.

# Data
arc에는 현재 여러명 배포자들이 배포를 진행하고 있습니다. 또한 각 배포의 라이센스는 다양하기 때문에 본 소스코드는 이들의 사용에 대해 신중을 기하려고 하고 있습니다. 만약 저작권적 문의가 있을시 issue란에 작성하면 수정하겠습니다.
arc의 배포중 현재 가장 핵심은 kaggle입니다. kaggle 이외에 추가로 지원을 만들 계획입니다. 
arc-prize의 문제는 다음과 같은 구조로 되어 있습니다. 추가하고 싶은 모든 데이터는 아래의 형식을 지켜야 합니다.

```json
{
  "arc_id": {
    train: [{
      input: []
      output: []
    }],
    test: [{
      input: []
    }]
  }
}
```

이외에 submission을 담는 곳도 존재합니다. 이는 앙상블을 위해 어떤 데이터셋을 풀었을때의 정보를 저장합니다.
```

def make_submission_file(data, solver_name, dataloader, data_mode):
    now = datetime.now()
 

    metadata = {
        "date": now.strftime("%Y-%m-%d"),  # YYYY-MM-DD 형식
        "time": now.strftime("%H:%M:%S"),   # HH:MM:SS 형식'
        "data_mode": data_mode,
        "solver_name": solver_name,
    } 
    json_file_data = {
        'metadata': metadata,
        'submission': data
    }

    # JSON 파일로 저장
    with open('./data/submissions/{}-{}-{}-{}.json'.format(metadata['date'], metadata['time'],metadata['data_mode'], metadata['solver_name'] ), 'w') as json_file:
        json.dump(json_file_data, json_file, indent=4)

make_submission_file(fdata1, "predict_repeating", data, data_mode)
```
여기에는 어떤 데이터 셋의 어떤 모드를 어떤 솔버로 풀었는지에 대한 메타데이터 정보를 포함해야 합니다.

# DataLoader
데이터 로더는 지속적으로 불러오는 데이터를 변경할수 있는 클래스로 싱클톤 구조로 짜여있지 않습니다. 따라서 잦은 재정의는 프로그램의 리소스를 낭비시킬 수 있습니다. 
데이터 로더를 통해 최초에 데이터를 불러올 때는 아래와 같이 넣어줄 수 있습니다.
arcprize 
```py

base_path = '/Users/lyuhyeonseog/dev/arc-enssamble'
local_path = base_path+"/data/kaggle/"
with open(base_path+"settings/kaggle_data_file_name.json",'r') as file:
    path_dict = json.load(file)
    data = MyDataLoader("arcprize", path_dict, local_path) 
    data.cur_data_mode("train")
```

데이터 로더는 1개의 method가 있습니다. 
+ cur_data_mode(dataname): 이는 이미 로드된 데이터로 부터 쉽게 train, test, evaㅣuation을 변경할 할 수 있도록 도와줍니다.
+ cur_problem[key]: 를 통해서 원하는 데이터를 위의 형식으로 불러올 수 있습니다.
+ 

```py
data_mode='evaluation'
data.cur_data_mode(data_mode)
data.cur_problem['bf699163']
```

# model
작성예정
# cooked model 
작성예정
# 철학

1. 앙상블을 위해 이미 돌아간 코드를 돌리고 또돌리고를 반복하지말자
2. 다양한 데이터를 추가할때는 어렵게 사용할때는 쉽게
3. 다양한 사람이 같이 사용할 수 있게
4. 시각적으로 풍부하게



