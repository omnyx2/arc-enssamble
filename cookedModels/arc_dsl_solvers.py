import multiprocessing
import time
from models.arcdsls import solvers
import numpy as np
from cookedModels.utils import *


def compare_2d_lists(list1, list2):
    return np.array_equal(np.array(list1), np.array(list2))

def check_sols(input_arc, output_arc, dsl_list):
    n_correct = 0
    n_try = 0
    corrections = []
    Q = tuple(map(tuple, input_arc)) # input 여러개 있음, 하나의 인풋자체에 풀어야하는개 여러개가 있는 문제가 있다. 
    A = tuple(map(tuple, output_arc))
    for dsl in dsl_list:
        func = getattr(solvers, dsl)
        try:
            if compare_2d_lists(func(Q), A): # result에 연산에 대한 적용이 성공한상태
                corrections.append(dsl)
        except Exception as error:
            continue # 언젠가 분석해보기
    return corrections
 
def process_task_kit(task_id, data_sets, dsl_list, result_dict):
    flag = False
    dsl_store = []
    
    # 각 task_kit의 데이터를 처리
    for idx, singleQA in enumerate(data_sets[task_id]["train"]):
        if idx == 0:
            dsl_store = check_sols(singleQA['input'], singleQA['output'], dsl_list)
        else:
            dsl_store = check_sols(singleQA['input'], singleQA['output'], dsl_store)
    if len(dsl_store) > 0:
        flag = True
    
    if flag:
        # 결과를 공유 딕셔너리에 저장
        result_dict[task_id] = { "dsl_sols_store": dsl_store }
    return result_dict

# 병렬 처리를 적용한 sub_arc_dsl 함수
def sub_arc_dsl(data_sets, dsl_list):
 
    manager = multiprocessing.Manager()
    result_dict = manager.dict()  # 공유 딕셔너리
    
    processes = []
    start_time = time.time()  # Start time

    # 각 task_kit을 병렬로 처리
    for task_id in data_sets:
        process = multiprocessing.Process(
            target=process_task_kit, 
            args=(task_id, data_sets, dsl_list, result_dict)
        )
        processes.append(process)
        process.start()

    # 모든 프로세스가 종료될 때까지 대기
    for process in processes:
        process.join()

    end_time = time.time()  # End time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"Execution Time: {execution_time} seconds")

    return dict(result_dict)  # 결과를 일반 딕셔너리로 변환하여 반환

def run_dsl_solvers(data, data_mode): 
    # ...............................................................................
    dsl_list = [item for item in dir(solvers) if item.startswith('solve')]
    result = sub_arc_dsl(data.cur_problem, dsl_list)
    
    return result
# arc_dsl_answer = run_dsl_solvers(data)

def cooked_arc_dsl_solvers(store, data, data_mode, pic_mode=False):
    data.cur_data_mode(data_mode)
    arc_dsl_answer = run_dsl_solvers(data, data_mode)
    for keys in arc_dsl_answer:
            prn = []
            print(keys, arc_dsl_answer[keys]['dsl_sols_store'])
            for i in range(len(arc_dsl_answer[keys]['dsl_sols_store'])):
                try:
                    func = getattr(solvers, arc_dsl_answer[keys]['dsl_sols_store'][i])
                    Q = tuple(map(tuple, data.cur_problem[keys]['test'][0]['input']))
                    A = func(Q)
                    prn.append([list(row) for row in A ])
                except:
                    pass
            if(len(prn) != 0 ) and pic_mode:
                pic_diff(data.cur_problem[keys], prn, prn, keys, solver_name="arc_dsls")

            store[keys] = prn
    return store