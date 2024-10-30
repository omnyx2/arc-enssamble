import inspect
import numpy as np
# ..................................................................................... 3
def prn_plus(prn, answer):

    for j in range(len(answer)):
        prn = prn + [answer[j]]

        if (j == 0):
            prn = prn + [answer[j]]

    return prn

# ..................................................................................... 1
def ganswer_answer(ganswer):

    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j].tolist()

        if (ganswer_j not in answer):
            answer.append(ganswer_j)

    return answer

# only my data loader
def depency_inject_funcs(dataloader, mode, mainfunc, check_funcs_with_args, *args, **kwargs):

    mainfunc_params = inspect.signature(mainfunc).parameters
    mainfunc_arg_names = mainfunc_params.keys()

    dataloader.cur_data_mode(mode)
    result = {}
    
    for task_id in dataloader.cur_problem:
        task = dataloader.cur_problem[task_id]
        for i in range(len(task['test'])):
            test_input = np.array(task['test'][i]['input'])
            prn = []
          
            # Check if all conditions in check_funcs_with_args are met
            if all(check_func(task, *func_args, **func_kwargs) for check_func, func_args, func_kwargs in check_funcs_with_args) or (len(check_funcs_with_args) == 0):
                main_args = []
                if 'task' in mainfunc_arg_names:
                    main_args.append(task)
                if 'test_input' in mainfunc_arg_names:
                    main_args.append(test_input)
 
                # Call mainfunc with only the required arguments
                ganswer = mainfunc(*main_args)
                
                if ganswer:
                    answer = ganswer_answer(ganswer)
                    prn = prn_plus(prn, answer)
                    result[task_id] = prn

    return result

