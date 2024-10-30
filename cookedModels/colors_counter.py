from cookedModels.utils import *
from models.color_counter import *

def cooked_color_counter(store, task, task_id, i, ganswer, prn, pic_mode=False):
    basic_task = Create(task, i)
    answer = colors_counter(basic_task)

    if (ganswer != -1):
        answer = [answer]
        prn = prn_plus(prn, answer)
        if prn[0] != -1:
            store[task_id] = prn
            if pic_mode :
                # print(ganswer)
                pic_diff(task, prn, answer, task_id, solver_name="colors_counter")
        
    return store