from models.symmetry_repariring import *
from cookedModels.utils import *

def cooked_symmetry_repairing(store, task, task_id, i, ganswer, prn, pic_mode=False):
    basic_task = Create(task, i)
    ganswer = symmetry_repairing(basic_task)

    if (ganswer != -1):
        answer = ganswer_answer_1(ganswer)
        prn = prn_plus(prn, answer)
        store[task_id] = prn
        if pic_mode and len(prn) != 0 and len(answer) != 0:
            pic_diff(task, prn, answer, task_id, solver_name="symmetry_repairing")
    
    return store