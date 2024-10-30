from models.via_tree import *
from cookedModels.utils import *
def cooked_sklearn_tree(store, task, test_input, task_id, prn, pic_mode=False):
    if check_subitem(task):
        train_t = format_features(task)
        test_t = make_features(test_input)
        ganswer = tree1(train_t, test_t, test_input)

        if (ganswer!= []):
            answer = ganswer_answer(ganswer)
            prn = prn_plus(prn, answer)
            store[task_id] = prn
            if pic_mode:
                pic_diff(task,prn,answer, task_id, solver_name="sklearn tree")
    