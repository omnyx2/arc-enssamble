from visualization.my_vis import *
def ganswer_answer(ganswer):

    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j].tolist()

        if (ganswer_j not in answer):
            answer.append(ganswer_j)

    return answer

# ..................................................................................... 2
def ganswer_answer_1(ganswer):

    answer = []
    for j in range(len(ganswer)):
        ganswer_j = ganswer[j]

        if (ganswer_j not in answer):
            answer.append(ganswer_j)

    return answer

# ..................................................................................... 3
def prn_plus(prn, answer):

    for j in range(len(answer)):
        prn = prn + [answer[j]]

        if (j == 0):
            prn = prn + [answer[j]]

    return prn

# ..................................................................................... 4
def prn_select_2(prn):
    if (len(prn) > 2):

        value_list = []
        string_list = []
        for el in prn:
            value = 0
            for i in range(len(prn)):
                if el == prn[i]:
                    value +=1
            value_list.append(value)
            string_list.append(str(el))

        prn_df  = pd.DataFrame({'prn': prn , 'value': value_list, 'string': string_list})
        prn_df1 = prn_df.drop_duplicates(subset=['string'])
        prn_df2 = prn_df1.sort_values(by='value', ascending=False)

        prn = prn_df2['prn'].values.tolist()[:2]

    return prn

def pic_diff(task, prn, answer, task_id, solver_name=""):
    print(solver_name, prn)
    plotter = ARCPlottor()
    plotter.check_type(task, task_id)
    plotter.check_type(prn, task_id)
    plotter.check_type(answer, task_id)