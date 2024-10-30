import numpy as np
 
#::::::::::::::::::::::::::::::::::::::::::::::
from sklearn.tree import *

import os, gc
 
import json, random

import numpy as np
import pandas as pd
 
 
from matplotlib import colors
import matplotlib.pyplot as plt
 
################################################################################
# 9 Functions - Via Colors Counter
################################################################################ 1
def Defensive_Copy(A):
    n = len(A)
    k = len(A[0])
    L = np.zeros((n,k), dtype = int)
    for i in range(n):
        for j in range(k):
            L[i,j] = 0 + A[i][j]
    return L.tolist()

################################################################################ 2
def Create(task, task_id=0):
    n = len(task['train'])
    Input = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output

################################################################################ 3
def colors_counter(task):
    Input = task[0]
    Output = task[1]
    Test_Picture = Input[-1]
    Input = Input[:-1]
    N = len(Input)

    for x, y in zip(Input, Output):
        if len(x) != len(y) or len(x[0]) != len(y[0]):
            return -1

    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))

    for Q1, Q2 in Pairs:
        for v in range(4):


            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}

            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:

                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2

                            color1 = x[i][j]
                            rule = (p1,p2,color1)

                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v


    if Best_Dict == -1:
        return -1 #meaning that we didn't find a rule that works for the traning cases

    #Otherwise there is a rule: so let's use it:
    n = len(Test_Picture)
    k = len(Test_Picture[0])

    answer = np.zeros((n,k), dtype = int)

    for i in range(n):
        for j in range(k):
            if Best_v == 0 or Best_v ==2:
                p1 = i%Best_Q1
            else:
                p1 = (n-1-i)%Best_Q1
            if Best_v == 0 or Best_v ==3:
                p2 = j%Best_Q2
            else :
                p2 = (k-1-j)%Best_Q2

            color1 = Test_Picture[i][j]
            rule = (p1, p2, color1)
            if (p1, p2, color1) in Best_Dict:
                answer[i][j] = 0 + Best_Dict[rule]
            else:
                answer[i][j] = 0 + color1


    return answer.tolist()

################################################################################ 4
def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

################################################################################ 5
def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1

    plt.tight_layout()
    plt.show()

################################################################################ 6
def plot_task1(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        fig_num += 1

    plt.tight_layout()
    plt.show()

################################################################################ 7
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

# plt.figure(figsize=(5, 2), dpi=200)
# plt.imshow([list(range(10))], cmap=cmap, norm=norm)
# plt.xticks(list(range(10)))
# plt.yticks([])
# plt.show()

def plot_picture(x):
    plt.imshow(np.array(x), cmap=cmap, norm=norm)
    plt.show()

################################################################################ 8
def color_count(testing_path):
    testing_tasks = sorted(os.listdir(testing_path))
    void = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    count_correct = 0
    count_solved = 0

    solved = []
    output_id = []
    testing_files = []
    testing_solved = []
    testing_correct = []

    for t in testing_tasks:
        with open(str(testing_path + '/' + t), 'r') as read_file:
            task = json.load(read_file)
            testing_files.append(task)

        L = len(task['test'])
        for i in range(L):
            basic_task = Create(task, i)
            answer = Recolor(basic_task)
            task_id = t.replace('.json', '_' + str(i))
            output_id.append(task_id)

            if (answer == -1):
                solved.append(void)

            if (answer != -1):
                solved.append(answer)
                testing_solved.append(task_id)
                count_solved += 1

                # print(24*'=')
                # print('No.', count_solved, '- Solved Answer')
                # print('Task:', task_id)
                # print(24*'=')
                # print('\nTask:', answer)

                # plot_picture(answer)
                # plot_task(task)


            if (answer != -1) and (task['test'][i]['output'] == answer):
                testing_correct.append(task_id)
                count_correct += 1

                print(24*'=')
                print('No.', count_correct, '- Correct Answer')
                print('Task:', task_id)
                print(24*'=')
                print('\nTask:', answer)

                plot_picture(answer)
                plot_task(task)

    print('=' * 100)
    print('Solved List:  Len =', len(testing_solved))
    print(testing_solved)
    print('=' * 100)
    print('Correct List:  Len =', len(testing_correct))
    print(testing_correct)
    print('=' * 100)
    print('END', '='*96, '\n')

    sub = pd.DataFrame({'output_id': output_id , 'solved': solved})
    return sub

################################################################################ 9
def color_count_test(testing_path, sample_path):
    testing_tasks = sorted(os.listdir(test_path))
    void = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    count_solved = 0

    solved = []
    output = []
    output_id = []
    testing_files = []
    testing_solved = []

    for t in testing_tasks:
        with open(str(testing_path + '/' + t), 'r') as read_file:
            task = json.load(read_file)
            testing_files.append(task)

        L = len(task['test'])
        for i in range(L):
            basic_task = Create(task, i)
            answer = Recolor(basic_task)
            task_id = t.replace('.json', '_' + str(i))
            output_id.append(task_id)

            if (answer == -1):
                solved.append(void)
                flv = flattener(void)
                output.append(flv+' '+flv+' '+flv)

            if (answer != -1):
                solved.append(answer)
                fla = flattener(answer)
                output.append(fla+' '+fla+' '+fla)
                testing_solved.append(task_id)
                count_solved += 1

                print(24*'=')
                print('No.', count_solved, '- Solved Answer')
                print('Task:', task_id)
                print(24*'=')
                print('\nTask:', answer)

                plot_picture(answer)

    print('=' * 100)
    print('Solved List:  Len =', len(testing_solved))
    print(testing_solved)
    print('=' * 100)

    sub = pd.DataFrame({'output_id': output_id , 'solved': solved})

    submission = pd.read_csv(sample_path)
    submission['output'] = output

    if(list(submission['output_id']) == output_id):
        print('END', '='*96, '\n')

    return sub, submission

