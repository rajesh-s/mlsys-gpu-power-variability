import pandas as pd

df = pd.read_csv("combined.csv")
for i in range(1,33):
    if i == 9 or i == 30:
        continue
    case_num = "case" + str(i)
    for j in range(0, 4):
        if len(df[(df.case_num == case_num) & (df.GPU_NO == j)]) > 0:
            #print("Average runtime for " + case_num + " on GPU " + str(j) + " is " + str(df[(df.case_num == case_num) & (df.GPU_NO == j)]['Runtime'].mean()))
            print(case_num + "," + str(j) + "," + str(df[(df.case_num == case_num) & (df.GPU_NO == j)]['Runtime'].mean()))
            #print(df[(df.case_num == case_num) & (df.GPU_NO == j)]['Runtime'].mean())
