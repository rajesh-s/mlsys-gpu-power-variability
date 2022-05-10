import matplotlib.pyplot as plt
import pandas as pd

# Change to name of CSV
df = pd.read_csv("average_runtime.csv")
groups = df.groupby('GPU_NO')
for name, group in groups:
    plt.plot(group.runtime, group.case_num, marker='o', linestyle='', markersize=12, label=name)
    #plt.plot(group.case_num, group.runtime, marker='o', linestyle='', markersize=12, label=name)
#plt.scatter(df.runtime, df.case_num, c = df.GPU_NO)
plt.xlabel("Average Kernel Runtime")
plt.ylabel("Case Number")
plt.legend()
plt.grid(axis="y")
plt.yticks(df.case_num.unique())
#plt.legend(handles=scatter_plot.legend_elements()[0], labels=df.GPU_NO)
plt.savefig("del.png")
