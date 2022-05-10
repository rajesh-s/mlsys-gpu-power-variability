import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("agg_temp_freq.csv")
groups = df.groupby('device')
for name, group in groups:
    plt.plot(group.pwr, group.case_num, marker='o', linestyle='', markersize=12, label=name)
    #plt.plot(group.kernel, group.pwr, marker='o', linestyle='', markersize=12, label=name)
    #plt.plot(group.case_num, group.runtime, marker='o', linestyle='', markersize=12, label=name)
#plt.scatter(df.runtime, df.case_num, c = df.GPU_NO)
plt.ylabel("Case")
plt.xlabel("Power (W)")
plt.legend()
#plt.grid(axis="y")
#plt.yticks(df.case_num.unique())
#plt.legend(handles=scatter_plot.legend_elements()[0], labels=df.GPU_NO)
plt.savefig("del.png")
