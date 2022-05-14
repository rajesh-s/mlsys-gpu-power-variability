import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.4)

df1 = pd.read_csv("agg_temp_freq.csv")
df = df1[((df1.case_num == "case17") | (df1.case_num == "case14") | (df1.case_num == "case15") | (df1.case_num == "case16"))]# & (df1.exp == "sgemm")]
df_resnet = df[df.exp == "resnet"]
df_sgemm = df[df.exp == "sgemm"]
#groups = df.groupby('device')
#for name, group in groups:
    #plt.plot(group.pwr, group.case_num, marker='o', linestyle='', markersize=12, label=name)
    #plt.plot(group.kernel, group.pwr, marker='o', linestyle='', markersize=12, label=name)
    #plt.plot(group.case_num, group.runtime, marker='o', linestyle='', markersize=12, label=name)
#plt.scatter(df.runtime, df.case_num, c = df.GPU_NO)
#print(len(df.pwr))
#print(len(df.case_num))
#bplot = sns.stripplot(x = df.exp, y=df.pwr, hue = df.case_num)
#bplot1 = sns.catplot(x = "exp", col = "case_num", y = "pwr", palette="Set2", ax = axes[0], data = df)
bplot1 = sns.boxplot(x = df_resnet.exp, y = df_resnet.pwr, palette="Set2", ax = axes[0])
bplot1 = sns.swarmplot(x = df_resnet.exp, y = df_resnet.pwr, hue = df_resnet.device, ax = axes[0])
bplot1 = sns.boxplot(x = df_sgemm.exp, y = df_sgemm.pwr, palette="Set2", ax = axes[1])
bplot1 = sns.swarmplot(x = df_sgemm.exp, y = df_sgemm.pwr, hue = df_sgemm.device, ax = axes[1])
axes[0].get_legend().remove()
axes[1].legend(loc = 'right', bbox_to_anchor = (1.25, 0.5), ncol = 1, fontsize = "6")
#bplot2 = sns.catplot(x = df.exp, col = df.case_num, y = df.pwr, palette="Set2", ax = axes[1])
#bplot2 = sns.swarmplot(x = df.exp, y = df.pwr, hue = df.device, ax = axes[1])
fig = bplot1.get_figure()
#plt.ylabel("Case")
#plt.grid(axis="y")
#plt.yticks(df.case_num.unique())
#plt.legend(handles=scatter_plot.legend_elements()[0], labels=df.GPU_NO)
fig.savefig("del.png", dpi=300)
