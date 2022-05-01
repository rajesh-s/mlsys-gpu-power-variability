'''
    File name: sgemm-tacc-plotter.py
    Author: Prasoon Sinha
    Python Version: 3.7
    Created: 12/7/21
    Updated: 12/7/21
'''
import argparse
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cbook import boxplot_stats
import matplotlib as mpl
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter
from explorer import *
from matplotlib.lines import Line2D


# List of metrics
metrics = ['freq', 'temp', 'perf', 'pwr']

# Map metric to axis title
axis_titles = {
    'freq': ' Frequency (MHz)', 'temp': 'Temperature (C)', 'pwr': 'Power (W)', 'perf': 'Kernel Duration (ms)'}

# Range of y-values to show for each plot
lim_dir = {}
lim_dir['freq'] = (1224, 1230)
lim_dir['median_iter_dur'] = (2200, 2550)
lim_dir['temp'] = (40, 90)
lim_dir['pwr'] = (255, 300)

# Global variables
path = ''
cluster = ''
boxplot_no_breakdown = False
boxplot_node_breakdown = False
scatterplots = False
full_timeline = False
zoomed_timeline = False
include_points = False
get_raw_data_kernels = False

plt.style.use('seaborn-whitegrid')

# TACC Cabinets Order
tacc_cab = ['c002', 'c003', 'c004', 'c005', 'c006', 'c007', 'c008', 'c009']


def raw_data_kernel_timestamps(resnet_raw_path):
    file_dict = {f.split(".")[0]: os.path.join(resnet_raw_path, f)
                 for f in os.listdir(resnet_raw_path) if "csv" in f}
    df_dict = {k: read_nvprof_gpu_trace(v) for k, v in file_dict.items()}

    for k, v in df_dict.items():
        GPU = v.data.Device.min()
        device_id = v.device
        GPU = GPU[:-2] + device_id + ')'
        df = v.data

        kernel_start_times = df[(df.Device == GPU) & (
            df.GridX.isnull() == False)]
        kernel_start_times = kernel_start_times[kernel_start_times.Name !=
                                                '[CUDA memcpy HtoD]']
        kernels = kernel_start_times[['Start', 'Name']]
        kernels = kernels[(kernels.Start <= 400)]
        print(kernels)
        second_kernel_start_time = kernel_start_times.iloc[1]
        # print(second_kernel_start_time)


# Plot zoomed timeline of the raw csv data
def plot_zoomed_timeline(resnet_raw_path):
    # Create dictionary of dataframes, one per raw csv file in base_dir
    file_dict = {f.split(".")[0]: os.path.join(resnet_raw_path, f)
                 for f in os.listdir(resnet_raw_path) if "csv" in f}
    df_dict = {k: read_nvprof_gpu_trace(v) for k, v in file_dict.items()}

    # Array conttaining the data from two runs on this GPU
    data = []

    node = ''
    gpu = ''
    counter = 1

    # Make timeline plots one per metric, for each raw csv file in base_dir
    for k, v in df_dict.items():
        li = k.split("_")

        # Get device that this raw csv file corresponds to
        GPU = v.data.Device.min()
        device_id = v.device
        GPU = GPU[:-2] + device_id + ')'
        df = v.data

        a = df[(df.Device == GPU) & (df['Start'] >= 20)
               & (df['Start'] < 175)]
        a = a.drop(columns=['Duration', 'Device', 'System',
                            'Name', 'ones', 'memfreq', 'temp'])

        # Plot median frequency data
        # fig, ax = plt.subplots(figsize=(40, 20))
        # charts = sns.lineplot(
        #     x='Start', y='freq', data=a, lw=5)
        # ax.grid(False)
        # charts.set(xscale='linear')
        # # Specify size of axis labels
        # charts.set_yticklabels(charts.get_yticks(), size=40)
        # charts.set_xticklabels(charts.get_xticks(), size=40)
        # plt.xlabel('Time (s)', fontsize=65, fontweight=600, labelpad=30)
        # plt.ylabel(axis_titles['freq'], fontsize=65,
        #            fontweight=600, labelpad=30)
        # plt.savefig('../charts/timeline/resnet-on-tacc-freq-zoomed-timeline-' +
        #             li[3] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        # plt.close()

        # # Plot median pwr data
        # fig, ax = plt.subplots(figsize=(40, 20))
        # charts = sns.lineplot(
        #     x='Start', y='pwr', data=a, lw=5)
        # ax.grid(False)
        # charts.set(xscale='linear')
        # # Specify size of axis labels
        # charts.set_yticklabels(charts.get_yticks(), size=40)
        # charts.set_xticklabels(charts.get_xticks(), size=40)
        # plt.xlabel('Time (s)', fontsize=65, fontweight=600, labelpad=30)
        # plt.ylabel(axis_titles['pwr'], fontsize=65,
        #            fontweight=600, labelpad=30)
        # plt.savefig('../charts/timeline/resnet-on-tacc-pwr-zoomed-timeline-' +
        #             li[3] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        # plt.close()

        # Median frequency and power overlay
        fig, ax1 = plt.subplots(figsize=(40, 20))
        ax2 = ax1.twinx()
        charts = sns.lineplot(
            x='Start', y='pwr', data=a, lw=5, ax=ax1)
        charts = sns.lineplot(
            x='Start', y='freq', data=a, lw=5, color='orange', ax=ax2)
        charts.set(xscale='linear')
        ax1.grid(False)
        ax1.set_xticklabels(ax1.get_xticks(), size=40)
        ax1.set_yticklabels(ax1.get_yticks(), size=40)
        ax2.set_yticklabels(ax2.get_yticks(), size=40)
        ax1.set_xlabel('Time (s)', fontsize=65, fontweight=600, labelpad=30)
        ax1.set_ylabel('Power (W)', fontsize=65, fontweight=600, labelpad=30)
        ax2.set_ylabel('Frequency (MHz)', fontsize=65,
                       fontweight=600, labelpad=30)
        charts.legend(handles=[Rectangle((0, 0), 0, 0, color='blue', label='Nontouch device counts'), Line2D(
            [], [], marker='o', color='orange', label='Detections rate for nontouch devices')], loc=(1.1, 0.8))

        # plt.xlabel('Time (s)', fontsize=65, fontweight=600, labelpad=30)
        plt.savefig('../charts/timeline/resnet-on-tacc-freq-pwr-overlay-zoomed-timeline-' +
                    li[3] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        plt.close()


# Plot entire timeline of the raw csv data
def plot_full_timeline(resnet_raw_path):
    # Create dictionary of dataframes, one per raw csv file in base_dir
    file_dict = {f.split(".")[0]: os.path.join(resnet_raw_path, f)
                 for f in os.listdir(resnet_raw_path) if "csv" in f}
    df_dict = {k: read_nvprof_gpu_trace(v) for k, v in file_dict.items()}
    case_dir = os.path.basename(os.path.normpath(resnet_raw_path))
    os.mkdir('../charts/timeline/' + case_dir)

    # Array conttaining the data from two runs on this GPU
    data = []

    node = ''
    gpu = ''
    counter = 1

    # Make timeline plots one per metric, for each raw csv file in base_dir
    for k, v in df_dict.items():
        li = k.split("_")

        # Get device that this raw csv file corresponds to
        GPU = v.data.Device.min()
        #device_id = v.device
        device_id = li[4]
        GPU = GPU[:-2] + device_id + ')'
        df = v.data

        a = df[(df.Device == GPU)]
        a = a.drop(columns=['Duration', 'Device', 'System',
                            'Name', 'ones', 'memfreq', 'temp'])
        # if cluster == 'tacc':
        #     # a['GPU'] = c009-003-1 (cabinet 9, node 3, GPU 1)
        #     a['GPU'] = li[3] + '-' + device_id
        # data.append(a)
        # counter += 1

        # Create one large data frame consisting of different raw csv file data
        # results = pd.concat(data)

        # Plot median frequency data
        # fig, ax = plt.subplots(figsize=(40, 20))
        # charts = sns.lineplot(
        #     x='Start', y='freq', data=a, lw=5)
        # ax.grid(False)
        # charts.set(xscale='linear')
        # # Specify size of axis labels
        # charts.set_yticklabels(charts.get_yticks(), size=40)
        # charts.set_xticklabels(charts.get_xticks(), size=40)
        # plt.xlabel('Time (s)', fontsize=65, fontweight=600, labelpad=30)
        # plt.ylabel(axis_titles['freq'], fontsize=65,
        #            fontweight=600, labelpad=30)
        # plt.savefig('../charts/timeline/resnet-on-tacc-freq-timeline-' +
        #             li[3] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        # plt.close()

        # # Plot median pwr data
        # fig, ax = plt.subplots(figsize=(40, 20))
        # charts = sns.lineplot(
        #     x='Start', y='pwr', data=a, lw=5)
        # ax.grid(False)
        # charts.set(xscale='linear')
        # # Specify size of axis labels
        # charts.set_yticklabels(charts.get_yticks(), size=40)
        # charts.set_xticklabels(charts.get_xticks(), size=40)
        # plt.xlabel('Time (s)', fontsize=65, fontweight=600, labelpad=30)
        # plt.ylabel(axis_titles['pwr'], fontsize=65,
        #            fontweight=600, labelpad=30)
        # plt.savefig('../charts/timeline/resnet-on-tacc-pwr-timeline-' +
        #             li[3] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        # plt.close()

        fig, ax1 = plt.subplots(figsize=(40, 10))
        ax2 = ax1.twinx()
        charts = sns.lineplot(
            x='Start', y='pwr', data=a, lw=5, ax=ax1)
        charts = sns.lineplot(
            x='Start', y='freq', data=a, lw=5, color='orange', ax=ax2)
        charts.set(xscale='linear')
        ax1.grid(False)
        ax1.set_xticklabels(ax1.get_xticks(), size=40)
        ax1.set_yticklabels(ax1.get_yticks(), size=40)
        ax2.set_yticklabels(ax2.get_yticks(), size=40)
        ax1.set_xlabel('Time (s)', fontsize=35, fontweight=600, labelpad=20)
        ax1.set_ylabel('Power (W)', fontsize=35, fontweight=600,
                       labelpad=20, color='blue')
        ax2.set_ylabel('Frequency (MHz)', fontsize=35,
                       fontweight=600, labelpad=20, color='orange')
        # ax1.legend(handles=[Line2D([], [], color='blue', label='Power (W)'), Line2D(
        #     [], [], color='orange', label='Frequency (MHz)')], loc=(0, 0.85), prop={"size": 24})
        # ax1.legend(frameon=False)
        plt.savefig('../charts/timeline/'+ case_dir +'/resnet-on-tacc-freq-pwr-overlay-timeline-' +
                    li[5] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        #            li[3] + '-gpu' + device_id, bbox_inches='tight', dpi=300)
        plt.close()


# Plot scatterplots for aggregated data
def plot_scatterplots(df, hue, hue_order):

    filename = '../charts/sgemm-tacc'
    breakdown = '-cabinet-breakdown'

    df = df[df.cabinet != '-1']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2.2))

    # Perf vs. Temp
    plot1 = sns.scatterplot(
        data=df, x='perf', y='temp', hue=hue, hue_order=hue_order, s=40, ax=ax1)
    # plot1.yaxis.set_major_locator(ticker.MultipleLocator(50))
    # plot1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plot1.set_yticklabels(plot1.get_yticks(), size=10)
    plot1.set_xticklabels(plot1.get_xticks(), size=10)
    plot1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plot1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.set_xlabel(axis_titles['perf'], fontsize=10,
                   fontweight=600, labelpad=8)
    ax1.set_ylabel(axis_titles['temp'], fontsize=10,
                   fontweight=600, labelpad=8)
    ax1.text(0.5, -0.32, "(a)", size=9, ha="center",
             transform=ax1.transAxes)

    # Make legend: https://stackoverflow.com/questions/44071525/matplotlib-add-titles-to-the-legend-rows/44072076#44072076
    ph = [plt.plot([], marker="", ls="")[0]]
    h, l = ax1.get_legend_handles_labels()
    handles = ph + h
    labels = ['Cabinet:'] + l
    leg = ax1.legend(handles, labels, fontsize=10, handletextpad=0.1,
                     loc="upper center", bbox_to_anchor=[1.1, 1.2], columnspacing=1.2, ncol=9)

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    ax1.plot(rasterized=True)

    # Perf vs. Pwr
    plot2 = sns.scatterplot(
        data=df, x='perf', y='pwr', hue=hue, hue_order=hue_order, s=40, ax=ax2)
    plot2.set_yticklabels(plot2.get_yticks(), size=10)
    plot2.set_xticklabels(plot2.get_xticks(), size=10)
    plot2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plot2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # plot.yaxis.set_major_locator(ticker.MultipleLocator(50))
    # plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # lgnd = plt.legend(
    #     title="Row", title_fontsize=14, fontsize=12, handletextpad=0.1)
    ax2.set_xlabel(axis_titles['perf'], fontsize=10,
                   fontweight=600, labelpad=8)
    ax2.set_ylabel(axis_titles['pwr'], fontsize=10,
                   fontweight=600, labelpad=8)
    ax2.plot(rasterized=True)
    ax2.get_legend().remove()
    ax2.text(0.5, -0.32, "(b)", size=9, ha="center",
             transform=ax2.transAxes)

    plt.savefig(filename + breakdown + '-scatterplot.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    print('Plotted perf v. pwr')


# Plot boxplots for aggregated data (specify breakdown or no breakdown)
def plot_boxplots_old(df, include_points, hue, hue_order, breakdown, png_name):

    # One plot per metric, save as png
    for metric in metrics:

        # Each png file has multiple box plots, one per breakdown which passed as a parameter
        if breakdown != None:
            fig, ax = plt.subplots(figsize=(35, 8))
            plot = sns.boxplot(x=breakdown, y=metric, data=df,
                               showfliers=(not include_points), notch=False, boxprops={"zorder": 10}, whiskerprops={'linewidth': 3, "zorder": 10}, zorder=10)
            plt.xticks(rotation=90)
        # One boxplot per png file
        elif breakdown == None:
            fig, ax = plt.subplots(figsize=(11, 9))
            # Solution for ordering boxplot in front of stripplot is in this link https://stackoverflow.com/questions/44615759/how-can-box-plot-be-overlaid-on-top-of-swarm-plot-in-seaborn
            plot = sns.boxplot(y=metric, data=df, width=0.4,
                               showfliers=(not include_points), linewidth=5, notch=False, boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={'linewidth': 5, "zorder": 10}, zorder=10)

            # Boxplots should have black outline, no color on inside
            # for i, box in enumerate(plot.artists):
            #     box.set_edgecolor('black')
            #     box.set_facecolor('white')

        # Specify size of axis labels
        plot.set_yticklabels(plot.get_yticks(), size=40)
        plot.set_xticklabels(plot.get_xticks(), size=40)

        # Include indivudal data points, one point per row in dataframe
        if include_points:
            if breakdown != None:
                plot = sns.stripplot(x=breakdown, y=metric,
                                     data=df, s=12)
            elif breakdown == None:
                plot = sns.stripplot(
                    x='exp', y=metric, hue=hue, hue_order=hue_order, data=df, s=12, zorder=1)
                # Adjust legend title
                lgnd = plt.legend(
                    title="Cabinet", title_fontsize=38, fontsize=36, handletextpad=0.1)
                for handle in lgnd.legendHandles:
                    handle.set_sizes([140])
                # Remove border around legend
                # leg = plt.legend()
                # leg.get_frame().set_linewidth(0.0)
                plt.style.use('seaborn-whitegrid')
                # Hide axis ticks
                plt.gca().axes.xaxis.set_ticks([])

        # Set x and y axis labels
        xlabel = ''
        if breakdown == 'node':
            xlabel = 'Node'
        plt.xlabel(xlabel, fontsize=36, fontweight=600, labelpad=15)
        plt.ylabel(axis_titles[metric], fontsize=36,
                   fontweight=600, labelpad=15)

        # Increase granularity of y-axis ticks
        if metric == 'iter_dur':
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            plot.yaxis.set_major_locator(ticker.MultipleLocator(200))
            plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
        if metric == 'freq' or 'temp' or 'pwr':
            # plt.ylim(lim_dir['sclk_mhz'])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        # Save plot as png
        plt.plot(rasterized=True)
        plt.savefig('../charts/sgemm-' + cluster + '-' + metric + '-' +
                    png_name + '.png', bbox_inches='tight', dpi=300)
        plt.close()
        print('Plotted Boxplot: ' + metric)

        q1 = df[metric].quantile(0.25)
        q2 = df[metric].quantile(0.50)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        range = q3 + 1.5 * iqr - (q1 - 1.5 * iqr)
        percent_variability = range/q2 * 100

        print('Q1: ' + str(q1) + ' Q2: ' + str(q2) + ' Q3: ' + str(q3) +
              ' Percent Variability: ' + str(percent_variability) + '%')
        print()


def plot_boxplots(df, include_points, hue, hue_order, breakdown, png_name):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        1, 4, sharex=True, figsize=(25, 4.75))

    # Freq
    # Solution for ordering boxplot in front of stripplot is in this link https://stackoverflow.com/questions/44615759/how-can-box-plot-be-overlaid-on-top-of-swarm-plot-in-seaborn
    plot1 = sns.boxplot(y='freq', data=df, width=0.7,
                        showfliers=(not include_points), linewidth=5, notch=False, boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={'linewidth': 5, "zorder": 10}, zorder=10, ax=ax1)
    plot1.set_yticklabels(plot1.get_yticks(), size=25)
    plot1.set_xticklabels(plot1.get_xticks(), size=40)
    plot1 = sns.stripplot(
        x='exp', y='freq', hue=hue, hue_order=hue_order, data=df, s=10, zorder=1, ax=ax1)
    ax1.set_ylabel(axis_titles['freq'], fontsize=30,
                   fontweight=600, labelpad=10)
    ax1.set_xlabel('', fontsize=16,
                   fontweight=600, labelpad=10)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.gca().axes.xaxis.set_ticks([])
    ax1.text(0.5, -0.1, "(a)", size=20, ha="center",
             transform=ax1.transAxes)

    # Make legend: https://stackoverflow.com/questions/44071525/matplotlib-add-titles-to-the-legend-rows/44072076#44072076
    ph = [plt.plot([], marker="", ls="")[0]]
    h, l = ax1.get_legend_handles_labels()
    handles = ph + h
    labels = ['Cabinet:'] + l
    leg = ax1.legend(handles, labels, title_fontsize=26, fontsize=26, handletextpad=0.1,
                     loc="upper center", bbox_to_anchor=[3.0, 1.22], columnspacing=1.2, ncol=9)

    count = 0
    for handle in leg.legendHandles:
        if count != 0:
            handle.set_sizes([180])
        count += 1

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    ax1.plot(rasterized=True)

    # Perf
    plot2 = sns.boxplot(y='perf', data=df, width=0.7,
                        showfliers=(not include_points), linewidth=5, notch=False, boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={'linewidth': 5, "zorder": 10}, zorder=10, ax=ax2)
    plot2.set_yticklabels(plot2.get_yticks(), size=25)
    plot2.set_xticklabels(plot2.get_xticks(), size=40)
    plot2 = sns.stripplot(
        x='exp', y='perf', hue=hue, hue_order=hue_order, data=df, s=10, zorder=1, ax=ax2)
    ax2.set_ylabel(axis_titles['perf'], fontsize=25,
                   fontweight=600, labelpad=10)
    ax2.set_xlabel('', fontsize=16,
                   fontweight=600, labelpad=10)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.gca().axes.xaxis.set_ticks([])
    ax2.plot(rasterized=True)
    ax2.get_legend().remove()
    ax2.text(0.5, -0.1, "(b)", size=20, ha="center",
             transform=ax2.transAxes)

    # Power
    plot3 = sns.boxplot(y='pwr', data=df, width=0.7,
                        showfliers=(not include_points), linewidth=5, notch=False, boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={'linewidth': 5, "zorder": 10}, zorder=10, ax=ax3)
    plot3.set_yticklabels(plot3.get_yticks(), size=25)
    plot3.set_xticklabels(plot3.get_xticks(), size=40)
    plot3 = sns.stripplot(
        x='exp', y='pwr', hue=hue, hue_order=hue_order, data=df, s=10, zorder=1, ax=ax3)
    ax3.set_ylabel(axis_titles['pwr'], fontsize=30,
                   fontweight=600, labelpad=10)
    ax3.set_xlabel('', fontsize=16,
                   fontweight=600, labelpad=10)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().axes.xaxis.set_ticks([])
    ax3.plot(rasterized=True)
    ax3.get_legend().remove()
    ax1.text(0.5, -0.1, "(c)", size=20, ha="center",
             transform=ax3.transAxes)

    # Temp
    plot4 = sns.boxplot(y='temp', data=df, width=0.7,
                        showfliers=(not include_points), linewidth=5, notch=False, boxprops={'facecolor': 'None', "zorder": 10}, whiskerprops={'linewidth': 5, "zorder": 10}, zorder=10, ax=ax4)
    plot4.set_yticklabels(plot4.get_yticks(), size=25)
    plot4.set_xticklabels(plot4.get_xticks(), size=40)
    plot4 = sns.stripplot(
        x='exp', y='temp', hue=hue, hue_order=hue_order, data=df, s=10, zorder=1, ax=ax4)
    ax4.set_ylabel(axis_titles['temp'], fontsize=30,
                   fontweight=600, labelpad=10)
    ax4.set_xlabel('', fontsize=16,
                   fontweight=600, labelpad=10)
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().axes.xaxis.set_ticks([])
    ax4.plot(rasterized=True)
    ax4.get_legend().remove()
    ax1.text(0.5, -0.1, "(d)", size=20, ha="center",
             transform=ax4.transAxes)

    plt.subplots_adjust(wspace=0.7)
    plt.savefig('../charts/sgemm-' + cluster + '-' +
                png_name + '.png', bbox_inches='tight', dpi=300)
    plt.close()


def create_command_line_args():
    global path, cluster, boxplot_no_breakdown, include_points, boxplot_node_breakdown, scatterplots, full_timeline, zoomed_timeline, get_raw_data_kernels

    # Create actual parser
    parser = argparse.ArgumentParser(
        description='Application to create custom plots for AMD GPU data analysis')

    # Two metadata variables - path to aggregated data and cluster
    parser.add_argument('path', metavar='P', type=str,
                        nargs='?', help='path to aggregated csv file', default='./aggregate.csv')
                        #nargs='?', help='path to aggregated csv file', default='../aggregated-data/sgemm-on-tacc-aggregated-data.csv')

    parser.add_argument('cluster', metavar='C', type=str, default='tacc',
                        nargs='?', help='gpu cluster (i.e. tacc, cloudlab, summit, vortex, corona)')

    parser.add_argument('--boxplot-no-breakdown', dest='boxplot_no_breakdown',
                        help='Plot box plot, one per metric', action='store_const', const=True, default=False)

    parser.add_argument('--boxplot-node-breakdown', dest='boxplot_node_breakdown',
                        help='Plot box plot, one per metric, x-axis is node', action='store_const', const=True, default=False)

    parser.add_argument('--scatterplots', dest='scatterplots',
                        help='Plot scatter plots', action='store_const', const=True, default=False)

    parser.add_argument('--full-timeline', dest='full_timeline',
                        help='Plot full timeline of raw data, png for frequency, pwr, temperature', type=str, nargs='+')

    parser.add_argument('--zoomed-timeline', dest='zoomed_timeline',
                        help='Plot zoomed timeline of raw data, png for frequency, pwr, temperature', type=str, nargs='+')

    parser.add_argument('--include-points', dest='include_points',
                        help='Include individual points on box plot (default=False)', action='store_const', const=True, default=False)

    parser.add_argument('--get-raw-data-kernels', dest='get_raw_data_kernels',
                        help='Get data about start times of kernels in raw csv file', type=str, nargs='+')

    # Little bit of pre-processing of arguments
    args = parser.parse_args()
    cluster = args.cluster
    path = args.path
    boxplot_no_breakdown = args.boxplot_no_breakdown
    boxplot_node_breakdown = args.boxplot_node_breakdown
    scatterplots = args.scatterplots
    full_timeline = args.full_timeline
    zoomed_timeline = args.zoomed_timeline
    include_points = args.include_points
    get_raw_data_kernels = args.get_raw_data_kernels


def handle_args(df):

    # --boxplot-no-breakdown: one boxplot per metric, no visual aggregation
    if boxplot_no_breakdown:
        plot_boxplots(df=df, include_points=include_points,
                      hue='cabinet', hue_order=tacc_cab, breakdown=None, png_name='no-breakdown')

    # --boxplot-node-breakdown: one boxplot per metric, x-axis is node
    if boxplot_node_breakdown:
        df['node'] = df['node'].str[6:]
        plot_boxplots(df=df, include_points=include_points, hue=None, hue_order=None,
                      breakdown='node', png_name='node-breakdown')

    # --scatterplots: make scatterplots of all data, visually color by node
    if scatterplots:
        plot_scatterplots(df=df, hue='cabinet', hue_order=tacc_cab)

    # --timeline: make timeline plots of raw data for frequency, temperature, and power
    if full_timeline != None:
        plot_full_timeline(full_timeline[0])

    # --zoomed-timeline: make zoomed in timeline plots of raw data for frequency, temperature, and power
    if zoomed_timeline != None:
        plot_zoomed_timeline(zoomed_timeline[0])

    if get_raw_data_kernels:
        raw_data_kernel_timestamps(get_raw_data_kernels[0])


def main():
    # Parse command line arguments
    create_command_line_args()

    # Read in aggregated data file
    df = pd.read_csv(path)
    # df['perf'] = df['median_iter_dur'].multiply((10**3))
    # df['mean_iter_dur'] = df['mean_iter_dur'].multiply((10**3))
    # df['min_iter_dur'] = df['min_iter_dur'].multiply((10**3))
    # df['max_iter_dur'] = df['max_iter_dur'].multiply((10**3))
    # df['dcefclk_mhz'] = df.dcefclk.map(dcefclk_mhz_mappings)
    # df['fclk_mhz'] = df.fclk.map(fclk_mhz_mappings)
    # df['mclk_mhz'] = df.mclk.map(mclk_mhz_mappings)
    # df['sclk_mhz'] = df.sclk.map(sclk_mhz_mappings)
    # df['socclk_mhz'] = df.socclk.map(socclk_mhz_mappings)
    # df.to_csv(path)

    # Handle Arguments
    # exclude_first_run(df)
    handle_args(df)


if __name__ == '__main__':
    main()


# Performance vs. Frequency
# plot = sns.scatterplot(data=df, x='perf',
#                        y='freq', hue=hue, s=50)
# plot.yaxis.set_major_locator(ticker.MultipleLocator(50))
# plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel(axis_titles['perf'], fontsize=16,
#            fontweight=600, labelpad=15)
# plt.ylabel(axis_titles['freq'], fontsize=16,
#            fontweight=600, labelpad=15)
# # plt.legend([], [], frameon=False)
# plt.plot(rasterized=True)
# plt.xticks(rotation=45)
# plt.savefig(filename + '-perf-freq-' + breakdown + '-scatterplot.png',
#             bbox_inches='tight', dpi=300)
# plt.close()
# print('Plotted perf v. freq')

# Median Iteration Duration v. Max Power
# plot = sns.scatterplot(
#     data=df, x='perf', y='max_pwr', hue=hue, s=50)
# # plot.yaxis.set_major_locator(ticker.MultipleLocator(50))
# # plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel(axis_titles['perf'], fontsize=16,
#            fontweight=600, labelpad=15)
# plt.ylabel(axis_titles['max_pwr'], fontsize=16,
#            fontweight=600, labelpad=15)
# # plt.legend([], [], frameon=False)
# plt.plot(rasterized=True)
# plt.xticks(rotation=45)
# plt.savefig(filename + '-median-iter-dur-max-pwr-' + breakdown + '-scatterplot.png',
#             bbox_inches='tight', dpi=300)
# plt.close()
# print('Plotted perf v. max_pwr')

# Median Power v. Median Temp
# plot = sns.scatterplot(data=df, x='pwr', y='temp',
#                        hue=hue, s=50)
# # plot.yaxis.set_major_locator(ticker.MultipleLocator(10))
# # plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel(axis_titles['pwr'], fontsize=16,
#            fontweight=600, labelpad=15)
# plt.ylabel(axis_titles['temp'], fontsize=16,
#            fontweight=600, labelpad=15)
# # plt.legend([], [], frameon=False)
# plt.plot(rasterized=True)
# plt.xticks(rotation=45)
# plt.savefig(filename + '-median-pwr-median-temp-' + breakdown + '-scatterplot.png',
#             bbox_inches='tight', dpi=300)
# plt.close()
# print('Plotted pwr v. temp')

# Max Power v. Median Temp
# plot = sns.scatterplot(data=df, x='max_pwr', y='temp',
#                        hue=hue, s=50)
# # plot.yaxis.set_major_locator(ticker.MultipleLocator(10))
# # plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel(axis_titles['max_pwr'], fontsize=16,
#            fontweight=600, labelpad=15)
# plt.ylabel(axis_titles['temp'], fontsize=16,
#            fontweight=600, labelpad=15)
# # plt.legend([], [], frameon=False)
# plt.plot(rasterized=True)
# plt.xticks(rotation=45)
# plt.savefig(filename + '-max-pwr-median-temp-' + breakdown + '-scatterplot.png',
#             bbox_inches='tight', dpi=300)
# plt.close()
# print('Plotted max_pwr v. temp')

# Median Power v. Median Freq
# plot = sns.scatterplot(data=df, x='median_pwr', y='freq',
#                        hue=hue, s=50)
# # plot.yaxis.set_major_locator(ticker.MultipleLocator(10))
# # plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel(axis_titles['median_pwr'], fontsize=16,
#            fontweight=600, labelpad=15)
# plt.ylabel(axis_titles['freq'], fontsize=16,
#            fontweight=600, labelpad=15)
# # plt.legend([], [], frameon=False)
# plt.plot(rasterized=True)
# plt.xticks(rotation=45)
# plt.savefig(filename + '-median-pwr-median-freq-' + breakdown + '-scatterplot.png',
#             bbox_inches='tight', dpi=300)
# plt.close()
# print('Plotted median_pwr v. freq')

# # Max Power v. Median Freq
# plot = sns.scatterplot(data=df, x='max_pwr', y='freq',
#                        hue=hue, s=50)
# # plot.yaxis.set_major_locator(ticker.MultipleLocator(10))
# # plot.yaxis.set_major_formatter(ticker.ScalarFormatter())
# plt.xlabel(axis_titles['max_pwr'], fontsize=16,
#            fontweight=600, labelpad=15)
# plt.ylabel(axis_titles['freq'], fontsize=16,
#            fontweight=600, labelpad=15)
# # plt.legend([], [], frameon=False)
# plt.plot(rasterized=True)
# plt.xticks(rotation=45)
# plt.savefig(filename + '-max-pwr-median-freq-' + breakdown + '-scatterplot.png',
#             bbox_inches='tight', dpi=300)
# plt.close()
# print('Plotted max_pwr v. freq')
