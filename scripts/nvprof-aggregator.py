'''
    File name: nvprof.py
    Author: Prasoon Sinha
    Last Edited: 4/22/22
    Python Version: 3.7
'''

import pandas as pd
import sys
import numpy as np
import math
import collections
import os
import json
import re
from explorer import read_nvprof_gpu_trace, system_types
from itertools import islice
import datetime
import time


def chunk_dict(data, SIZE=1100):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def get_iter_dur(ts, cabinet, node, device, data_dir):
    path = data_dir + 'resnet_iterdur_' + ts + '_' + cabinet + '-' + node + '.txt'
    with open(path, 'r') as f:

        lines = f.readlines()

        iter_dur_vals = []
        final_data = []

        for line in lines:
            if 'Iteration' in line:
                datas = line.split('I')
                for data in datas:
                    if data == '':
                        continue
                    else:
                        x = data.split(' ')
                        if x[3] == str(device):
                            iter_dur_vals.append(float(x[5].split('\n')[0]))
            elif 'Training time' in line:
                datas = line.split(' ')
                train_time = datas[2].split('\n')[0].split('.')
                x = time.strptime(train_time[0].split(',')[0], '%H:%M:%S')
                seconds = datetime.timedelta(
                    hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
                ms = '0.' + train_time[1]
                final_data.append(seconds + float(ms))
        f.close()

        avg_iter_dur = sum(iter_dur_vals) / len(iter_dur_vals)
        final_data.append(avg_iter_dur)
        return final_data


def csv_aggregator(df_dict, cluster, data_dir):

    # Dictionary that will be returned in final_collection
    aggregate_data_dict = {}

    # Initial set up of aggregate_data_dict
    # We realize this initialization isn't necessary, but
    # include it to make it easier to read what keys are in the dict
    aggregate_data_dict['exp'] = []
    # aggregate_data_dict['input_size'] = []
    # aggregate_data_dict['reps'] = []
    # aggregate_data_dict['uuid'] = []
    if cluster == 'summit':
        aggregate_data_dict['cabinet'] = []
        aggregate_data_dict['node'] = []
        aggregate_data_dict['row'] = []
        aggregate_data_dict['col'] = []
        aggregate_data_dict['ts'] = []
    elif cluster == 'tacc':
        aggregate_data_dict['ts'] = []
        aggregate_data_dict['cabinet'] = []
        aggregate_data_dict['node'] = []
        aggregate_data_dict['device'] = []
    elif cluster == 'vortex':
        aggregate_data_dict['ts'] = []
    elif cluster == 'cloudlab':
        #     aggregate_data_dict['max_freq_set'] = []
        #     aggregate_data_dict['max_pwr_set'] = []
        aggregate_data_dict['ts'] = []
    #   aggregate_data_dict['kern'] = []
    aggregate_data_dict['kern_min_start'] = []
    aggregate_data_dict['kern_max_start'] = []
    aggregate_data_dict['num_kerns'] = []
    aggregate_data_dict['kern_sum'] = []
    
    for item, val in system_types.items():
        aggregate_data_dict[item] = []

    for k, v in df_dict.items():
        # Split the name of the csv file
        li = k.split('_')
        # Populate meta data in the dictionary
        aggregate_data_dict['exp'].append(li[0])
        # aggregate_data_dict['input_size'].append(li[1])
        # aggregate_data_dict['reps'].append(li[2][3])
        # aggregate_data_dict['uuid'].append(li[1])
        # aggregate_data_dict['device'].append(v.device)
        # aggregate_data_dict['cabinet'].append(li[3][0:4])
        # aggregate_data_dict['node'].append(li[3][5:])
        aggregate_data_dict['ts'].append(li[2])

        GPU = v.data.Device.min()
        #device_id = v.device
        device_id = int(li[4])
        GPU = GPU[:-2] + str(device_id) + ')'
        df = v.data

        # Start time of first kernel
        min_start = df[(df.Device == GPU) & (
            df.GridX.isnull() == False)].Start.min()

        aggregate_data_dict['kern_min_start'].append(min_start)

        # Start time of last kernel
        max_start = df[(df.Device == GPU) & (
            df.GridX.isnull() == False)].Start.max()

        aggregate_data_dict['kern_max_start'].append(max_start)

        # Number of kernels for this particular run of the benchmark
        # Count rows with GridX not NaN adn duration greater than 2 ms
        count = df[(df.Device == GPU) & (df.GridX.isnull() == False)
                   & (df.Duration > 2.00)].Start.count()
        aggregate_data_dict['num_kerns'].append(count)

        a = df[(df.Device == GPU)]
        b = a[(df.GridX.isnull() == False) & (df.Duration > 2.00)]

        # Sum of duration of all kernel instances in ms
        aggregate_data_dict['kern_sum'].append(
            b.Duration.sum())

        # Median value for each metric (frequency, memory frequency, power, temperature) for each run of benchmark
        for item, val in system_types.items():
            aggregate_data_dict[item].append(
                a[a.Name == val[0]][item].median(skipna=True))

        print("Completed aggregation for: " + k)

    return aggregate_data_dict


def handle_data(data_dir, cluster):
    # For all the csv files in the directory, populate dictionary with the file name followed by its path
    base_file_dict = {}
    for f in os.listdir(data_dir):
        if "csv" in f:
            li = f.split(".")
            key = "_".join(i for i in li if i != 'csv')
            base_file_dict[key] = os.path.join(data_dir, f)

    # Print number of csv files in directory
    print("The number of csv files in this directory is: " +
          str(len(base_file_dict)))

    # Read each csv and aggregate
    chunk = 0
    dir = data_dir.split('/')
    for file_dict in chunk_dict(base_file_dict):
        # k (key) - the csv file name
        # v (value) - the csv file path
        df_dict = {}
        for k, v in file_dict.items():
            tmp = read_nvprof_gpu_trace(v)
            # print(tmp)
            # Some files may be empty
            if tmp is not None:
                df_dict[k] = tmp
        collection = csv_aggregator(df_dict, cluster, data_dir)
        for key in collection:
            print(key + ": " + str(len(collection[key])))
        aggregated_data = pd.DataFrame(collection)
        # chunk += 1
        aggregated_data.to_csv('aggregate.csv')
        # aggregated_data.to_csv('<Place path to save aggregated data csv file here')
        # print("Completed reading and aggregations for: " + str(chunk * 1100))


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 gpu-aggregator.py [path to data directory] [cluster]")
        exit()

    # Path to directory containing csv files to read and aggregate
    data_dir = sys.argv[1]
    # Cluster - summit, vortex, tacc, cloudlab, etc.
    cluster = sys.argv[2]

    # Read and aggregate data
    handle_data(data_dir, cluster)


if __name__ == "__main__":
    main()
