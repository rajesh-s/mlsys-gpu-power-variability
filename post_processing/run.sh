#!/bin/bash

for i in {12..17}
do
	eval "python3 nvprof-aggregator.py ../final_results/case${i}/ cloudlab"
	eval "python3 nvprof-plotter.py --full-timeline ../final_results/case${i}/"
	eval "python3 temp-plotter.py --full-timeline ../final_results/case${i}/"
	eval "mv aggregate.csv ../aggregated-data/case${i}.csv"
done
