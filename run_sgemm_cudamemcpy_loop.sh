#!/bin/bash
for i in {1..2}
do
	./run_sgemm_cudamemcpy.sh $i
	wait
done
mkdir new_results
mv *.csv *.out new_results
cp *.sh new_results
