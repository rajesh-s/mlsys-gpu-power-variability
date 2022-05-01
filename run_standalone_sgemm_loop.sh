#!/bin/bash

# Usage: ./run-sgemm-nvidia.sh

# Parameter to configure - change these as required!
NUM_KERN=100
DEVICE_ID=0
DEVICE_ID_1=1
DEVICE_ID_2=2
DEVICE_ID_3=3
SIZE=25536

echo "Number of kernels: ${NUM_KERN}"
echo "GPU ID: ${DEVICE_ID}"
echo "Matrix size: ${SIZE}"

# Get UUID of GPU SGEMM kernels are run on
UUID_list=(`nvidia-smi -L | awk '{print $NF}' | tr -d '[)]'`)
UUID=${UUID_list[${1}]}
echo "GPU UUID: ${UUID}"

# Get timestamp at the time of the run
ts=`date '+%s'`

# File name
FILE_NAME=sgemm_nvidia_${SIZE}_${NUM_KERN}_${DEVICE_ID}
FILE_NAME_1=sgemm_nvidia_${SIZE}_${NUM_KERN}_${DEVICE_ID_1}
FILE_NAME_2=sgemm_nvidia_${SIZE}_${NUM_KERN}_${DEVICE_ID_2}
FILE_NAME_3=sgemm_nvidia_${SIZE}_${NUM_KERN}_${DEVICE_ID_3}
echo "Output file name: ${FILE_NAME}"

# Run application with profiling via nvprof
#echo ""
#echo "Generating 2 matrices of size ${SIZE}. This will take a few minutes."
#./gen_data ${SIZE}
#echo "Completed generating 2 matrices"

echo ""
echo "Running ${NUM_KERN} kernels of SGEMM on GPU ${DEVICE_ID}. This will takes a few minutes."
for i in {1..10} 
do 
__PREFETCH=off nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME}_$i.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm ${SIZE} ${NUM_KERN} ${DEVICE_ID} > ${FILE_NAME}_$i.out &
__PREFETCH=off nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME_1}_$i.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm ${SIZE} ${NUM_KERN} ${DEVICE_ID_1} > ${FILE_NAME_1}_$i.out &
__PREFETCH=off nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME_2}_$i.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm ${SIZE} ${NUM_KERN} ${DEVICE_ID_2} > ${FILE_NAME_2}_$i.out &
__PREFETCH=off nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file ${FILE_NAME_3}_$i.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f ./sgemm ${SIZE} ${NUM_KERN} ${DEVICE_ID_3} > ${FILE_NAME_3}_$i.out &
done
wait
mkdir new_results
mv *.csv *.out new_results
echo "Completed SGEMM. Outputs in ../out"
