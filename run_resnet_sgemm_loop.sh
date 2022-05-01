#!/bin/bash
RESNET_GPU=0
RESNET1_GPU=1
RESNET2_GPU=2
RESNET3_GPU=3
RESNET_EPOCHS=3

{ time nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file resnet.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f python ./main.py --arch resnet50 --data-backend pytorch --batch-size 128 --epochs ${RESNET_EPOCHS} --gpu_core ${RESNET_GPU} --label-smoothing 0.1 /imagenet > resnet.out ; } &>> resnet.out &
RESNET_PID=$!
{ time nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file resnet1.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f python ./main.py --arch resnet50 --data-backend pytorch --batch-size 128 --epochs ${RESNET_EPOCHS} --gpu_core ${RESNET1_GPU} --label-smoothing 0.1 /imagenet > resnet1.out ; } &>> resnet1.out &
RESNET1_PID=$!
{ time nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file resnet2.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f python ./main.py --arch resnet50 --data-backend pytorch --batch-size 128 --epochs ${RESNET_EPOCHS} --gpu_core ${RESNET2_GPU} --label-smoothing 0.1 /imagenet > resnet2.out ; } &>> resnet2.out &
RESNET2_PID=$!
#{ time nvprof --print-gpu-trace --event-collection-mode continuous --system-profiling on --kernel-latency-timestamps on --csv --log-file resnet3.csv --device-buffer-size 128 --continuous-sampling-interval 1 -f python ./main.py --arch resnet50 --data-backend pytorch --batch-size 128 --epochs ${RESNET_EPOCHS} --gpu_core ${RESNET3_GPU} --label-smoothing 0.1 /imagenet > resnet3.out ; } &>> resnet3.out &
#RESNET3_PID=$!


for i in {1..2}
do
	./run_sgemm.sh $i
done

wait $RESNET_PID
wait $RESNET1_PID
wait $RESNET2_PID
#wait $RESNET3_PID
tar -czvf resnet.tar.gz resnet.csv resnet1.csv resnet2.csv
mkdir new_results
mv *.tar.gz *.csv *.out new_results
cp *.sh new_results
