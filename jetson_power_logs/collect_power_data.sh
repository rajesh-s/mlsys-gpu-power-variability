#!/bin/bash

if [ "$EUID" -ne 0 ]
	then echo "Please run as root"
	exit 1
fi

application=$1

matrix_dim=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
num_elements=15

num_model=0

while [ $num_model -le 4 ]
do
	sudo nvpmodel -m $num_model
	j=0
	while [ $j -le $num_elements ]
	do
		./log_power_data.sh &> ${application}_power_nvpmodel_${num_model}_${matrix_dim[j]}.tsv & { time ./${application} ${matrix_dim[j]} 1 0 &> /dev/null ; } &> ${application}_time_nvpmodel_${num_model}_${matrix_dim[j]}.txt
		kill $!
		j=$(($j + 1))
	done

	sudo ~/jetson_clocks.sh
	j=0
	while [ $j -le $num_elements ]
	do
		./log_power_data.sh &> ${application}_jc_power_nvpmodel_${num_model}_${matrix_dim[j]}.tsv & { time ./${application} ${matrix_dim[j]} 1 0 &> /dev/null ; } &> ${application}_jc_time_nvpmodel_${num_model}_${matrix_dim[j]}.txt
		kill $!
		j=$(($j + 1))
	done
	num_model=$(($num_model + 1))
done

sudo nvpmodel -m 0
