echo "TIME	TOTAL	SOC	CPU	GPU	DDR"
START_TIME=$(bc <<< "scale=10; $(date +%s%N)/1000000000")
while :
do
	#echo $((10#$(date +%N)/1000000000))
    #CUR_TIME=$(bc <<< "scale=10; $(date +%s) + $(bc <<< "scale=10; $(date +%N)/1000000000")")
	sleep 0.47 &
	CUR_TIME=$(bc <<< "scale=10; $(date +%s%N)/1000000000")
	TIME=$(bc <<< "scale=10; ${CUR_TIME} - ${START_TIME}")
    TOTPOW=$(cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio\:device1/in_power0_input) #To record total power consumption
	SOCPOW=$(cat /sys/bus/i2c/drivers/ina3221x/0-0040/iio\:device0/in_power1_input) #To record SOC power consumption = CPU + GPU power consumption
	CPUPOW=$(cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio\:device1/in_power1_input) #To record CPU power consumption
	GPUPOW=$(cat /sys/bus/i2c/drivers/ina3221x/0-0040/iio\:device0/in_power0_input) #To record GPU power consumption
	DDRPOW=$(cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio\:device1/in_power2_input) #To record DDR power consumption
	echo "${TIME}	${TOTPOW}	${SOCPOW}	${CPUPOW}	${GPUPOW}	${DDRPOW}"
	wait
done

