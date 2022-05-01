all: sgemm dgemm gen_data

gen_data: gen_data.cpp
	g++ -o $@ $<

sgemm_debug: simple_sgemm.cu
	nvcc -g -G -o $@ $< -lcurand -lcudart -lcublas

sgemm: simple_sgemm.cu
	nvcc -arch=compute_70 -code=sm_70 -o $@ $< -lcurand -lcudart -lcublas

dgemm: simple_dgemm.cu
	nvcc -arch=compute_70 -code=sm_70 -o $@ $< -lcurand -lcudart -lcublas

clean:
	rm sgemm dgemm
