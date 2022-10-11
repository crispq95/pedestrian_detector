##############################################################################
## THE PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
## OF ANY KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, 
## ANY WARRANTIES ON ITS, NON-INFRINGEMENT, MERCHANTABILITY, SECURED, 
## INNOVATIVE OR RELEVANT NATURE, FITNESS FOR A PARTICULAR PURPOSE OR 
## COMPATIBILITY WITH ANY EQUIPMENT OR SOFTWARE.
## In the event of publication, the following notice is applicable:
##
##              (C) COPYRIGHT 2010 THALES RESEARCH & TECHNOLOGY
##                            ALL RIGHTS RESERVED
##              (C) COPYRIGHT 2012 Universitat Politècnica de Catalunya
##                            ALL RIGHTS RESERVED
##
## The entire notice above must be reproduced on all authorized copies.
##
##
## Title:             Makefile pedestrian detection application
##
## File:              Makefile
## Authors:           Paul Brelet  <paul.brelet@thalesgroup.com>
##                    Matina Maria Trompouki  <mtrompou@ac.upc.edu>
##					  Alvaro Jover-Alvarez <alvaro.jover@bsc.es>
##					  Cristina Peralta Quesada 
## 			dpcpp -g
##					   nvprof --print-gpu-trace -o profile_output_detailed.nvvp 
## /opt/nvidia/hpc_sdk/Linux_x86_64/22.2/compilers/bin/ -ta=tesla:cc61,managed  
## -unroll-count=8 -Rpass=unroll :managed
############################################################################### 

LIBRARY= -lm -std=c99  
ACCFLAGS = -acc=gpu -ta=tesla:pinned -Munroll -Minline -Minfo=acc

all: violajones 
c: violajones
acc: violajones_acc
sycl: violajones_sycl

violajones: violajones.c violajones.h
	mkdir -p bin
	gcc -g -Wall -L. -o ./bin/violajones violajones.c $(LIBRARY)
 
violajones_acc: violajones_acc.c violajones.h
	mkdir -p bin
	/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/compilers/bin/nvc++ -g -L. -o ./bin/violajones_acc violajones_acc.c $(LIBRARY) $(ACCFLAGS) 
#-fast 

violajones_sycl: violajones_sycl.cpp violajones.h
	mkdir -p bin 
	/home/cperalta/sycl_workspace/llvm/build/bin/clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -arch=sm_71 -g -Wall  -L. -O3  -o ./bin/violajones_sycl violajones_sycl.cpp -lm -std=c++17 

clean: 
	rm -rf bin/

run: all
	./bin/violajones classifier.txt ./dataset/*.pgm

run_acc: acc
	./bin/violajones_acc classifier.txt ./dataset/*.pgm

run_sycl: sycl
	./bin/violajones_sycl classifier.txt ./dataset/*.pgm
