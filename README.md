# versalfpga

```
# on any machine with vitis ai
singularity run --nv --bind /cvmfs ../vitis-ai-pytorch-cpu_latest.sif # nv and build not necessary
git clone https://github.com/badeaa3/versalfpga.git
cd versalfpga
python cnn.py -q calib # quant_mode = "calib", builds part of model
python cnn.py -q test # quant_mode = "test", builds xmodel file
vai_c_xir -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -x quant_dir/SimpleCNN_int.xmodel -o ./compiled -n SimpleCNN # build compiled xmodel file specific that is ready for architecture -a 

# scp to the fpga board (VCK190 as specified in vai_c_xir)
scp -O -r compiled/ root@10.120.102.89:/home/root/badea/test
scp -O cnn_run.py root@10.120.102.89:/home/root/badea/test

# log onto board
ssh root@10.120.102.89
cd /home/root/badea/test
python3 -m vaitrace_py --txt ./cnn_run.py 1 compiled/SimpleCNN.xmodel

# output in folder
root@xilinx-vck190-20222:~/badea/test# ls -lrth
total 52K
drwxr-xr-x 2 root root 4.0K Jan 11 09:14 compiled
-rwxr-xr-x 1 root root 6.9K Jan 11 09:14 cnn_run.py
-rw-r--r-- 1 root root 7.6K Jan 11 09:14 vart_trace.csv
-rw-r--r-- 1 root root  574 Jan 11 09:14 profile_summary.csv
-rw-r--r-- 1 root root  15K Jan 11 09:14 vitis_ai_profile.csv
-rw-r--r-- 1 root root 4.8K Jan 11 09:14 summary.csv
-rw-r--r-- 1 root root  727 Jan 11 09:14 xrt.run_summary
```

```
root@xilinx-vck190-20222:~/badea/test# python3 -m vaitrace_py --txt ./cnn_run.py 1 compiled/SimpleCNN.xmodel
INFO:root:VART will run xmodel in [NORMAL] mode
Analyzing symbol tables...
63 / 63
81 / 81
6 / 6
45 / 45
3 / 3
INFO:root:vaitrace compile python code: ./cnn_run.py
INFO:root:vaitrace exec poython code: ./cnn_run.py
final return
1
XAIEFAL: INFO: Resource group Avail is created.
XAIEFAL: INFO: Resource group Static is created.
XAIEFAL: INFO: Resource group Generic is created.
1
10
FPS=11372.07, total frames = 360.00 , time=0.031657 seconds
NOC Stop Collecting
XAIEFAL: INFO: Resource group Avail is created.
XAIEFAL: INFO: Resource group Static is created.
XAIEFAL: INFO: Resource group Generic is created.
INFO:root:Generating ascii-table summary
INFO:root:Processing xmodel information
DPU Summary:
================================================================================================================================
DPU Id      | Bat | DPU SubGraph                             | WL    | SW_RT | HW_RT | Effic | LdWB  | LdFM  | StFM  | AvgBw
------------+-----+------------------------------------------+-------+-------+-------+-------+-------+-------+-------+----------
DPUCVDX8G_1 | 6   | SimpleCNN__SimpleCNN_Conv2d_conv1__ret_5 | 0.012 | 0.106 | 0.048 | 2.6   | 0.511 | 0.024 | 0.022 | 11568.417
================================================================================================================================

Notes:
"~0": Value is close to 0, Within range of (0, 0.001)
Bat: Batch size of the DPU instance
WL(Work Load): Computation workload (MAC indicates two operations), unit is GOP
SW_RT(Software Run time): The execution time calculate by software in milliseconds, unit is ms
HW_RT(Hareware Run time): The execution time from hareware operation in milliseconds, unit is ms
Effic(Efficiency): The DPU actual performance divided by peak theoretical performance,unit is %
Perf(Performance): The DPU performance in unit of GOP per second, unit is GOP/s
LdFM(Load Size of Feature Map): External memory load size of feature map, unit is MB
LdWB(Load Size of Weight and Bias): External memory load size of bias and weight, unit is MB
StFM(Store Size of Feature Map): External memory store size of feature map, unit is MB
AvgBw(Average bandwidth): External memory average bandwidth. unit is MB/s
....


CPU Functions(Not in Graph, e.g.: pre/post-processing, vai-runtime):
===========================================================================
Function                               | Device | Runs | AverageRunTime(ms)
---------------------------------------+--------+------+-------------------
vart::TensorBuffer::copy_tensor_buffer | CPU    | 120  | 0.025
xir::XrtCu::run                        | CPU    | 60   | 0.095
===========================================================================
```