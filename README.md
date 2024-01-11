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