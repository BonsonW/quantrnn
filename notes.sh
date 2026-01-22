# commands for profiling
ncu -f --set full --export profile_rep python bench_ampere.py 

# check occupancy of tensor cores
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed_pipe_tensor_op_imma.sum,gpu__time_duration.sum python bench_ampere.py 

# check warps active
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed python bench_ampere.py 

# check whether limited by registers or shared memory
ncu --metrics launch__registers_per_thread,launch__shared_mem_per_block_static,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem python bench_ampere.py 


# troubleshooting

# if frozen at compiling cuda
rm -rf ~/.cache/torch_extensions/