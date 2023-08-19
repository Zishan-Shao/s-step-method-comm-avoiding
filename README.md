# s-step-method-comm-avoiding
s-step method communication avoiding algorithms

Workflow:

1. Preparing for datasets
   - all datasets are available in libsvm official website: [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
   - datasets used in this experiment: duke breast-cancer, mnist (scaled), news20.binary

2. Compile the C files
   - loading the intel compiler intel/2023.1.0 (on NERSC, use command: [1] module unload cray-mpich; [2] module load PrgEnv-intel)
   - inside each C file, edit the output file directory (especially the name !!!) for instance, we need to change the filename in command fp = fopen("caksvm_data_mnist_gauss.csv", "a"); so it will be saved to correct output file directory
   - compile & optimize the files via following command: cc -qmkl="sequential" -O2 -qopt-report=3 -qopt-report-phase=vec -vec -w [bdcd_new_gauss.c] -o [bdcd_mnist_linear]

3. submit the jobs (iteratively) via .slurm
   - Copy [1] all files in output_data and [2] all compiled executables (regardless of the order) to scratch directory, please do not copy the entire directory but in separate files!
   - enter the directory where *for_loop_submission.slurm* lies, adjust the resources to be allocated (memory, nodes, etc.)
   - edit the settings such as s values, num processes, maximum iterations (MAXIT)
   - submit the job via: sbatch for_loop_submission.slurm

