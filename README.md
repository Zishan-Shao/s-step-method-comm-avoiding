# s-step-method-comm-avoiding

Dual Coordinate Descent (DCD) and Block Dual Coordinate Descent (BDCD) are important iterative methods for solving convex optimization problems. In this work, we develop scalable DCD and BDCD methods for the kernel support vector machines (K-SVM) and kernel ridge regression (K-RR) problems. On distributed-memory parallel machines the scalability of these methods is limited by the need to communicate every iteration. On modern hardware where communication is orders of magnitude more expensive, the running time of the DCD and BDCD methods is dominated by communication cost. We address this communication bottleneck by deriving s-step variants of DCD and BDCD for solving the K-SVM and K-RR problems, respectively. The s-step variants reduce the frequency of communication by a tunable factor of s at the expense of additional bandwidth and computation. The s-step variants compute the same solution as the existing methods in exact arithmetic. We perform numerical experiments to illustrate that the s-step variants are also numerically stable in finite-arithmetic, even for large values of s. We perform theoretical analysis to bound the computation and communication costs of the newly designed variants, up to leading order. Finally, we develop high performance implementations written in C and MPI and present scaling experiments performed on a Cray EX cluster. The new s-step variants achieved strong scaling speedups of up to 9.8× over existing methods using up to 512 cores.


For more details regarding this research project, see: Shao, Zishan, and Aditya Devarakonda. "Scalable Dual Coordinate Descent for Kernel Methods." arXiv preprint arXiv:2406.18001 (2024).


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

