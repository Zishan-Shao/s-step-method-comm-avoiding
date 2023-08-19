#!/bin/bash
#SBATCH --qos=regular   # shared   # 
#SBATCH --job-name=CA_kernel_ridge
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --constraint=cpu
#SBATCH --account=m4293
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=shaoz20@wfu.edu
#SBATCH --output=/global/homes/z/zishan24/CA_Kernel_ML/kernel_ridge/outputs/trial-%j.o
#SBATCH --error=/global/homes/z/zishan24/CA_Kernel_ML/kernel_ridge/outputs/trial-%j.e
#SBATCH --mem=300gb




#ESTABLISH ENVIRONMENT SETTINGS AND MODULES
cd "/pscratch/sd/z/zishan24/"


module unload cray-mpich
module unload cray-libsci
module load PrgEnv-intel


# Define the arrays
s_values=(64 32 16 8 4 2)
BLKSIZE=(1 2)   # 16 32 64 128
num_processes=(512 256 128)   #(512 256 128 64 32) # (1152 1024 896 768 640) # 512 384 256 128 64 32 16 8 4 2
MAXIT=640

# Loop through s_values and BLKSIZE
for nump in "${num_processes[@]}"
do
    for blksize in "${BLKSIZE[@]}"
    do  

        # Calculate MAXIT
        MAXIT=$((640 / blksize))

        for s in "${s_values[@]}"
        do
            echo "========================================"
            echo "Trial (MNIST Linear): Process=$nump, s=$s, BLKSIZE=$blksize, MAXIT=$MAXIT"
            echo "========================================"

            # Submit parallel job using mpirun
            # input order: BLKSIZE, s, MAXIT
            srun -n "$nump" para_cabdcd_mnist_linear "$blksize" "$s" "$MAXIT"
        done

         # run the bdcd after
        srun -n "$nump" para_bdcd_mnist_linear "$blksize" "$MAXIT"
    
    done
done