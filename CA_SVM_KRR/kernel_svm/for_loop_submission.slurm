#!/bin/bash
#SBATCH --qos=regular   # shared   # 
#SBATCH --job-name=CA_kernel_svm
#SBATCH --time=12:00:00
#SBATCH --nodes=5
#SBATCH --constraint=cpu
#SBATCH --account=m4293
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=shaoz20@wfu.edu
#SBATCH --output=/global/homes/z/zishan24/CA_Kernel_ML/kernel_svm/outputs/trial-%j.o
#SBATCH --error=/global/homes/z/zishan24/CA_Kernel_ML/kernel_svm/outputs/trial-%j.e
#SBATCH --mem=300gb




#ESTABLISH ENVIRONMENT SETTINGS AND MODULES
cd "/pscratch/sd/z/zishan24/"


module unload cray-mpich
module unload cray-libsci
module load PrgEnv-intel


# Define the arrays
s_values=(64 32 16 8 4 2)   # 16 32 64 128
# 1280 1088 1024 960 768 640 512 256 128 64 32
num_processes=(640) # (1280 1152 1088 1024 960 896 832 ) # (768 704 640 576 512 448 384 320 256 128 64 32)
# iter=(1 2 3 4 5)
MAXIT=640
# MIN_NNODE=4

# Loop through s_values and BLKSIZE
for nump in "${num_processes[@]}"
do

    # for i in "${iter[@]}"
    # do
    # Calculate MAXIT
    # MIN_NNODE=$((nump / 128))

        for s in "${s_values[@]}"
        do

                echo "============================================="
                echo "Trial SVM (NEWS Gauss) iter $i : Process=$nump, s=$s, MAXIT=$MAXIT"
                echo "============================================="

                # Submit parallel job using mpirun
                # input order: BLKSIZE, s, MAXIT
                # srun -n "$nump" -N "$MIN_NNODE" caksvm_news_gauss "$s" "$MAXIT"
                srun -n "$nump" caksvm_news_gauss "$s" "$MAXIT"
        done   
        
        # run the bdcd after
        # srun -n "$nump" -N "$MIN_NNODE" ksvm_news_gauss "$MAXIT"
        srun -n "$nump" ksvm_news_gauss "$MAXIT"
    # done
    
done