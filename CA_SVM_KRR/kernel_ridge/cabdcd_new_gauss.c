#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <mkl_lapacke.h>
#include <mkl_spblas.h>
#include <mpi.h>

// in this script: N is number of observations, M is number of features

void printMatrix();
double *random_array();
void linear();
void polynomial();
void gaussian();
int* rand_index();
void createIdentityMatrix();
void subset_identity();
void indexOverlap();
void csr_setup_bdcd();
void read_data_sparse_bdcd();
void sparse_sample_A_bdcd();
double objective_value_cabdcd();



typedef struct {
    int *row_b;  // record where the row, start 0
    int *row_e;  // where the row end, end num_row - 1
    int *col;
    double *val;
    double *label;
    int nnz;  // record number of nnz
} Element;


int main(int argc, char* argv[]) {

    if(argc != 4) {
        printf("Error: invalid inputs\n");
        return 1;
    }
    
    /*
    char filename[50] = "a9a.txt";// "a9a_short.txt";// "a9a.txt";

    int N = 32561;   // n is num observation, m is num features
    int M = 123;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    int BLKSIZE = atoi(argv[1]); // 1;  // always less than n
    char KERNEL[10] = "poly";
    int MAXIT = atoi(argv[3]);// 5000;
    double GAMMA = 0.0001;
    int seed = 42;
    double LAMBDA = 0.00001;  // 0.0113;
    int degree = 5;
    
    // s in bdcd always 0
    int s = atoi(argv[2]);  // 2, 4, 8, 16, 32, 64
    */

    
    char filename[50] = "mnist.scale";

    int N = 60000;   // n is num observation, m is num features
    int M = 780;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    int BLKSIZE = atoi(argv[1]); // 1;  // always less than n
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[3]);
    double GAMMA = 0.0000005;
    int seed = 42;
    double LAMBDA =  0.000005;  // 0.00001 works with blksize = 100, s = 12;
    int s = atoi(argv[2]); 
    int degree = 5;

    /*
    char filename[50] = "news20.binary";

    int N = 19996;   // n is num observation, m is num features
    int M = 1355191;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    int BLKSIZE = atoi(argv[1]);  // always less than n
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[3]);
    double GAMMA = 0.0001;
    int seed = 42;
    double LAMBDA =  0.00001;  // 0.0113;
    int degree = 5;
    int s = atoi(argv[2]); 
    */

    /*
    char filename[50] = "duke";// "a9a_short.txt";// "a9a.txt";

    int N = 44;   // n is num observation, m is num features
    int M = 7129;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    int BLKSIZE = atoi(argv[1]); // 1;  // always less than n
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[3]);// 5000;
    double GAMMA = 0.0001;
    int seed = 42;
    double LAMBDA = 0.00001;  // 0.0113;
    int degree = 5;
    
    // s in bdcd always 0
    int s = atoi(argv[2]);  // 2, 4, 8, 16, 32, 64
    */



    srand(seed);

    // initialize the MPI stuff
    MPI_Init(NULL, NULL);

    // ############# Program Start Time ###############
   
    // Get the size of the communicator
    int size = 0;  // define the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    // Get my rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // communicate across all processes


    // open the CSV file for writing
    FILE* fp = NULL;
    if (rank == 0) {
        // /deac/csc/devarakondaGrp/shaoz20/C-Code-krr-cakrr/CSC393/testing/result_data/
        // para_ca_data_newsbinary_poly.csv
        fp = fopen("para_ca_data_E2006_linear.csv", "a");
        if (fp == NULL) {
            printf("Failed to open file for writing\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }


    double runtime = 0.0;
    double memory_reset = 0.0;
    double memory_reset_start = 0.0;
    double memory_reset_end = 0.0;
    
    double runtime_begin = MPI_Wtime();
    
    
    int i, iter;

    int max_line = 5000000;   // note: if one line of feature has more than 2000 characters, need to enlarge it

    memory_reset_start = MPI_Wtime();
    // initialize the struct
    Element* csr = (Element *) malloc(sizeof(Element));

    // allocate memory for the label
    csr->label = (double*) malloc(N * sizeof(double));
    csr->nnz = 0;


    int * idx = (int *) malloc(sizeof(int)*BLKSIZE*s);  // the index array should be BLKSIZE * s size
    int * idx_s = (int *) malloc(sizeof(int)*BLKSIZE); // record part of blksize index of the selected blksize * s indices
    double * r, * alpha, * del_a, * T, * v_alpha, * M_kernel, * k, // * sample_A, 
           * v, //* idx_overlap, 
           * v_sub, * del_a_sub, * result, * result_2;
    
    result = (double *) calloc(1*BLKSIZE, sizeof(double));
    v_sub = (double *) calloc(BLKSIZE*s*BLKSIZE, sizeof(double));
    del_a_sub = (double *) calloc(s*BLKSIZE, sizeof(double));
    result_2 = (double *) calloc(1*BLKSIZE, sizeof(double));

    // create the shift matrix for lambda * I
    T = (double*) calloc(BLKSIZE*BLKSIZE, sizeof(double));
    double * I = (double*) calloc(BLKSIZE*BLKSIZE, sizeof(double));  // we don't need this because we can just do it inside of the function
    createIdentityMatrix(BLKSIZE, I);
    cblas_dscal(BLKSIZE * BLKSIZE, 1.0/N, I, 1);   
    
    M_kernel = (double*) calloc(BLKSIZE*BLKSIZE, sizeof(double));

    // initialize the alpha and del_a
    alpha = (double*) calloc(N, sizeof(double));
    del_a = (double*) calloc(BLKSIZE*s, sizeof(double));


    // r is blk * 1 dimensional in T del_a = r (T is blk*blk)
    r = (double*) calloc(BLKSIZE, sizeof(double));      // I cannot free the r, and I don't know why
    v_alpha = (double*) calloc(BLKSIZE*1, sizeof(double));

    // M_kernel = (double *) calloc(BLKSIZE*BLKSIZE*s*s, sizeof(double)); // [s*blksize, s*blksize]
    k = (double *) calloc(BLKSIZE*N*s, sizeof(double));
    v = (double *) calloc(BLKSIZE*N, sizeof(double));
    
    // sample from A of BLKSIZE rows based on the randomly generated index vector
    // should have BLKSIZE rows and M columns    
    double *r_copy = (double*) malloc(BLKSIZE*sizeof(double));

    // compute the const_caant part for (1/(lambda*m^2))
    double const_ca = 1.0 / (LAMBDA * pow(N, 2));  // not correct, as pow(N), not pow M, differ from the matlab code 
   

    // initialize the struct
    Element* sample_A_sparse = (Element *) malloc(sizeof(Element));

    // Allocate memory for the sparse sample_A
    sample_A_sparse->row_b = (int*) calloc(BLKSIZE*s + 1, sizeof(int));   // as many as number of rows
    sample_A_sparse->row_e = (int*) calloc(BLKSIZE*s + 1, sizeof(int));   // as many as number of rows
    sample_A_sparse->nnz = 0;


    iter = 0;
    int ptr_start = 0;
    int ptr_end = 0;    
    int aa = 0;

    // parallel part declaration
    double * k_para = (double *) calloc(BLKSIZE*N*s, sizeof(double));

    // there will be ceiling( M / size ) of observations, but I shall use the pivot to define where to find the index of the feature
    int sub_blk = 0;
    int pivot = 0;

    int rank_by_mode = M % size;
    int sub_blk_large = 0;
    if (rank_by_mode != 0) {
        sub_blk_large =(M / size) + 1;
    } else {
        sub_blk_large = M / size;    
    }

    int sub_blk_small = M / size;
    
    if (rank < rank_by_mode) {
        sub_blk = sub_blk_large;
        pivot = rank * sub_blk;
    } else {
        sub_blk = sub_blk_small;
        pivot = (rank - rank_by_mode) * sub_blk + rank_by_mode * sub_blk_large;
    }

    // create the feature_indices and feature size
    int * feature_indices = (int *) calloc(sub_blk, sizeof(int));
    
    for(int i = 0; i < sub_blk; ++i) {
        feature_indices[i] = pivot + i;
    }

    if (rank == 0) {
        printf("Basic Matrices Setup Finished! |    %s\n", filename);
    }
    memory_reset_end = MPI_Wtime();
    memory_reset = memory_reset + memory_reset_end - memory_reset_start;



    // ############### csr setup time ##################
    double csr_start_time = MPI_Wtime();
    

    // get the labels and nnz, allocate memory for csr
    csr_setup_bdcd(csr, max_line, filename, N, feature_indices, sub_blk); 

    int total_nnz = 0;   

    MPI_Allreduce(&csr->nnz, &total_nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\n===========================\n");
        printf("Total nnz: %d\n", total_nnz);
        printf("\n===========================\n");
    }  
    
    if (rank == 0) {
        printf("csr setup finished!\n"); 
    }
    // define the b
    double * b = csr->label;

    double csr_end_time = MPI_Wtime();
    double csr_setup_time = csr_end_time - csr_start_time;

    double read_start_time = 0.0;
    double read_end_time = 0.0;
    double csr_read_time = 0.0;
    double sample_start_time= 0.0;
    double sample_end_time = 0.0;
    double sample_time = 0.0;
    double kernel_start_time = 0.0;
    double allreduce_start_time = 0.0;
    double allreduce_end_time = 0.0;
    double allreduce_time = 0.0;
    double kernel_end_time = 0.0;
    double kernel_computation = 0.0;
    double full_kernel_start_time = 0.0;
    double full_kernel_end_time = 0.0;
    double alpha_updating_start = 0.0;
    double alpha_updating_end = 0.0;
    double alpha_updating = 0.0;   // summing the alpha updating time
    double gradient_comp_start = 0.0;
    double gradient_comp_end = 0.0;
    double gradient_comp = 0.0;



    // ############### csr reading time ##################
    read_start_time = MPI_Wtime();

    read_data_sparse_bdcd(csr, max_line, filename, feature_indices, sub_blk);
    
    read_end_time = MPI_Wtime();
    csr_read_time = read_end_time - read_start_time;

    if (rank == 0) {
        printf("CSR Reading Finished!! \n");
    }
    
    // do the norm here
    double * norm = (double *) calloc(N, sizeof(double)); // norm vector should be n sized, as diagonal vector of a [N,N] matrix
    
    if (csr->nnz > 0) {
        
        // double * temp = (double *) calloc(n*blksize, sizeof(double)); 
        // do the for loop here
        // for aj^T aj

        // cblas_dnrm2 (const MKL_INT n, const double *x, const MKL_INT incx);
        // int elts = 0; // specify number of elements in this feature
        double val = 0.0;
        for (int i = 0; i < N; ++i) {
            // elts = csr->row_e[i] - csr->row_b[i];
            // val = cblas_dnrm2(elts, csr, n);
            for (int j = csr->row_b[i]; j < csr->row_e[i]; ++j) {
                val = csr->val[j];
                norm[i] += val * val;
            }

            // norm[i] = val*val;
        }
    }
   

    while (iter < MAXIT) {

        // ############### sample_A setup time ##################
        sample_start_time = MPI_Wtime();

        ptr_start = 0;
        ptr_end = 0;
        aa = 0;

        if (rank == 0) {
            printf("%d ", iter);
        }
        
        
        for (int i = 0; i < s; ++i) { 
                ptr_start = i*BLKSIZE; // because in C, index start from 0, so no plus 1
                
                for (int j = 0; j < BLKSIZE; ++j) {
                    aa = (int) rand() % (N - 0) + 0;
                    idx[ptr_start + j] = aa;
                }
                
        }
        

        if (csr->nnz > 0) {
            sparse_sample_A_bdcd(sample_A_sparse, csr, BLKSIZE*s, sub_blk, idx); 
        }

        sample_end_time = MPI_Wtime();
        sample_time = sample_time + sample_end_time - sample_start_time;
        
        

        // ############### kernel setup time ##################
        kernel_start_time = MPI_Wtime();

        if (sample_A_sparse->nnz > 0 && csr->nnz > 0) {

            if (strcmp(KERNEL, "linear") == 0) {
                // 1. compute the kernelized matrix with size 
                linear(sample_A_sparse, csr, N, sub_blk, BLKSIZE*s, k_para);   // as return a pointer
            }
            else if (strcmp(KERNEL, "poly") == 0) {
                polynomial(sample_A_sparse, csr, N, sub_blk, BLKSIZE*s, k_para, degree); 
            }
            else if (strcmp(KERNEL, "gauss") == 0) {
                gaussian(BLKSIZE*s, N, sub_blk, sample_A_sparse, csr, k_para, GAMMA, idx, norm); 
            }
            
        }

        // ############### kernel setup time ##################
        kernel_end_time = MPI_Wtime();
        kernel_computation = kernel_computation + kernel_end_time - kernel_start_time;


        // ############### AllReduce() time ##################
        allreduce_start_time = MPI_Wtime();


        // Each MPI process sends its rank to reduction, root MPI process collects the result
        MPI_Allreduce(k_para, k, BLKSIZE*s*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        // ############### AllReduce() end time ##################
        allreduce_end_time = MPI_Wtime();
        allreduce_time = allreduce_time + allreduce_end_time - allreduce_start_time;


        full_kernel_start_time = MPI_Wtime();
        if (strcmp(KERNEL, "poly") == 0) {
            // make element wise multiplication
            for (int i = 0; i < N*BLKSIZE*s; ++i) {
                k[i] = pow(k[i], degree) * GAMMA;
            }
        }
        else if (strcmp(KERNEL, "gauss") == 0) {
             // Apply the exponential function element-wise to the k array
            for (int i = 0; i < BLKSIZE*N*s; i++) {
                k[i] = exp(-GAMMA * k[i]);
            }
        }
        full_kernel_end_time = MPI_Wtime();
        kernel_computation = kernel_computation + full_kernel_end_time - full_kernel_start_time;
        
    
        gradient_comp_start = MPI_Wtime();

        for (i = 0; i < s; ++i) {
            
            ptr_start = i*BLKSIZE; // because in C, index start from 0, so no plus 1
            ptr_end = (i + 1)*BLKSIZE;

            for (int j = 0; j < BLKSIZE; ++j) {
                idx_s[j] = idx[ptr_start + j];
            }

            for (int p = 0; p < BLKSIZE; ++p) {
                for (int j = 0; j < BLKSIZE; j ++) {
                    M_kernel[j * BLKSIZE + p] = k[(j+ptr_start) * N + idx_s[p]];  // not 100% sure if correct, suspicious
                }
            }


            // create the Identity matrix for later computation
            // T = (1/(lambda*m^2))*M + (eye(blksize)/m);
            
            // do the summation here
            for (int t = 0; t < BLKSIZE; t++) {
                for (int j = 0; j < BLKSIZE; j++) {
                    T[t * BLKSIZE + j] = const_ca * M_kernel[t * BLKSIZE + j] + I[t * BLKSIZE + j];
                }
            }

    
            // Part 3: r = b(idx_s)/m - alpha(idx_s)/m - (1/(lambda*m^2))*v(ptr_start:ptr_end, :)*alpha;
            for (int p = 0; p < BLKSIZE; ++p) {
                for (int g = 0; g < N; ++g) {
                    v[p*N + g] = k[(ptr_start + p)* N + g];
                }
            }

            // dot produce between the k (BLKSIZE * N) and alpha (N * 1)
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        BLKSIZE, N, 1.0, v, N, alpha, 1, 0.0, v_alpha, 1);


            // compute the final v_alpha matrix with it times with const_ca
            for (int p = 0; p < BLKSIZE; ++p) {
                v_alpha[p] = v_alpha[p] * const_ca;
            }

          
            // use the for loop to compute the final result r
            // r = b(idx_s)/m - alpha(idx_s)/m - (1/(lambda*m^2))*v(ptr_start:ptr_end, :)*alpha;
            for (int p = 0; p < BLKSIZE; ++p) {
                r[p] = (b[idx_s[p]] / (N * 1.0)) - (alpha[idx_s[p]] / (N * 1.0)) - v_alpha[p];
            }
            
            
            if (i > 0) {  // as matlab for i > 1, this case, i > 0    // penalty has some problem
                // r = r - (1/(lambda*m^2))*v(ptr_start:ptr_end, idx(1:i*blksize))*del_a(1:i*blksize);

                for (int m = ptr_start; m < ptr_end; ++m) {
                    for (int p = 0; p < i*BLKSIZE; ++p) {
                        v_sub[(m - ptr_start)*i*BLKSIZE + p] = k[m * N + idx[p]];  
                    }
                }
                
                for (int p = 0; p < i*BLKSIZE; ++p) {
                    del_a_sub[p] = del_a[p];
                }

                // Fill v_sub and del_a_sub with your data
                cblas_dgemv(CblasRowMajor, CblasNoTrans, BLKSIZE, BLKSIZE * i, 1.0,
                            v_sub, BLKSIZE * i, del_a_sub, 1, 0.0,
                            result, 1);

                for (int p = 0; p < BLKSIZE; ++p) {
                    r[p] = r[p] - const_ca * result[p];
                }
                
                
                // r = r - idx_overlap(ptr_start:ptr_end, 1:i*blksize)*del_a(1:i*blksize)/m;
                for (int j = 0; j < BLKSIZE; ++j) {
                    for (int k = 0; k < i*BLKSIZE; ++k) {
                        if (idx_s[j] == idx[k]) {
                            result_2[j] = result_2[j] + del_a[k]; 
                        }
                    }
                }

                for (int p = 0; p < BLKSIZE; ++p) {
                    r[p] = r[p] - result_2[p] / (N * 1.0);
                    result_2[p] = 0;
                }
            }

            // printf("\nr vector: \n");
            for (int i = 0; i < BLKSIZE; i++) {
                r_copy[i] = r[i];
            }

            int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', BLKSIZE, 1, T, BLKSIZE, r_copy, 1);
            if (info != 0) {
                printf("\nError: info %d\n", info);
                exit(0);
            }

            // the outputted information was in r, so we need to put them to the del_a
            for(int p=0; p<BLKSIZE; p++) {
                del_a[p + ptr_start] = r_copy[p];
                r_copy[p] = 0.0;
            }

            /*
            if (rank == 0) {
                printf("\n");
                printf("del_a: \n");
                for (int i = 0; i < BLKSIZE*s; ++i) {
                    printf("%.16f \n", del_a[i]);
                }
            }
            */

            // 5. Repeat 1-4 until convergence
            iter = iter + 1;

            if (iter >= MAXIT) {
                break;
            }    
        }
        gradient_comp_end = MPI_Wtime();
        gradient_comp = gradient_comp + gradient_comp_end - gradient_comp_start;


        alpha_updating_start = MPI_Wtime();
        
        // add up the alpha based on the del_a
        for (int i = 0; i < BLKSIZE*s; ++i) {
            alpha[idx[i]] = del_a[i] + alpha[idx[i]];
            // del_a[i] = 0.0;
        }    
        alpha_updating_end = MPI_Wtime();
        alpha_updating = alpha_updating + alpha_updating_end - alpha_updating_start;


        memory_reset_start = MPI_Wtime();
        memset(del_a, 0, BLKSIZE*s*sizeof(double));
        memset(k, 0, BLKSIZE*s*N*sizeof(double));
        memset(k_para, 0, BLKSIZE*s*N*sizeof(double));
        
        if (sample_A_sparse->nnz > 0) {
            free(sample_A_sparse->col);
            free(sample_A_sparse->val);
        }
        memory_reset_end = MPI_Wtime();
        memory_reset = memory_reset + memory_reset_end - memory_reset_start;

    }

    if (rank == 0) {
        printf("\n");
        printf("Alpha: \n");
        for (i = 0; i < 10; ++i) {
            printf("%.16f \n", alpha[i]);
        }
    }


    double runtime_end = MPI_Wtime();
    runtime = runtime_end - runtime_begin;

    double major_time = 0.0;

    // make everything one file
    if (rank == 0) {
        
        major_time = (kernel_computation + allreduce_time + sample_time + gradient_comp + alpha_updating + memory_reset) / (MAXIT * 1.0);

        // write some data rows to the CSV file
        // probably ensure that this make sense
        // major time is the time for 6 major piece of runtime component, the part we care most
        fprintf(fp, "%s, %s, %d, %d, %d, %d, %d, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", filename, KERNEL, size, s, BLKSIZE, degree, MAXIT, GAMMA, LAMBDA, csr_setup_time, kernel_computation / (MAXIT * 1.0), allreduce_time / (MAXIT * 1.0), sample_time / (MAXIT * 1.0), csr_read_time, gradient_comp / (MAXIT * 1.0), alpha_updating / (MAXIT * 1.0), memory_reset / (MAXIT * 1.0), major_time,runtime);
        // close the CSV file
        fclose(fp);
    }

    MPI_Finalize();

    
    // free all allocated memories of array
    free(k);
    free(T);
    free(v_alpha);
    free(M_kernel);
    free(del_a);
    free(r);
    free(v_sub);
    free(del_a_sub);
    free(idx);
    free(idx_s);
    free(result);
    free(r_copy);
    free(result_2);
    free(v);
    free(feature_indices);
    free(k_para);
    free(csr->col);
    free(csr->label);
    free(csr->row_b);
    free(csr->row_e);
    free(csr->val);
    free(csr);

    free(sample_A_sparse->row_b);
    free(sample_A_sparse->row_e);
    free(sample_A_sparse);

   
    printf("All freed \n");
    printf("Finally done!\n");

    return 0;
}




// ---------------- Kernels ----------------------------
// linear kernel
// u is sample matrix (BLKSIZE * M)
// v is the full matrix A (N * M)
// m is num features in A
// n is num observations in A
// blksize is num features selected
void linear(Element* sample_A, Element* csr, int n, int m, int blksize, double *k) {

    // Create the handle for the CSR matrix
    sparse_matrix_t sample_handle;
    mkl_sparse_d_create_csr(&sample_handle, SPARSE_INDEX_BASE_ZERO, blksize, m, sample_A->row_b, sample_A->row_e, sample_A->col, sample_A->val);
    // Descriptor of main sparse matrix properties
    // struct matrix_descr descrA;
    // Create matrix descriptor
    struct matrix_descr descr_sampledRows;
    descr_sampledRows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_sampledRows.mode = SPARSE_FILL_MODE_FULL;
    descr_sampledRows.diag = SPARSE_DIAG_NON_UNIT;

    
    sparse_matrix_t csr_handle;
    mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO, n, m, csr->row_b, csr->row_e, csr->col, csr->val);
    // Descriptor of main sparse matrix properties
    struct matrix_descr descr_Rows;
    descr_Rows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_Rows.mode = SPARSE_FILL_MODE_FULL;
    descr_Rows.diag = SPARSE_DIAG_NON_UNIT;

   
    // Compute the dot product between the two matrices
    sparse_status_t error = mkl_sparse_d_sp2md(SPARSE_OPERATION_NON_TRANSPOSE, descr_sampledRows, sample_handle, SPARSE_OPERATION_TRANSPOSE, descr_Rows, csr_handle, 1.0, 0.0, k, SPARSE_LAYOUT_ROW_MAJOR, n);

    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    if (error != SPARSE_STATUS_SUCCESS) {
        printf("error from linear kernel comp: %d | rank = %d\n", error, curr_rank);
    }

    // destroy the handles to save the memory
    mkl_sparse_destroy(sample_handle);
    mkl_sparse_destroy(csr_handle);

}




// ---------------- Kernels ----------------------------
// polynomial kernel
// u is sample matrix (BLKSIZE * M)
// v is the full matrix A (N * M)
// m is num features in A
// n is num observations in A
// blksize is num features selected
// k is the kernel to be returned
// degree is the degree of the polynomial
void polynomial(Element* sample_A, Element* csr, int n, int m, int blksize, double *k, int degree) {

    // Create the handle for the CSR matrix
    sparse_matrix_t sample_handle;
    mkl_sparse_d_create_csr(&sample_handle, SPARSE_INDEX_BASE_ZERO, blksize, m, sample_A->row_b, sample_A->row_e, sample_A->col, sample_A->val);
    // Descriptor of main sparse matrix properties                              
    // struct matrix_descr descrA;
    // Create matrix descriptor
    struct matrix_descr descr_sampledRows;
    descr_sampledRows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_sampledRows.mode = SPARSE_FILL_MODE_FULL;
    descr_sampledRows.diag = SPARSE_DIAG_NON_UNIT;


    sparse_matrix_t csr_handle;
    mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO, n, m, csr->row_b, csr->row_e, csr->col, csr->val);
    // Descriptor of main sparse matrix properties
    struct matrix_descr descr_Rows;
    descr_Rows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_Rows.mode = SPARSE_FILL_MODE_FULL;
    descr_Rows.diag = SPARSE_DIAG_NON_UNIT;

   
    // Compute the dot product between the two matrices
    sparse_status_t error = mkl_sparse_d_sp2md(SPARSE_OPERATION_NON_TRANSPOSE, descr_sampledRows, sample_handle, SPARSE_OPERATION_TRANSPOSE, descr_Rows, csr_handle, 1.0, 0.0, k, SPARSE_LAYOUT_ROW_MAJOR, n);
    // print the error:
    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    if (error != SPARSE_STATUS_SUCCESS) {
        printf("error from poly kernel comp: %d | rank = %d\n", error, curr_rank);
    }

    // destroy the handles to save the memory
    mkl_sparse_destroy(sample_handle);
    mkl_sparse_destroy(csr_handle);

}




// Gaussian Kernel
// gamma: regulator of the 
// u is sample matrix (BLKSIZE * M) --> u columns
// v is the full matrix A (N * M) --> v columns
// m is num features (columns) in A (v)
// n is num observations (rows) in A (v)
// blksize is num features selected
void gaussian(int blksize, int n, int m, Element *sample_A_sparse, Element * csr, double *k, double gamma, int * idx, double * norm) {

        
    // computes the ai ^T aj matrix, which is a [blksize * s, N] matrix

    // Create the handle for the CSR matrix
    sparse_matrix_t sample_handle;
    mkl_sparse_d_create_csr(&sample_handle, SPARSE_INDEX_BASE_ZERO, blksize, m, sample_A_sparse->row_b, sample_A_sparse->row_e, sample_A_sparse->col, sample_A_sparse->val);
    // Descriptor of main sparse matrix properties                              
    // struct matrix_descr descrA;
    // Create matrix descriptor
    struct matrix_descr descr_sampledRows;
    descr_sampledRows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_sampledRows.mode = SPARSE_FILL_MODE_FULL;
    descr_sampledRows.diag = SPARSE_DIAG_NON_UNIT;


    sparse_matrix_t csr_handle;
    mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO, n, m, csr->row_b, csr->row_e, csr->col, csr->val);
    // Descriptor of main sparse matrix properties
    struct matrix_descr descr_Rows;
    descr_Rows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_Rows.mode = SPARSE_FILL_MODE_FULL;
    descr_Rows.diag = SPARSE_DIAG_NON_UNIT;

   
    // Compute the dot product between the two matrices
    sparse_status_t error = mkl_sparse_d_sp2md(SPARSE_OPERATION_NON_TRANSPOSE, descr_sampledRows, sample_handle, SPARSE_OPERATION_TRANSPOSE, descr_Rows, csr_handle, 1.0, 0.0, k, SPARSE_LAYOUT_ROW_MAJOR, n);
    // print the error:
    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    if (error != SPARSE_STATUS_SUCCESS) {
        printf("error from gaussian kernel comp: %d | rank = %d\n", error, curr_rank);
    }

    // destroy the handles to save the memory
    mkl_sparse_destroy(sample_handle);
    mkl_sparse_destroy(csr_handle);

    // printf("\nMKL Kernel Done!\n");

    // compute the gaussian kernel
    for(int i = 0; i < blksize; ++i) {
        for (int j = 0; j < n; ++j) {
            k[j + i*n] = norm[idx[i]] + norm[j] - 2 * k[j + i*n];
        }
    }

}






// ------------- Useful Functions ----------------------


// randomly generate an array of index with no repetation
// size: size of array
// range: range of index to be sampled from
int* rand_index(int size, int range) {

    int *b = (int *) malloc (sizeof(int)*size);

    // rand() % (max_number - minimum_number) + minimum_number
    // srand ( time ( NULL));   // this works if every iter takes a lot of time
    int max_number = range;
    int minimum_number= 0;
    int haha;

    int i, j, curr;
    int aa; // boolean value to determine if repetition happens
    for (i = 0; i < size; i++) {
        haha = (int) rand() % (max_number - minimum_number) + minimum_number;
        curr = i;
        j = 0;
        aa = 0;

        // check if the index sampled already exist previously
        while (j < curr){
            if ( *(b + j) == haha) {  
                i--;
                aa = 1;  // set the boolean value to 1
                break;
            }
            j++;
        }
        // if there is repeating number exist in the array, quickly continue
        if (aa == 1) {
            continue;
        }
        *(b + i) = haha;

    }

    return b;
}


// range: the range of number generated
// iter: size of the array
double * random_array(int range, int iter) {
    double random_value;  // declare a varaible to 

    double *b = (double * ) malloc(sizeof(double)*iter);

    srand ( time ( NULL));
    int i;

    for (i = 0; i < iter; ++i) {
        // the range will give you the random number with the 
        random_value = (double) rand() / RAND_MAX*range;
        *(b + i) = random_value;
    }
    
    return b;
}

// n: num rows
// m: num columns
void printMatrix(double *x, int n, int m) {
    int curr_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    printf("Test matrix is | rank = %d \n", curr_rank);
    int i,j;
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++) {
            printf("%.4f ", x[i*m +j]);   // matrix is row-based
        }
        printf("\n");
    }
}

// create an identity matrix with size M * M
// matrix: the setup identity matrix pointer
// M: size of the identity matrix
void createIdentityMatrix(int M, double * matrix) {

    // printf("Given M: %d\n", M);
    // printf("See what's going on\n");


    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (i == j) {
                matrix[i*M + j] = 1;
            } else {
                matrix[i*M + j] = 0;
            }
        }
    }
}

// this will find the sub_I', therefore, later we need to use matrix %*% matrix'
void subset_identity(Element* sub_I, int * idx, int blksize) {

    // Initialize the struct members
    for (int i = 0; i < blksize; i++) {
        sub_I->row_b[i] = i;
        sub_I->row_e[i] = i+1;
    }

    for (int j = 0; j < blksize; j++) {
        sub_I->col[j] = idx[j];
        sub_I->val[j] = 1.0;
    }

    sub_I->nnz = blksize;
}


// find the index overlap matrix
void indexOverlap(Element* sample_A, int m, int blksize, double *index_overlap) {


    // Create the handle for the CSR matrix
    sparse_matrix_t sample_handle;
    mkl_sparse_d_create_csr(&sample_handle, SPARSE_INDEX_BASE_ZERO, blksize, m, sample_A->row_b, sample_A->row_e, sample_A->col, sample_A->val);
    // Descriptor of main sparse matrix properties
    // struct matrix_descr descrA;
    // Create matrix descriptor
    struct matrix_descr descr_sampledRows;
    descr_sampledRows.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr_sampledRows.mode = SPARSE_FILL_MODE_FULL;
    descr_sampledRows.diag = SPARSE_DIAG_NON_UNIT;

    // mkl_sparse_
    // double * result = (double *) calloc(blksize*blksize,sizeof(double)); 
    // mkl_sparse_d_syrkd (SPARSE_OPERATION_NON_TRANSPOSE, sample_handle, 1.0, 0.0, index_overlap, SPARSE_LAYOUT_ROW_MAJOR, blksize);
    // find sub_I' %*% sub_I, result in a [blksize * s , blksize * s] matrix
    sparse_status_t error = mkl_sparse_d_sp2md(SPARSE_OPERATION_NON_TRANSPOSE, descr_sampledRows, sample_handle, SPARSE_OPERATION_TRANSPOSE, descr_sampledRows, sample_handle, 1.0, 0.0, index_overlap, SPARSE_LAYOUT_ROW_MAJOR, blksize);
    // print the error:

    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    if (error != SPARSE_STATUS_SUCCESS) {
        printf("error from linear kernel comp: %d | rank = %d\n", error, curr_rank);
    }

    // destroy the handles to save the memory
    mkl_sparse_destroy(sample_handle);

}






// ====================== Sparsity Data Reading =========================

// setup the csr, with the labels, memory begin allocated
void csr_setup_bdcd(Element * csr, int MAX_LINE_LENGTH, char * filename, int N, int * feature_indices, int sub_blk) {

    char buffer_first[MAX_LINE_LENGTH];

    FILE *hp;
    hp = fopen(filename, "r");
    if (hp == NULL) {
        printf("Failed to open file: setup stage\n");
        exit(0);
    }


    int index = 0;

    while (fgets(buffer_first, MAX_LINE_LENGTH, hp)) {
        // Split the line into tokens
        char *token = strtok(buffer_first, " ");
        double label = atof(token);
        csr->label[index] = label;
        
        // Parse the remaining tokens
        while ((token = strtok(NULL, " "))) {

            char *colon = strchr(token, ':');  // search where the ":" is in the first line
            if (colon == NULL) {  // if ":" not found, skip token
                break;
            }
            int feature = atoi(token);   // ascii to integer
            
            for (int i = 0; i < sub_blk; ++i) {
                if ((feature - 1) == feature_indices[i]) {
                    // printf("%d ", feature - 1);
                    csr->nnz++;
                }
            }
        }

        index++;
    }

    fclose(hp);

        
    // Allocate memory for the sparse matrix after we get the nnz
    csr->row_b = (int*) calloc(N + 1, sizeof(int));   // as many as number of rows
    csr->row_e = (int*) calloc(N + 1, sizeof(int));   // as many as number of rows
    csr->col = (int*) calloc(csr->nnz, sizeof(int));   // as many as nnz (as each location of nnz will be recorded here)
    csr->val = (double*) malloc(csr->nnz * sizeof(double));   

    printf("csr setup nnz: %d \n", csr->nnz);

}



// read the data in sparse format
void read_data_sparse_bdcd(Element * csr, int MAX_LINE_LENGTH, char * filename, int * feature_indices, int sub_blk) {

    // define the parameters to be used later
    // Note: row is a special case, as it updates based on the col. If there is no more col indices to be recorded in this feature,
    // we need to record at which index of col array, the value shift to the next observation
    int elt_in_row = 0;  // count how many nnz is in this observation (define the leap of the row value)
    int row_value = 0;  // row_value record the value of the current row element
    int row_value_index = 0; // where the row value should locates in row array

    int curr_col = 0; // current col record the location of current nnz value's location in the col array
    int curr_val = 0; // current value record the location of current nnz value to be filled in the val array
    char buffer[MAX_LINE_LENGTH];


    // read the data
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("Failed to open file: data reading stage\n");
        exit(1);
    }

    while (fgets(buffer, MAX_LINE_LENGTH, fp)) {
        char *token = strtok(buffer, " ");
        
        // Parse the remaining tokens
        while ((token = strtok(NULL, " "))) {
            
            char *colon = strchr(token, ':');  // search where the ":" is in the first line
            if (colon == NULL) {  // if ":" not found, skip token
                break;
            }
            int feature = atoi(token);   // ascii to integer
            
            double value = 0;

            // check if negative values
            if (*(colon + 1) == '-') {  // check if the value is negative
                value = -1 * atof(colon + 2);  // convert the absolute value to float and negate it
            } else {
                value = atof(colon + 1);   // convert to float as usual
            }

            
            for (int i = 0; i < sub_blk; ++i) {
                if ((feature - 1) == feature_indices[i]) {
                    // Add the element to the sparse matrix
                    csr->row_b[row_value_index] = row_value;
                    csr->col[curr_col] = i;
                    csr->val[curr_val] = value;

                    elt_in_row++;  // this will help us see 
                    curr_col++;
                    curr_val++;

                }
            }

        }

        if (elt_in_row == 0) {
            csr->row_b[row_value_index] = row_value;
        }

        // update the row_value_index and row_value (as we are going to the next observation)
        row_value = row_value + elt_in_row; // leap across number of nnz in previous observation

        csr->row_e[row_value_index] = row_value;  // row_e should hav row_value updated, so one place different
        row_value_index++; // update the row_value_index that store the value of row_value

        // reset elements in row to 0
        elt_in_row = 0;
    }


    fclose(fp);

    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

}



// get the sample_A in sparse format
void sparse_sample_A_bdcd(Element * sample_A_sparse, Element * csr, int blksize, int M, int* idx) {

    
    // we need to get the number of nnz that will be included in the sample_A
    // we can get the number of nnz by sum up the difference between csr->row_e[idx[i]] -  csr->row_b[idx[i]]
    int sample_nnz = 0;
    for (int i = 0; i < blksize; ++i) {
        sample_nnz = sample_nnz + csr->row_e[idx[i]] -  csr->row_b[idx[i]];
    }

    sample_A_sparse->col = (int*) malloc(sample_nnz * sizeof(int));   // as many as nnz (as each location of nnz will be recorded here)
    sample_A_sparse->val = (double*) malloc(sample_nnz * sizeof(double));  


    int begin = 0;
    int end = 0;
    int row_begin = 0;
    int k = 0;

    // get the values of sample_A
    for (int i = 0; i < blksize; ++i) {
        sample_A_sparse->row_b[i] = row_begin;

        begin = csr->row_b[idx[i]];
        end = csr->row_e[idx[i]];

        sample_A_sparse->row_e[i] = row_begin + (end - begin);

        row_begin = row_begin + (end - begin);

       
        // read the column index information
        for (int j = begin; j < end; ++j) {
            sample_A_sparse->col[k] = csr->col[j];
            sample_A_sparse->val[k] = csr->val[j];
            k++;
        }
    }

    sample_A_sparse->nnz = sample_nnz;

    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

}


// ====================== Dense Data Reading =========================


// objective value computation
// objective value = 1 / (lambda * m^2) alpha.T K alpha + 1 / 2m ||alpha - y||^2, in this case, K matrix is M
// ||alpha - y||^2, alpha is n * 1, so this one equals one as it is diff.T * diff
// alpha.T K alpha is 
double objective_value_cabdcd(double * M, double * alpha, double *b, int m, double lambda, int blksize, int * idx) {

    // part 1: 1 / (lambda * m^2) alpha.T K alpha
    double const_ca = 1 / pow((lambda * m),2);

    // alpha.T K alpha
    double p1 = 0.0;
    double first_part = 0.0;
    for (int i = 0; i < blksize; ++i) {
        for (int j = 0; j < blksize; ++j) {
            first_part = first_part + alpha[idx[i]] * M[blksize*i + j];
        }
        p1 = p1 + first_part * alpha[idx[i]];
        first_part = 0;
    }

    p1 = p1 * const_ca;


    // part 2: 1 / 2m ||alpha - y||^2
    double const_2 = 1 / (2*m);
    double p2 = 0.0;
    for (int i = 0; i < blksize; ++i) {
        p2 = p2 + pow((alpha[idx[i]] - b[idx[i]]),2);
    }

    // find the objective value
    double obj_val = 0.0;
    obj_val = p1 + p2 * const_2;

    return obj_val;
}
