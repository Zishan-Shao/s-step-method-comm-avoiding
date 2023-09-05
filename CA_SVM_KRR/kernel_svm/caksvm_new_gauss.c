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



typedef struct {
    int *row_b;  // record where the row, start 0
    int *row_e;  // where the row end, end num_row - 1
    int *col;
    double *val;
    double *label;
    int nnz;  // record number of nnz
} Element;

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


int main(int argc, char *argv[]) {
    
    if(argc != 3) {
        printf("Error: incorrect inputs\n");
        return 1;
    }
    
    /*
    char filename[50] = "leu";// "breast-cancer_scale.txt";

    int N = 38;   // n is num observation, m is num features
    int M = 7129;    // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[2]);// 5000;
    double GAMMA = 1;
    int seed = 42;
    double LAMBDA = 1;  // 0.011314236798389;
    int degree = 5;
    double OMEGA = 0;
    int s = atoi(argv[1]); //64;
    int BLKSIZE = 1;
    */

    /*
    char filename[50] = "webspam_wc_normalized";

    int N = 350000;   // n is num observation, m is num features
    int M = 16609143;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[2]);// 5000;
    double GAMMA = 1;
    int seed = 42;
    double LAMBDA = 1;  // 0.011314236798389;
    int degree = 5;
    double OMEGA = 0;
    int s = atoi(argv[1]); //64;
    int BLKSIZE = 1;
    */

    /*
    char filename[50] = "leu";// "breast-cancer_scale.txt";

    int N = 38;   // n is num observation, m is num features
    int M = 7129;    // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[2]);// 5000;
    double GAMMA = 1;
    int seed = 42;
    double LAMBDA = 1;  // 0.011314236798389;
    int degree = 5;
    double OMEGA = 0;
    int s = atoi(argv[1]); //64;
    int BLKSIZE = 1;
    */
    
   /*
    char filename[50] = "colon-cancer";// "breast-cancer_scale.txt";

    int N = 62;   // n is num observation, m is num features
    int M = 2000;    // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[2]);// 5000;
    double GAMMA = 1;
    int seed = 42;
    double LAMBDA = 1;  // 0.011314236798389;
    int degree = 5;
    double OMEGA = 0;
    int s = atoi(argv[1]); //64;
    int BLKSIZE = 1;
    */

    
    char filename[50] = "news20.binary";

    int N = 19996;   // n is num observation, m is num features
    int M = 1355191;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[2]);
    double GAMMA = 1;
    int seed = 42;
    double LAMBDA = 1;  
    int degree = 5;
    double OMEGA = 0;
    int s = atoi(argv[1]); //64;
    int BLKSIZE = 1;
    

    /*
    char filename[50] = "duke";

    int N = 44;   // n is num observation, m is num features
    int M = 7129;   // n,m are dimension of the A, dimension of b is n * 1

    // define hyper parameters: BLKSIZE, LAMBDA, MAXIT, GAMMA, KERNEL, tol, seed, s (if cabdccd)
    char KERNEL[10] = "gauss";
    int MAXIT = atoi(argv[2]);// 5000;
    double GAMMA = 1;
    int seed = 42;
    double LAMBDA = 1;  // 0.011314236798389;
    int degree = 5;
    double OMEGA = 0;
    int s = atoi(argv[1]); //64;
    int BLKSIZE = 1;
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


    double runtime = 0.0;
    double memory_reset = 0.0;
    double memory_reset_start = 0.0;
    double memory_reset_end = 0.0;
    
    double runtime_begin = MPI_Wtime();
    
    
    int i, iter;
    double proj_grad = 0.0;
    double talpup = 0.0;
    double * theta = (double *) calloc(s, sizeof(double));
    double eta = 0.0;
    double grad = 0.0;

    int max_line = 5000000;    // note: if one line of feature has more than 2000 characters, need to enlarge it

    memory_reset_start = MPI_Wtime();
    // initialize the struct
    Element* csr = (Element *) malloc(sizeof(Element));

    // allocate memory for the label
    csr->label = (double*) malloc(N * sizeof(double));
    csr->nnz = 0;


    int * idx = (int *) malloc(sizeof(int)*s);  // the index array should be BLKSIZE * s size
   
    // initialize the alpha and del_a
    double * alpha = (double*) calloc(N, sizeof(double));

    // M_kernel = (double *) calloc(BLKSIZE*BLKSIZE*s*s, sizeof(double)); // [s*blksize, s*blksize]
    double * k_row = (double *) calloc(N*s, sizeof(double));
    
    // struct for product
    Element* scale_A = (Element *) malloc(sizeof(Element));

    // allocate memory for the label
    scale_A->label = (double*) malloc(N * sizeof(double));
    scale_A->nnz = 0;

    // double * grad = (double *) calloc(s*1, sizeof(double));

    // initialize the struct
    Element* scale_sample = (Element *) malloc(sizeof(Element));

    // Allocate memory for the sparse sample_A
    // sample_A is one observation
    scale_sample->row_b = (int*) calloc(1*s + 1, sizeof(int));   // as many as number of rows
    scale_sample->row_e = (int*) calloc(1*s + 1, sizeof(int));   // as many as number of rows
    scale_sample->nnz = 0;


    iter = 0;
  

    // parallel part declaration
    double * k_para = (double *) calloc(N*s, sizeof(double));

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
        printf("Basic Matrices Setup Finished! |    %s  ;  s = %d \n", filename, s);
    }
    memory_reset_end = MPI_Wtime();
    memory_reset = memory_reset + memory_reset_end - memory_reset_start;



    // ############### csr setup time ##################
    double csr_start_time = MPI_Wtime();
    

    // get the labels and nnz, allocate memory for csr
    csr_setup_bdcd(csr, max_line, filename, N, feature_indices, sub_blk); 

    if (csr->nnz > 0) {
        scale_A->nnz = csr->nnz;
        scale_A->row_b = (int*) calloc(N*s + 1, sizeof(int));   // as many as number of rows
        scale_A->row_e = (int*) calloc(N*s + 1, sizeof(int));   // as many as number of rows
        scale_A->col = (int*) calloc(csr->nnz, sizeof(int));   // as many as nnz (as each location of nnz will be recorded here)
        scale_A->val = (double*) malloc(csr->nnz * sizeof(double));   
    }

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
    double kernel_loop_time = 0.0;
    
    double kernel_in_gauss = 0.0;
    double kernel_sp2md = 0.0;



    // ############### csr reading time ##################
    read_start_time = MPI_Wtime();

    if (rank == 0) {
        printf("start reading now\n");
    }

    if (csr->nnz > 0) {
        read_data_sparse_bdcd(csr, scale_A, max_line, filename, feature_indices, sub_blk);
    }

    read_end_time = MPI_Wtime();
    csr_read_time = read_end_time - read_start_time;

    if (rank == 0) {
        printf("CSR Reading Finished!! \n");
    }
    

    // do the norm here
    double * norm = (double *) calloc(N, sizeof(double)); // norm vector should be n sized, as diagonal vector of a [N,N] matrix
    // sparse_matrix_t csr_handle;
    
    //struct matrix_descr descr_Rows;
    //descr_Rows.type = SPARSE_MATRIX_TYPE_GENERAL;
    //descr_Rows.mode = SPARSE_FILL_MODE_FULL;
    //descr_Rows.diag = SPARSE_DIAG_NON_UNIT;

    // Initialize vectors of 1s, there should be two, one for norm[idx[j]], one for norm[j]
    double * ones_blksize = (double *) calloc(s*BLKSIZE, sizeof(double));
    double * ones_N = (double *) calloc(N, sizeof(double));

    // setup the ones vector
    memset(ones_blksize, 1.0, s*BLKSIZE*sizeof(double));
    memset(ones_N, 1.0, N*sizeof(double));


    // define the selected norm entries to later we can use it for 
    // cblas_dger computation, it is [s,1] and we times it with a 
    // 1 vector with size [1,N] to create a [s,N] matrix which each
    // row are the current idx[i] entries of norm
    double * idx_norm = (double *) calloc(s*BLKSIZE, sizeof(double));
    
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

        // compute the csr handle
        // mkl_sparse_d_create_csr(&csr_handle, SPARSE_INDEX_BASE_ZERO, N, M, scale_A->row_b, scale_A->row_e, scale_A->col, scale_A->val);
        // Descriptor of main sparse matrix properties
    }
   

    while (iter < MAXIT) {

        // ############### sample_A setup time ##################
        sample_start_time = MPI_Wtime();

        for (int i = 0; i < s; ++i) { 
            idx[i] = (int) rand() % (N - 0) + 0;
        }
        

        if (scale_A->nnz > 0) {
            sparse_sample_A_bdcd(scale_sample, scale_A, 1*s, sub_blk, idx);

            // updates the idx_norm, selecting elements from the norm vector
            for(int i = 0; i < s*BLKSIZE; ++i) {
                idx_norm[i] = norm[idx[i]];
            }

        }

        sample_end_time = MPI_Wtime();
        sample_time = sample_time + sample_end_time - sample_start_time;
        
        

        // ############### kernel setup time ##################
        kernel_start_time = MPI_Wtime();

        if (scale_sample->nnz > 0 && scale_A->nnz > 0) {
            if (strcmp(KERNEL, "linear") == 0) {
                linear(scale_sample, scale_A, N, sub_blk, 1*s, k_para);   // as return a pointer
            }
            else if (strcmp(KERNEL, "poly") == 0) {
                linear(scale_sample, scale_A, N, sub_blk, 1*s, k_para); 
            }
            else if (strcmp(KERNEL, "gauss") == 0) {
                gaussian(1*s, N, sub_blk, scale_sample, scale_A, k_para, idx, norm, &kernel_in_gauss, &kernel_sp2md, ones_N, ones_blksize, idx_norm); 
            }
        }

        // ############### kernel setup time ##################
        kernel_end_time = MPI_Wtime();
        kernel_computation = kernel_computation + kernel_end_time - kernel_start_time;

        // MPI_Barrier(MPI_COMM_WORLD);

        // ############### AllReduce() time ##################
        allreduce_start_time = MPI_Wtime();


        // Each MPI process sends its rank to reduction, root MPI process collects the result
        MPI_Allreduce(k_para, k_row, s*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        // ############### AllReduce() end time ##################
        allreduce_end_time = MPI_Wtime();
        allreduce_time = allreduce_time + allreduce_end_time - allreduce_start_time;


        full_kernel_start_time = MPI_Wtime();
        if (strcmp(KERNEL, "poly") == 0) {
            // make element wise multiplication
            for (int i = 0; i < N*s; ++i) {
                k_row[i] = pow(k_row[i], degree);
            }
            cblas_dscal(s*N, GAMMA, k_row, 1);
        }
        else if (strcmp(KERNEL, "gauss") == 0) {
            // Apply the exponential function element-wise to the k array
            // Scale the array using cblas_dscal
            cblas_dscal(s*N, GAMMA*(-1.0), k_row, 1);
            for (int i = 0; i < N*s; i++) {
                // do the -GAMMA * k_row[i] with cblas_dscal()
                k_row[i] = exp(k_row[i]);
            }
        }
        full_kernel_end_time = MPI_Wtime();
        // kernel_computation = kernel_computation + full_kernel_end_time - full_kernel_start_time;
        kernel_loop_time = full_kernel_end_time - full_kernel_start_time;
        kernel_computation = kernel_computation + kernel_loop_time;
        
        gradient_comp_start = MPI_Wtime();

        for (int i = 0; i < s; ++i) {
            grad = 0.0;
            talpup = 0.0;
            
            //grad = k_row(i,:)*(alpha) - 1 + opt.omega*alpha(idx(i));
            for (int p = 0; p < N; ++p) {
                grad = k_row[p + i * N] * alpha[p] + grad;
            }
            grad = grad - 1 + OMEGA*alpha[idx[i]];
            
            // for all previous iterations, check if overlapped indices
            // if yes, make penalties on grad and talpup
            if (i > 0) {
               for (int j = 0; j < i; ++j) {
                    // grad = grad + theta(j)*k_row(i,idx(j)) + opt.omega*(idx(j)==idx(i))*theta(j);
                    grad = grad + theta[j] * k_row[i * N + idx[j]];
                    // printf("\ncurr_grad: %.5f",grad);
                    if (idx[j]==idx[i]) {
                        grad = grad + OMEGA * theta[j];
                        // talpup = talpup + (idx(j) == idx(i))*theta(j);
                        talpup = talpup + theta[j];
                    }
               } 
            }
            // eta = k_row(i,idx(i)) + opt.omega;
            eta = k_row[i * N + idx[i]] + OMEGA;
        
            // proj_grad = abs(min(max(alpha(idx(i)) + talpup - grad, 0), opt.nu) - alpha(idx(i)) - talpup);
            double diff = alpha[idx[i]] - grad + talpup;
            proj_grad = MIN(MAX(diff,0),LAMBDA) - alpha[idx[i]] - talpup; // abs method does not work as expected, so add a simple if condition
            
            if (proj_grad < 0) {
                proj_grad = 0 - proj_grad;
            } 
            
            // if(proj_grad > 1e-14)
            //     theta(i) = min(max(alpha(idx(i)) + talpup - grad/eta, 0), opt.nu) - alpha(idx(i)) - talpup;
            // else
            //     theta(i) = 0;
            // end
            
            diff = alpha[idx[i]] + talpup - grad / eta;
            if (proj_grad > 1e-14) {
                theta[i] = MIN(MAX(diff,0),LAMBDA) - alpha[idx[i]] - talpup;
            } else {
                theta[i] = 0.0;
            }


            // 5. Repeat 1-4 until convergence
            iter = iter + 1;

            if (iter >= MAXIT) {
                break;
            }    


            // double result = 0.0; 
            // for (int i = 0; i < N; ++i) {
            //     result = result + alpha[i] * csr->label[i];
            // }

            //if (rank == 0) {
            //    printf("\nresult: %.12f\n", result);
            //}
            /*
            printf("theta:\n");
            for (int p = 0; p < s; ++p) {
                printf("%.6f ", theta[p]);
            }
            printf("\n");

            // printf("\nresult: %.4f\n", result);
            */
        }

        
        gradient_comp_end = MPI_Wtime();
        gradient_comp = gradient_comp + gradient_comp_end - gradient_comp_start;


        alpha_updating_start = MPI_Wtime();
        
        // add up the alpha based on the theta
        for (int i = 0; i < s; ++i) {
            alpha[idx[i]] = theta[i] + alpha[idx[i]];
        }    
        alpha_updating_end = MPI_Wtime();
        alpha_updating = alpha_updating + alpha_updating_end - alpha_updating_start;


        memory_reset_start = MPI_Wtime();
        memset(theta, 0, s*sizeof(double));
        // memset(grad, 0, s*sizeof(double));
        memset(k_row, 0, s*N*sizeof(double));
        memset(k_para, 0, s*N*sizeof(double));
        memset(idx_norm, 0, s*BLKSIZE*sizeof(double));
        
        
        if (scale_sample->nnz > 0) {
            free(scale_sample->col);
            free(scale_sample->val);
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

        fp = fopen("caksvm_data_newsbinary_gauss.csv", "a");
        if (fp == NULL) {
            printf("Failed to open file for writing\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    
        
        major_time = (kernel_computation + allreduce_time + sample_time + gradient_comp + alpha_updating + memory_reset) / (MAXIT * 1.0);

        // write some data rows to the CSV file
        // probably ensure that this make sense
        // major time is the time for 6 major piece of runtime component, the part we care most
        fprintf(fp, "%s, %s, %d, %d, %d, %d, %d, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", filename, KERNEL, size, s, BLKSIZE, degree, MAXIT, GAMMA, LAMBDA, csr_setup_time, kernel_computation / (MAXIT * 1.0), allreduce_time / (MAXIT * 1.0), sample_time / (MAXIT * 1.0), csr_read_time, gradient_comp / (MAXIT * 1.0), alpha_updating / (MAXIT * 1.0), memory_reset / (MAXIT * 1.0), major_time,runtime);
        // close the CSV file
        fclose(fp);

        // print the result
        printf("%s, %s, %d, %d, %d, %d, %d, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n", filename, KERNEL, size, s, BLKSIZE, degree, MAXIT, GAMMA, LAMBDA, csr_setup_time, kernel_computation / (MAXIT * 1.0), allreduce_time / (MAXIT * 1.0), sample_time / (MAXIT * 1.0), csr_read_time, gradient_comp / (MAXIT * 1.0), alpha_updating / (MAXIT * 1.0), memory_reset / (MAXIT * 1.0), major_time,runtime);
        printf("%.10f, %.10f, %.10f, %.10f\n", kernel_computation / (MAXIT * 1.0), kernel_loop_time / (MAXIT * 1.0), kernel_in_gauss / (MAXIT * 1.0), kernel_sp2md / (MAXIT * 1.0));
    }

    
    // free all allocated memories of array
    free(alpha);
    // free(grad);
    free(feature_indices);
    free(k_para);
    free(k_row);
    free(idx);
    free(theta);
    free(norm);
    free(ones_blksize);
    free(ones_N);
    free(idx_norm);
   
    // free the csr matrices
    free(csr->col);
    free(csr->label);
    free(csr->row_b);
    free(csr->row_e);
    free(csr->val);
    free(csr);


    free(scale_A->label);
    if (csr->nnz > 0) {
        free(scale_A->col);
        free(scale_A->row_b);
        free(scale_A->row_e);
        free(scale_A->val);
    }
    free(scale_A);
    

    free(scale_sample->row_b);
    free(scale_sample->row_e);
    free(scale_sample);

    //if (csr->nnz > 0) {
    //    mkl_sparse_destroy(csr_handle);
    //}
   
    printf("All freed \n");
    printf("Finally done!\n");



    MPI_Finalize();

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
// void gaussian(int blksize, int n, int m, Element *sample_A_sparse, Element * csr, double *k, double gamma, int * idx, double * norm, double * gauss_in_loop, double * kernel_sp2md) {
void gaussian(int blksize, int n, int m, Element *sample_A_sparse, Element * csr, double *k, double gamma, int * idx, double * norm, double * gauss_in_loop, double * kernel_sp2md, double * ones_N, double * ones_blksize, double * idx_norm) {

    
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


    double dot_product = 0.0;
    dot_product = MPI_Wtime();
    // Compute the dot product between the two matrices
    sparse_status_t error = mkl_sparse_d_sp2md(SPARSE_OPERATION_NON_TRANSPOSE, descr_sampledRows, sample_handle, SPARSE_OPERATION_TRANSPOSE, descr_Rows, csr_handle, -2.0, 0.0, k, SPARSE_LAYOUT_ROW_MAJOR, n);
    // print the error:
    int curr_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

    if (error != SPARSE_STATUS_SUCCESS) {
        printf("error from gaussian kernel comp: %d | rank = %d\n", error, curr_rank);
    }
    kernel_sp2md[0] = kernel_sp2md[0] + MPI_Wtime() - dot_product;

    // destroy the handles to save the memory
    mkl_sparse_destroy(sample_handle);
    mkl_sparse_destroy(csr_handle);

    // printf("\nMKL Kernel Done!\n");

    // compute the gaussian kernel
    // do not use this double loop here
    double gauss_loop = 0.0;
    gauss_loop = MPI_Wtime();
    /*
    for(int i = 0; i < blksize; ++i) {
        for (int j = 0; j < n; ++j) {
            k[j + i*n] = norm[idx[i]] + norm[j] - 2 * k[j + i*n];
        }
    }
    */
    
    
    // Perform rank-1 update to add norm[j] to k
    cblas_dger(CblasRowMajor, blksize, n, 1.0, ones_blksize, 1, norm, 1, k, n);
    // Perform rank-1 update to add norm[idx[i]] to k
    cblas_dger(CblasRowMajor, blksize, n, 1.0, idx_norm, 1, ones_N, 1, k, n);

    gauss_in_loop[0] = gauss_in_loop[0] + MPI_Wtime() - gauss_loop;
    // printf("gauss in loop: %.12f\n", gauss_in_loop);

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

    //printf("csr setup nnz: %d \n", csr->nnz);

}



// read the data in sparse format
void read_data_sparse_bdcd(Element * csr, Element * scale_A, int MAX_LINE_LENGTH, char * filename, int * feature_indices, int sub_blk) {

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
        double label = atof(token);
        
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
                    // also setup the scale_A dataset
                    scale_A->row_b[row_value_index] = row_value;
                    scale_A->col[curr_col] = i;
                    scale_A->val[curr_val] = value * label;

                    elt_in_row++;  // this will help us see 
                    curr_col++;
                    curr_val++;

                }
            }

        }

        if (elt_in_row == 0) {
            csr->row_b[row_value_index] = row_value;
            scale_A->row_b[row_value_index] = row_value;
        }

        // update the row_value_index and row_value (as we are going to the next observation)
        row_value = row_value + elt_in_row; // leap across number of nnz in previous observation

        csr->row_e[row_value_index] = row_value;  // row_e should hav row_value updated, so one place different
        scale_A->row_e[row_value_index] = row_value;  // row_e should hav row_value updated, so one place different
        
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
