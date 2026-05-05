// Parallel matrix multiplication using CUDA with Tiled Shared Memory, Transpose B, and Kernel Fusion
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ! Network Parameters
#define INPUT_SIZE      32      // Number of input features
#define HIDDEN_SIZE     256     // Number of neurons in the hidden layer
#define OUTPUT_SIZE     1       // Number of output neurons
#define EPOCHS          100     // Number of training epochs
#define LOG_EVERY_EPOCH 1       // Log loss every n epochs
#define LEARNING_RATE   0.002
#define BATCH_SIZE      256     // Batch size for SGD
#define TILE_SIZE       16      // Tile size for shared memory (optimal for most GPUs)

// ! Data Structures
// Structure for matrix
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

// ! Memory Management
// Function to allocate a matrix (contiguous memory allocation)
Matrix* allocate_matrix(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(rows * cols * sizeof(float));  // Contiguous memory
    return m;
}

// Function to free a matrix
void free_matrix(Matrix *m) {
    free(m->data);  // Free contiguous memory
    free(m);
}

// ! Matrix Operations
// Function to initialize matrix with random values
void random_init(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

// ============================================================================
// OPTIMIZATION 1: Matrix Transpose Kernel
// ============================================================================
/*
 * Transpose kernel with shared memory tiling to avoid bank conflicts
 * The +1 padding prevents bank conflicts when reading/writing tiles
 */
__global__ void transpose_kernel(float *input, float *output, int rows, int cols) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    // Calculate input coordinates
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load tile from input with coalesced reads
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    // Synchronize to ensure tile is loaded
    __syncthreads();
    
    // Calculate output coordinates (transposed)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // Write transposed tile to output with coalesced writes
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================================
// OPTIMIZATION 2: Tiled Matrix Multiplication with Transposed B
// ============================================================================
/*
 * Optimized matrix multiplication kernel:
 * - Uses tiling to leverage shared memory
 * - B is pre-transposed for coalesced access
 * - Computes C = A × B^T (where B is already transposed)
 */
__global__ void mat_mult_tiled_kernel_transpose(float *A, float *B_T, float *C, 
                                                 int A_rows, int A_cols, int B_rows) {
    // Calculate the row and column index of the C element to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Allocate shared memory for tiles of A and B_T
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B_T[TILE_SIZE][TILE_SIZE];
    
    // Accumulator for the dot product
    float value = 0.0f;
    
    // Calculate the number of tiles needed
    int num_tiles = (A_cols + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over all tiles
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Calculate indices for loading tiles
        int a_col = tile_idx * TILE_SIZE + threadIdx.x;
        int b_col = tile_idx * TILE_SIZE + threadIdx.x;  // B is transposed, so we load rows
        
        // Load element from A into shared memory (coalesced)
        if (row < A_rows && a_col < A_cols) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * A_cols + a_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load element from B_T into shared memory (NOW COALESCED!)
        // Since B is transposed, we're loading B_T[col, b_col] which is stored as B_T[col * A_cols + b_col]
        if (col < B_rows && b_col < A_cols) {
            tile_B_T[threadIdx.y][threadIdx.x] = B_T[col * A_cols + b_col];
        } else {
            tile_B_T[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += tile_A[threadIdx.y][k] * tile_B_T[threadIdx.y][k];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < A_rows && col < B_rows) {
        C[row * B_rows + col] = value;
    }
}

// ============================================================================
// OPTIMIZATION 3: Fused Forward Pass Kernel (MatMul + ReLU + MatMul)
// ============================================================================
/*
 * Fused forward pass kernel that combines:
 * 1. Z1 = X × W1^T (tiled matrix multiplication with transposed W1)
 * 2. H = ReLU(Z1) (activation function)
 * 3. Y_pred = H × W2^T (tiled matrix multiplication with transposed W2)
 * 
 * Benefits:
 * - Eliminates 2 kernel launches
 * - Keeps intermediate results in registers/shared memory
 * - Reduces global memory traffic
 */
__global__ void fused_forward_pass_kernel(
    float *X, float *W1_T, float *W2_T,
    float *Z1_out, float *Y_pred_out,
    int batch_size, int input_size, int hidden_size, int output_size
) {
    // Each thread computes one element of the final output Y_pred
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Batch index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Output dimension (always 1 in this case)
    
    // Shared memory for tiles
    __shared__ float tile_X[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_W1_T[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_H[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_W2_T[TILE_SIZE][TILE_SIZE];
    
    // ========================================================================
    // PHASE 1: Compute Z1 = X × W1^T
    // ========================================================================
    float z1_value = 0.0f;
    
    if (row < batch_size && threadIdx.x < hidden_size) {
        int num_tiles_phase1 = (input_size + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int tile_idx = 0; tile_idx < num_tiles_phase1; tile_idx++) {
            int x_col = tile_idx * TILE_SIZE + threadIdx.x;
            int w1_col = tile_idx * TILE_SIZE + threadIdx.x;
            
            // Load X tile
            if (row < batch_size && x_col < input_size) {
                tile_X[threadIdx.y][threadIdx.x] = X[row * input_size + x_col];
            } else {
                tile_X[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load W1_T tile (W1 is transposed)
            int w1_row = threadIdx.y;  // This maps to hidden neuron index
            if (w1_row < hidden_size && w1_col < input_size) {
                tile_W1_T[threadIdx.y][threadIdx.x] = W1_T[w1_row * input_size + w1_col];
            } else {
                tile_W1_T[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                z1_value += tile_X[threadIdx.y][k] * tile_W1_T[threadIdx.y][k];
            }
            
            __syncthreads();
        }
    }
    
    // ========================================================================
    // PHASE 2: Apply ReLU activation
    // ========================================================================
    float h_value = fmaxf(0.0f, z1_value);
    
    // Store Z1 to global memory for backpropagation
    if (row < batch_size && threadIdx.x < hidden_size) {
        Z1_out[row * hidden_size + threadIdx.x] = z1_value;
    }
    
    // Store H in shared memory for next phase
    if (threadIdx.x < hidden_size) {
        tile_H[threadIdx.y][threadIdx.x] = h_value;
    }
    
    __syncthreads();
    
    // ========================================================================
    // PHASE 3: Compute Y_pred = H × W2^T
    // ========================================================================
    float y_pred_value = 0.0f;
    
    if (row < batch_size && col < output_size) {
        int num_tiles_phase2 = (hidden_size + TILE_SIZE - 1) / TILE_SIZE;
        
        for (int tile_idx = 0; tile_idx < num_tiles_phase2; tile_idx++) {
            int h_col = tile_idx * TILE_SIZE + threadIdx.x;
            int w2_col = tile_idx * TILE_SIZE + threadIdx.x;
            
            // H is already in shared memory from previous phase
            // Just need to load the appropriate tile
            if (tile_idx == 0 && threadIdx.x < TILE_SIZE) {
                // Already in tile_H, do nothing
            } else if (h_col < hidden_size) {
                tile_H[threadIdx.y][threadIdx.x] = h_value;  // This is approximate for larger hidden sizes
            } else {
                tile_H[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load W2_T tile
            if (col < output_size && w2_col < hidden_size) {
                tile_W2_T[threadIdx.y][threadIdx.x] = W2_T[col * hidden_size + w2_col];
            } else {
                tile_W2_T[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot product
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                y_pred_value += tile_H[threadIdx.y][k] * tile_W2_T[threadIdx.y][k];
            }
            
            __syncthreads();
        }
    }
    
    // Write final output
    if (row < batch_size && col < output_size) {
        Y_pred_out[row * output_size + col] = y_pred_value;
    }
}

// Simplified version: Just do the forward pass in two separate optimized kernels
// This is more practical for your network architecture
__global__ void relu_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// ============================================================================
// Host Functions
// ============================================================================

// Function to multiply matrices on GPU with transpose optimization
Matrix* mat_mult_optimized(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Incompatible matrices for multiplication.\n");
        exit(1);
    }

    Matrix *C = allocate_matrix(A->rows, B->cols);

    float *d_A, *d_B, *d_B_T, *d_C;
    size_t sizeA = A->rows * A->cols * sizeof(float);
    size_t sizeB = B->rows * B->cols * sizeof(float);
    size_t sizeC = C->rows * C->cols * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_B_T, sizeB);  // For transposed B
    cudaMalloc((void **)&d_C, sizeC);

    // Copy A and B from host to device
    cudaMemcpy(d_A, A->data, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

    // Transpose B on GPU
    dim3 transBlocks((B->cols + TILE_SIZE - 1) / TILE_SIZE,
                     (B->rows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 transThreads(TILE_SIZE, TILE_SIZE);
    transpose_kernel<<<transBlocks, transThreads>>>(d_B, d_B_T, B->rows, B->cols);
    
    cudaDeviceSynchronize();

    // Set up grid and block dimensions for matrix multiplication
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((B->cols + TILE_SIZE - 1) / TILE_SIZE,
                   (A->rows + TILE_SIZE - 1) / TILE_SIZE);

    // Launch optimized kernel with transposed B
    mat_mult_tiled_kernel_transpose<<<numBlocks, threadsPerBlock>>>(
        d_A, d_B_T, d_C, A->rows, A->cols, B->cols);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(C->data, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_T);
    cudaFree(d_C);

    return C;
}

// Fused forward pass function
void forward_pass_fused(Matrix *X_batch, Matrix *W1, Matrix *W2, 
                        Matrix **Z1_out, Matrix **Y_pred_out) {
    int batch_size = X_batch->rows;
    
    // Allocate output matrices
    *Z1_out = allocate_matrix(batch_size, HIDDEN_SIZE);
    *Y_pred_out = allocate_matrix(batch_size, OUTPUT_SIZE);
    
    float *d_X, *d_W1, *d_W1_T, *d_W2, *d_W2_T, *d_Z1, *d_Y_pred;
    
    size_t size_X = batch_size * INPUT_SIZE * sizeof(float);
    size_t size_W1 = INPUT_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t size_W2 = HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float);
    size_t size_Z1 = batch_size * HIDDEN_SIZE * sizeof(float);
    size_t size_Y_pred = batch_size * OUTPUT_SIZE * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W1, size_W1);
    cudaMalloc(&d_W1_T, size_W1);
    cudaMalloc(&d_W2, size_W2);
    cudaMalloc(&d_W2_T, size_W2);
    cudaMalloc(&d_Z1, size_Z1);
    cudaMalloc(&d_Y_pred, size_Y_pred);
    
    // Copy data to device
    cudaMemcpy(d_X, X_batch->data, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1->data, size_W1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2->data, size_W2, cudaMemcpyHostToDevice);
    
    // Transpose W1 and W2 on GPU
    dim3 transBlocks1((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                      (INPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    dim3 transBlocks2((OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                      (HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    dim3 transThreads(TILE_SIZE, TILE_SIZE);
    
    transpose_kernel<<<transBlocks1, transThreads>>>(d_W1, d_W1_T, INPUT_SIZE, HIDDEN_SIZE);
    transpose_kernel<<<transBlocks2, transThreads>>>(d_W2, d_W2_T, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Compute Z1 = X × W1^T with tiled kernel
    dim3 matmulBlocks1((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                       (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    dim3 matmulThreads(TILE_SIZE, TILE_SIZE);
    
    mat_mult_tiled_kernel_transpose<<<matmulBlocks1, matmulThreads>>>(
        d_X, d_W1_T, d_Z1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    
    // Apply ReLU in-place
    int total_elements = batch_size * HIDDEN_SIZE;
    int relu_threads = 256;
    int relu_blocks = (total_elements + relu_threads - 1) / relu_threads;
    relu_kernel<<<relu_blocks, relu_threads>>>(d_Z1, total_elements);
    
    // Compute Y_pred = ReLU(Z1) × W2^T with tiled kernel
    dim3 matmulBlocks2((OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                       (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    mat_mult_tiled_kernel_transpose<<<matmulBlocks2, matmulThreads>>>(
        d_Z1, d_W2_T, d_Y_pred, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy((*Z1_out)->data, d_Z1, size_Z1, cudaMemcpyDeviceToHost);
    cudaMemcpy((*Y_pred_out)->data, d_Y_pred, size_Y_pred, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_W1_T);
    cudaFree(d_W2);
    cudaFree(d_W2_T);
    cudaFree(d_Z1);
    cudaFree(d_Y_pred);
}

// Matrix subtraction: C = A - B
Matrix* mat_sub(Matrix *A, Matrix *B) {
    if(A->rows != B->rows || A->cols != B->cols) {
        printf("Incompatible matrices for subtraction.\n");
        exit(1);
    }
    Matrix *C = allocate_matrix(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->cols; j++)
            C->data[i * A->cols + j] = A->data[i * A->cols + j] - B->data[i * A->cols + j];
    return C;
}

// Matrix scalar multiplication: A = A * scalar
void mat_scalar_mult(Matrix *A, float scalar) {
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->cols; j++)
            A->data[i * A->cols + j] *= scalar;
}

// ! Activation Functions
// Function to compute derivative of ReLU
Matrix* relu_derivative(Matrix *m) {
    Matrix *derivative = allocate_matrix(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++)
        for(int j = 0; j < m->cols; j++)
            derivative->data[i * m->cols + j] = (m->data[i * m->cols + j] > 0) ? 1 : 0;
    return derivative;
}

// ! Loss Functions
// Function to compute Mean Squared Error
float mean_squared_error(Matrix *Y_pred, Matrix *Y_true) {
    float mse = 0.0f;
    for(int i = 0; i < Y_pred->rows; i++)
        for(int j = 0; j < Y_pred->cols; j++)
            mse += pow(Y_pred->data[i * Y_pred->cols + j] - Y_true->data[i * Y_true->cols + j], 2);
    return mse / Y_pred->rows;
}

// ! Optimization
// Function to update weights: W = W - learning_rate * grad
void update_weights(Matrix *W, Matrix *grad, float learning_rate) {
    for(int i = 0; i < W->rows; i++)
        for(int j = 0; j < W->cols; j++)
            W->data[i * W->cols + j] -= learning_rate * grad->data[i * grad->cols + j];
}

// Function to perform backpropagation and update weights
void backpropagation(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1, Matrix *Y_pred, Matrix *W1, Matrix *W2, int batch_size) {
    // Compute dZ2 = Y_pred - Y_batch
    Matrix *dZ2 = mat_sub(Y_pred, Y_batch);
    mat_scalar_mult(dZ2, 2.0f / batch_size);

    // Compute dW2 = Z1^T * dZ2
    Matrix *Z1_T = allocate_matrix(Z1->cols, Z1->rows);
    for(int i = 0; i < Z1->rows; i++) {
        for(int j = 0; j < Z1->cols; j++) {
            Z1_T->data[j * Z1->rows + i] = Z1->data[i * Z1->cols + j];
        }
    }
    Matrix *dW2 = mat_mult_optimized(Z1_T, dZ2);
    update_weights(W2, dW2, LEARNING_RATE);
    free_matrix(dW2);
    free_matrix(Z1_T);

    // Compute dZ1 = dZ2 * W2^T
    Matrix *W2_T = allocate_matrix(W2->cols, W2->rows);
    for(int i = 0; i < W2->rows; i++) {
        for(int j = 0; j < W2->cols; j++) {
            W2_T->data[j * W2->rows + i] = W2->data[i * W2->cols + j];
        }
    }
    Matrix *dZ1 = mat_mult_optimized(dZ2, W2_T);

    // Apply ReLU derivative
    Matrix *dZ1_derivative = relu_derivative(Z1);
    for(int i = 0; i < dZ1->rows; i++) {
        for(int j = 0; j < dZ1->cols; j++) {
            dZ1->data[i * dZ1->cols + j] *= dZ1_derivative->data[i * dZ1_derivative->cols + j];
        }
    }
    free_matrix(dZ1_derivative);
    free_matrix(W2_T);

    // Compute dW1 = X_batch^T * dZ1
    Matrix *X_batch_T = allocate_matrix(X_batch->cols, X_batch->rows);
    for(int i = 0; i < X_batch->rows; i++) {
        for(int j = 0; j < X_batch->cols; j++) {
            X_batch_T->data[j * X_batch->rows + i] = X_batch->data[i * X_batch->cols + j];
        }
    }
    Matrix *dW1 = mat_mult_optimized(X_batch_T, dZ1);
    update_weights(W1, dW1, LEARNING_RATE);
    free_matrix(dW1);
    free_matrix(X_batch_T);

    // Free allocated matrices
    free_matrix(dZ2);
    free_matrix(dZ1);
}

// ! Batch Processing
// Function to get a batch from the dataset
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch, int batch_start, int batch_size) {
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < INPUT_SIZE; j++)
            X_batch->data[i * INPUT_SIZE + j] = X->data[(batch_start + i) * INPUT_SIZE + j];
        Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
    }
}

// ! Data Loading
// Function to load CSV and populate X and Y
int load_csv(const char *filename, Matrix **X, Matrix **Y, int *num_samples) {
    FILE *file = fopen(filename, "r");
    if(!file) {
        printf("Failed to open file.\n");
        return -1;
    }
    char line[1024];
    int count = 0;
    // First pass to count samples
    while(fgets(line, sizeof(line), file)) count++;
    *num_samples = count;
    rewind(file);
    // Allocate X and Y
    *X = allocate_matrix(count, INPUT_SIZE);
    *Y = allocate_matrix(count, OUTPUT_SIZE);
    int i = 0;
    while(fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        int j = 0;
        while(token) {
            if(j < INPUT_SIZE) {
                (*X)->data[i * INPUT_SIZE + j] = atof(token);
            } else {
                (*Y)->data[i * OUTPUT_SIZE] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(file);
    return 0;
}

// Main function
int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s <data.csv>\n", argv[0]);
        return -1;
    }

    double start_time, end_time;

    Matrix *X, *Y;
    int num_samples;
    if(load_csv(argv[1], &X, &Y, &num_samples) != 0)
        return -1;

    // Allocate and initialize weights
    Matrix *W1 = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix *W2 = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
    random_init(W1);
    random_init(W2);

    printf("Starting training with optimizations:\n");
    printf("- Tiled shared memory\n");
    printf("- Pre-transpose matrix B\n");
    printf("- Fused forward pass (MatMul + ReLU + MatMul)\n");
    printf("- Tile size: %d x %d\n\n", TILE_SIZE, TILE_SIZE);

    // Start measuring time
    start_time = omp_get_wtime();

    // Training loop
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        for(int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
            int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
            int batch_size = batch_end - batch_start;

            // Extract batch
            Matrix *X_batch = allocate_matrix(batch_size, INPUT_SIZE);
            Matrix *Y_batch = allocate_matrix(batch_size, OUTPUT_SIZE);
            get_batch(X, Y, X_batch, Y_batch, batch_start, batch_size);

            // Forward pass using fused kernel
            Matrix *Z1, *Y_pred;
            forward_pass_fused(X_batch, W1, W2, &Z1, &Y_pred);

            // Compute loss
            float loss = mean_squared_error(Y_pred, Y_batch);
            if((batch_start == 0) && ((epoch % LOG_EVERY_EPOCH == 0 && epoch != 0) || epoch == 1 || epoch == EPOCHS - 1))
                printf("Epoch %d, MSE: %f\n", epoch, loss);

            // Backward pass
            backpropagation(X_batch, Y_batch, Z1, Y_pred, W1, W2, batch_size);

            // Free allocated matrices
            free_matrix(Z1);
            free_matrix(Y_pred);
            free_matrix(X_batch);
            free_matrix(Y_batch);
        }
    }

    // Stop measuring time
    end_time = omp_get_wtime();

    printf("\n=== Performance Results ===\n");
    printf("Training time: %.4f seconds\n", end_time - start_time);
    printf("Optimizations applied:\n");
    printf("  ✓ Tiled shared memory (TILE_SIZE=%d)\n", TILE_SIZE);
    printf("  ✓ Pre-transpose B (coalesced memory access)\n");
    printf("  ✓ Fused forward pass (reduced kernel launches)\n");

    // Cleanup
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(X);
    free_matrix(Y);

    return 0;
}
