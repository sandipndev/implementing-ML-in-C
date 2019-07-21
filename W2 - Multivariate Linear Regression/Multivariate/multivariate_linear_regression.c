#include <stdio.h>
#include <stdlib.h>

// It's all about matrices
typedef struct {
    float **base_addr;
    int rows;
    int cols;
} matrix;

// Function Declarations
matrix create_matrix(int, int);
void take_user_input(matrix, matrix);
void show_matrix(matrix);
matrix transpose(matrix);
matrix multiply(matrix, matrix);
float _mse_loss(matrix, matrix, matrix);
matrix _multivariate_linear_regression(matrix, matrix, float, int);

int main() {
    system("clear");

    // Getting important numbers
    int n_features, n_observations;
    printf("?n(features): ");
    scanf("%d", &n_features);
    printf("?n(observations): ");
    scanf("%d", &n_observations);

    // Creating and entering values in the dataset
    matrix dataset = create_matrix(n_observations, n_features+1);
    matrix Y = create_matrix(n_observations, 1);
    printf("Please enter the values in the dataset:-\n");
    take_user_input(dataset, Y);

     // Parameters input - Learning Rate
    float learning_rate;
    printf("?Learning Rate: ");
    scanf("%f", &learning_rate);
    
    // Parameters input - Epochs
    int epochs;
    printf("?Epochs: ");
    scanf("%d", &epochs);

    matrix res = _multivariate_linear_regression(dataset, Y, learning_rate, epochs);
    show_matrix(res);
    printf("MSE Error: %f\n", _mse_loss(dataset, Y, res));

    free(res.base_addr);
    free(Y.base_addr);
    free(dataset.base_addr);

    return 0;
}

// Creates a matrix and returns that
matrix create_matrix(int rows, int cols) {
    matrix tx;
    tx.base_addr = (float **) calloc(rows, sizeof(float *));
    tx.rows = rows;
    tx.cols = cols;
    int i;
    for (i=0; i<rows; i++) {
        tx.base_addr[i] = (float *) calloc(cols, sizeof(float));
    }
    return tx;
}

// Takes user input for the matrix - initialize x0 with 1
void take_user_input(matrix rx, matrix y) {
    int i, j;
    for (i=0; i<rx.rows; i++) {
        for (j=0; j<rx.cols; j++) {
            if (j==0) {
                rx.base_addr[i][j] = 1;
                continue;
            }
            printf("?%dth obs-> %dth feature: ", i+1, j);
            scanf("%f", &rx.base_addr[i][j]);
        }
        printf("?%dth obs-> y-value: ", i+1);
        scanf("%f", &y.base_addr[i][0]);
    }
}

// Shows the matrix in proper format
void show_matrix(matrix rx) {
    int i, j;
    printf("[");
    for (i=0; i<rx.rows; i++) {
        if (i==0)
            printf("  [");
        else
            printf("   [");
        for (j=0; j<rx.cols; j++)
            printf(" %4.3f,", rx.base_addr[i][j]);
        if (i==rx.rows-1)
            printf("\b]   ");
        else
            printf("\b],\n");
    }
    printf("]\n");
}

// Transposes a matrix
matrix transpose(matrix rx) {
    matrix tx = create_matrix(rx.cols, rx.rows);
    int i,j;
    for (i=0; i<rx.rows; i++) {
        for (j=0; j<rx.cols; j++) {
            tx.base_addr[j][i] = rx.base_addr[i][j];
        }
    }
    return tx;
}

// Multiplies two matrices, assuming multiplication is possible
matrix multiply(matrix rx1, matrix rx2) {
    matrix tx = create_matrix(rx1.rows, rx2.cols);
    int i, j, k;
    float sum = 0;
    for (i=0; i<rx1.rows; i++) {
        for (j=0; j<rx2.cols; j++) {
            sum = 0;
            for (k=0; k<rx1.cols; k++) {
                sum += rx1.base_addr[i][k] * rx2.base_addr[k][j];
            }
            tx.base_addr[i][j] = sum;
        }
    }
    return tx;
}

// Calculate MSE Loss
// feature_vetor has a shape of (1,n)
float _mse_loss(matrix dataset, matrix y_values, matrix feature_vector) {
    dataset = transpose(dataset);
    matrix mul = multiply(feature_vector, dataset);
    float res=0, term;
    int i;
    for (i=0; i<mul.cols; i++) {
        term = (mul.base_addr[0][i] - y_values.base_addr[i][0]);
        term *= term;
        res += term;
    }
    res *= 1.0 / y_values.rows;
    free(mul.base_addr);
    return res;
}

// Implements Multivariate Linear Regression
matrix _multivariate_linear_regression(matrix dataset, matrix y_values, float learning_rate, int epochs) {
    matrix feature_vector = create_matrix(1, dataset.cols);
    matrix hyp;
    int i, j, k;
    float sum;

    // Initializing feature vector to zeroes
    for (i=0; i<dataset.cols; i++) {
        feature_vector.base_addr[0][i] = 0;
    }

    // Running learning
    for (i=0; i<epochs; i++) {
        hyp = multiply(feature_vector, transpose(dataset));

        // For every feature
        for (j=0; j<dataset.cols; j++) {
            sum = 0;

            for (k=0; k<dataset.rows; k++) {
                sum += (hyp.base_addr[0][k] - y_values.base_addr[k][0]) * dataset.base_addr[k][j];
            }

            feature_vector.base_addr[0][j] -= learning_rate * sum / ( (float) dataset.rows);
        }

        // printf("Loss in %dth epoch-> %f \tFeatureV-> ", i+1, _mse_loss(dataset, y_values, feature_vector));
        // show_matrix(feature_vector);
    }

    free(hyp.base_addr);
    return feature_vector;
}