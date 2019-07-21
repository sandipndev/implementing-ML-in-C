#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
matrix feature_scaling(matrix);
matrix single_feature_scaling(matrix);
matrix _multivariate_linear_regression(matrix, matrix, float, int);

int main() {
    system("clear");

    // Getting important numbers
    int n_features, n_observations;
    printf("?n(degree of polynomial): ");
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

    printf("--BEFORE FEATURE SCALING--\n");
    show_matrix(dataset);
    show_matrix(Y);

    // Feature Scaling
    dataset = feature_scaling(dataset);
    Y = single_feature_scaling(Y);

    printf("\n--AFTER FEATURE SCALING--\n");
    show_matrix(dataset);
    show_matrix(Y);

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
    float term;
    for (i=0; i<rx.rows; i++) {
        printf("?%dth x: ", i+1);
        scanf("%f", &term);
        for (j=0; j<rx.cols; j++) {
            if (j==0) {
                rx.base_addr[i][j] = 1;
                continue;
            }
            rx.base_addr[i][j] = pow(term, j);
        }
        printf("?%dth y: ", i+1);
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

// Implements feature scaling
matrix feature_scaling(matrix rx) {
    matrix tx = create_matrix(rx.rows, rx.cols);
    matrix means = create_matrix(1, rx.cols - 1);
    matrix std_devs = create_matrix(1, rx.cols - 1);

    int i, j;
    float sum = 0;

    // Mean
    for (i=1; i<rx.cols; i++) {
        sum = 0;
        for (j=0; j<rx.rows; j++) {
            sum += rx.base_addr[j][i];
        }
        means.base_addr[0][i-1] = sum / rx.rows;
    }

    // Standard Deviation
    for (i=1; i<rx.cols; i++) {
        sum = 0;
        for (j=0; j<rx.rows; j++) {
            sum += pow(rx.base_addr[j][i] - means.base_addr[0][i-1], 2);
        }
        std_devs.base_addr[0][i-1] = sqrt(sum / rx.rows);
    }

    // Normalizing into tx
    for(i=0; i<rx.rows; i++) {
        for (j=0; j<rx.cols; j++) {
            if (j==0)
                tx.base_addr[i][j] = rx.base_addr[i][j];
            else 
                tx.base_addr[i][j] = (rx.base_addr[i][j] - means.base_addr[0][j-1]) / std_devs.base_addr[0][j-1];
        }
    }

    // Freeing stuffs
    free(means.base_addr);
    free(std_devs.base_addr);
    free(rx.base_addr);


    return tx;

}

// For Y - for 1 col matrix
matrix single_feature_scaling(matrix rx) {
    int i;
    float std_dev, mean, sum=0;
    for (i=0; i<rx.rows; i++) {
        sum += rx.base_addr[i][0];
    }
    mean = sum / rx.rows;
    sum = 0;
    for (i=0; i<rx.rows; i++) {
        sum += pow(rx.base_addr[i][0] - mean, 2);
    }
    std_dev = sqrt(sum / rx.rows);
    matrix tx = create_matrix(rx.rows, rx.cols);
    for (i=0; i<rx.rows; i++) {
        tx.base_addr[i][0] = (rx.base_addr[i][0] - mean) / std_dev;
    }
    free(rx.base_addr);
    return tx;
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