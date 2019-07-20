#include <stdio.h>
#include <stdlib.h>

// Defines a 2D point
typedef struct {
    int x;
    int y;
} point;

// Defines a set of 2D points
typedef struct {
    point *base_addr;
    int len;
} datapoints;

// Defines the necessary attributes for a line (y = slope*x + y_intercept)
typedef struct {
    float slope;
    float y_intercept;
} line;

// Helper function to create datapoints quickly
datapoints create_datapoints(int);

// Helper function to take user input for the datapoints
void user_input(datapoints);

// Helper function to display entered datapoints
void display_datapoints(datapoints);

// Helper function to calculate MSE loss
float _mse_loss(line, datapoints);

// Helper functions to calculate the derivatives wrt slope and y_intercept
float _dL_dm(line, datapoints);
float _dL_dc(line, datapoints);

// Main function to fit a line to the given datapoints
line linear_regression_fit(datapoints, float, int);

// Main function - Execution begins here
int main() {

    // Length of the dataset
    int length;
    printf("?n(Data Points): ");
    scanf("%d", &length);

    // Generating the dataset
    datapoints dataset = create_datapoints(length);
    // Filling up the dataset
    printf("Please fill up the dataset:\n");
    user_input(dataset);

    // Showing the dataset
    printf("This is the data you've entered:\n");
    display_datapoints(dataset);

    // Parameters input - Learning Rate
    float learning_rate;
    printf("?Learning Rate: ");
    scanf("%f", &learning_rate);
    
    // Parameters input - Epochs
    int epochs;
    printf("?Epochs: ");
    scanf("%d", &epochs);

    // Fitting the line (Best fit)
    line rx = linear_regression_fit(dataset, learning_rate, epochs);
    printf("Calculated Best Fit Line:\n");
    printf("y = %4fx + %4f\n", rx.slope, rx.y_intercept);
    printf("MSE Loss = %6f\n", _mse_loss(rx, dataset));

    free(dataset.base_addr);
}

datapoints create_datapoints(int len) {
    // Creating the datapoints
    datapoints tx;

    // Assigning space and adding them to the struct
    tx.base_addr = (point *) calloc(len, sizeof(point));
    tx.len = len;

    return tx;
}

void user_input(datapoints rx) {
    int i;
    for (i=0; i<rx.len; i++) {
        printf("%dth -> ?x: ", i+1);
        scanf("%d", &rx.base_addr[i].x);
        printf("%dth -> ?y: ", i+1);
        scanf("%d", &rx.base_addr[i].y);
    }
}

void display_datapoints(datapoints rx) {
    int i;
    printf("[");
    for(i=0; i<rx.len; i++) {
        if (i==0)   printf("  ");
        else        printf("   ");
        printf("(%4d, %4d)", rx.base_addr[i].x, rx.base_addr[i].y);
        if (i==rx.len-1)    printf("   ");
        else                printf("\n");
    }
    printf("]\n");
}

float _mse_loss(line lx, datapoints dx) {
    float res=0;
    float term;
    int i;
    for (i=0; i<dx.len; i++) {
        term = (float) (lx.slope * dx.base_addr[i].x + lx.y_intercept) - (float) dx.base_addr[i].y;
        term *= term;
        res += term;
    }
    res *= 1.0/dx.len;
    return res;
}

float _dL_dm(line lx, datapoints dx) {
    float res=0;
    float term;
    int i;
    for (i=0; i<dx.len; i++) {
        term = (float) (lx.slope * dx.base_addr[i].x + lx.y_intercept) - (float) dx.base_addr[i].y;
        term *= (float) dx.base_addr[i].x;
        res += term;
    }
    res *= 2.0/dx.len;
    return res;
}

float _dL_dc(line lx, datapoints dx) {
    float res=0;
    float term;
    int i;
    for (i=0; i<dx.len; i++) {
        term = (float) (lx.slope * dx.base_addr[i].x + lx.y_intercept) - (float) dx.base_addr[i].y;
        res += term;
    }
    res *= 2.0/dx.len;
    return res;
}

line linear_regression_fit(datapoints dx, float lr, int epochs) {
    line res, temp;
    res.slope = 0;
    res.y_intercept = 0;
    int i;
    for ( i=0; i<epochs; i++) {
        temp = res;
        res.slope -= _dL_dm(temp, dx) * lr;
        res.y_intercept -= _dL_dc(temp, dx) * lr;
        printf("Loss after %d epochs = %f\n", i+1 , _mse_loss(res, dx));
    }
    return res;
}