#include "arm_math.h"
#include <stdio.h>

int main() {
    float64_t A[4] = {1, 1, 1, 1};  // 2x2 matrix
    float64_t B[4] = {1, 1, 1, 1};  // 2x2 matrix
    float64_t C[4];                 // Result 2x2 matrix
    arm_matrix_instance_f64 matA, matB, matC;

    arm_mat_init_f64(&matA, 2, 2, (float64_t *)A);
    arm_mat_init_f64(&matB, 2, 2, B);
    arm_mat_init_f64(&matC, 2, 2, C);

    if (arm_mat_add_f64(&matA, &matB, &matC) == ARM_MATH_SUCCESS) {
        printf("Matrix multiplication successful! Result:\n");
        for (int i = 0; i < 4; i++) {
            printf("%f ", C[i]);
        }
    } else {
        printf("Matrix multiplication failed.\n");
    }

    return 0;
}