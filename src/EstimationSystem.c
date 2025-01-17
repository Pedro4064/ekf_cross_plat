#include "EstimationSystem.h"
#include "math.h"

#define GLOBAL_TEST 1

float64_t fStateVector[8];

float64_t fZ[2][1];
float64_t fU[2][1];
float64_t fX_hat_priori[NUM_STATES];
float64_t fP_priori[NUM_STATES][NUM_STATES];
float64_t fK[NUM_STATES][2];
float64_t fF_jacobian[NUM_STATES][NUM_STATES];
float64_t fH_jacobian[2][NUM_STATES];
float64_t fH_nonlinear[2];

float64_t fP_posteriori[NUM_STATES][NUM_STATES] = {
                                                            {0.1, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 0.1, 0, 0, 0, 0, 0, 0},
                                                            {0, 0, 0.1, 0, 0, 0, 0, 0},
                                                            {0, 0, 0, 0.1, 0, 0, 0, 0},
                                                            {0, 0, 0, 0, 0.1, 0, 0, 0},
                                                            {0, 0, 0, 0, 0, 0.1, 0, 0},
                                                            {0, 0, 0, 0, 0, 0, 0.1, 0},
                                                            {0, 0, 0, 0, 0, 0, 0, 0.1},
                                                        };

float64_t fQ[NUM_STATES][NUM_STATES] = {
                                                            {0.1, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 0.1, 0, 0, 0, 0, 0, 0},
                                                            {0, 0, 0.1, 0, 0, 0, 0, 0},
                                                            {0, 0, 0, 0.1, 0, 0, 0, 0},
                                                            {0, 0, 0, 0, 0.1, 0, 0, 0},
                                                            {0, 0, 0, 0, 0, 0.1, 0, 0},
                                                            {0, 0, 0, 0, 0, 0, 0.1, 0},
                                                            {0, 0, 0, 0, 0, 0, 0, 0.1},
                                                        };

float64_t fR[2][2] = {
                                {10.0f, 0.0f},
                                {0.0f, 10.00f}
                            };

float64_t fF_P[NUM_STATES][NUM_STATES];     // Intermediate result F * P_posteriori




static arm_matrix_instance_f64 mZ;
static arm_matrix_instance_f64 mU;
static arm_matrix_instance_f64 mX_hat_posteriori;
static arm_matrix_instance_f64 mX_hat_priori;
static arm_matrix_instance_f64 mP_posteriori;
static arm_matrix_instance_f64 mP_priori;
static arm_matrix_instance_f64 mK;
static arm_matrix_instance_f64 mH_jacobian;
static arm_matrix_instance_f64 mH_nonlinear;
static arm_matrix_instance_f64 mF_jacobian;
static arm_matrix_instance_f64 mQ;
static arm_matrix_instance_f64 mR;



/**
 * @brief Computes the dynamic model  f(x, u)
 * 
 */
void vEstimationSystemComputeDynamicModel(float64_t u1, float64_t u2){
    float64_t Ts = 0.1f;
    float64_t r = 0.035f;
    float64_t L = 0.1f;

    // X Hat priori calculations
    fX_hat_priori[0] = fStateVector[0] + fStateVector[1] * Ts + 0.5f * fStateVector[2] * Ts * Ts;
    fX_hat_priori[1] = (r / 2.0f) * cos(fStateVector[6]) * (u1 + u2);
    fX_hat_priori[2] = fStateVector[2];
    fX_hat_priori[3] = fStateVector[3] + fStateVector[4] * Ts + 0.5f * fStateVector[5] * Ts * Ts;
    fX_hat_priori[4] = (r / 2.0f) * sin(fStateVector[6]) * (u1 + u2);
    fX_hat_priori[5] = fStateVector[5];
    fX_hat_priori[6] = fStateVector[6] + fStateVector[7] * Ts;
    fX_hat_priori[7] = (r / L) * (u2 - u1);
}

/**
 * @brief  Calculate the jacobian of the f(x, u) function
 * 
 * @param u1  Left motor RPM
 * @param u2  Right motor RPM
 */
void vEstimationSystemCalculateFjacobian(float64_t u1, float64_t u2){

    float64_t Ts = 0.1f;
    float64_t r = 0.035f;
    float64_t L = 0.1f;

    float64_t *x = fStateVector;

    fF_jacobian[0][0] = 1.0f;
    fF_jacobian[0][1] = Ts;
    fF_jacobian[0][2] = 0.5f * Ts * Ts;

    fF_jacobian[1][6] = -(r / 2.0f) * sin(x[6]) * (u1 + u2);

    fF_jacobian[3][3] = 1.0f;
    fF_jacobian[3][4] = Ts;
    fF_jacobian[3][5] = 0.5f * Ts * Ts;

    fF_jacobian[4][6] = (r / 2.0f) * cos(x[6]) * (u1 + u2);

    fF_jacobian[6][6] = 1.0f;
    fF_jacobian[6][7] = Ts;

    fF_jacobian[7][7] = 0.0f; // Add other terms as required
}

/**
 * @brief Computes both the nonlinear observation matrix h(x, u) and its jacobian H(x, u)
 * 
 */
void vEstimationSystemComputeObservation(void){

    // Compute Jacobian of H
    float64_t *x = fX_hat_priori;

    float64_t t2 = cos(x[6]); // x(7) in MATLAB, zero-based index 6 in C
    float64_t t3 = sin(x[6]);

    fH_jacobian[0][2] = t2; // Corresponds to t2
    fH_jacobian[0][5] = t3; // Corresponds to t3
    fH_jacobian[0][6] = -t3 * x[2] + t2 * x[5]; // -t3 * x(3) + t2 * x(6) in MATLAB
    fH_jacobian[1][7] = 1.0f; // Corresponds to 1.0

    // Computes the nonlinear observation matrix h
    fH_nonlinear[0] = x[2] * cos(x[6]) + x[5] * sin(x[6]); // x(3)*cos(x(7)) + x(6)*sin(x(7))
    fH_nonlinear[1] = x[7]; // x(8)



}

/**
 * @brief Predict P for next iteration
 * 
 */
void vEstimationSystemEkfPredict(void){

    #if GLOBAL_TEST == 0
        static char cFirstIteration = 1;
        if(cFirstIteration){
            // Initialize all matrices used in the extended kalman filter implementation
            arm_mat_init_f64(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
            arm_mat_init_f64(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
            arm_mat_init_f64(&mQ, NUM_STATES, NUM_STATES, &fQ[0][0]);
            arm_mat_init_f64(&mF_jacobian, NUM_STATES, NUM_STATES, &fF_jacobian[0][0]);

            cFirstIteration = 0;
            return;
        }
    #endif

    float64_t fF_trans[NUM_STATES][NUM_STATES]; // Transpose of F
    float64_t fF_P_Ft[NUM_STATES][NUM_STATES];  // Intermediate result F * P_posteriori * F^T

    arm_matrix_instance_f64 mF_trans, mF_P_Ft, mF_P;
    arm_mat_init_f64(&mF_trans, NUM_STATES, NUM_STATES, &fF_trans[0][0]);
    arm_mat_init_f64(&mF_P_Ft, NUM_STATES, NUM_STATES, &fF_P_Ft[0][0]);
    arm_mat_init_f64(&mF_P, NUM_STATES, NUM_STATES, &fF_P[0][0]);

    // Compute F * P_posteriori
    arm_mat_mult_f64(&mF_jacobian, &mP_posteriori, &mF_P);

    // Compute F^T
    arm_mat_trans_f64(&mF_jacobian, &mF_trans);

    // Compute F * P_posteriori * F^T
    arm_mat_mult_f64(&mF_P, &mF_trans, &mF_P_Ft);

    // Add Q to get P_priori
    arm_mat_add_f64(&mF_P_Ft, &mQ, &mP_priori);

}

void vEstimationSystemCalculateK(void){
    float64_t fH_trans[NUM_STATES][2];  // Transpose of H
    float64_t fHPH_t_R[2][2];          // H * P_priori * H' + R
    float64_t fHPH_t_R_inv[2][2];      // Inverse of H * P_priori * H' + R
    float64_t fK_temp[NUM_STATES][2];  // K intermediate
    float64_t fHPH_t[2][2];            // H * P_priori * H'

    arm_matrix_instance_f64 mH_trans;
    arm_matrix_instance_f64 mHPH_t;
    arm_matrix_instance_f64 mHPH_t_R;
    arm_matrix_instance_f64 mHPH_t_R_inv;
    arm_matrix_instance_f64 mK_temp;

    #if GLOBAL_TEST == 0

        static char cFirstIteration = 1;
        if(cFirstIteration){
            // Initialize all matrices used in the extended kalman filter implementation
            arm_mat_init_f64(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
            arm_mat_init_f64(&mR, 2, 2, &fR[0][0]);
            arm_mat_init_f64(&mK, NUM_STATES, 2, &fK[0][0]);
            arm_mat_init_f64(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);
            arm_mat_init_f64(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);   // INPUT

            cFirstIteration = 0;
        }

    #endif

    arm_mat_init_f64(&mH_trans, NUM_STATES, 2, (float64_t *)fH_trans);
    arm_mat_init_f64(&mHPH_t_R, 2, 2, (float64_t *)fHPH_t_R);
    arm_mat_init_f64(&mHPH_t_R_inv, 2, 2, (float64_t *)fHPH_t_R_inv);
    arm_mat_init_f64(&mK_temp, NUM_STATES, 2, (float64_t *)fK_temp);
    arm_mat_init_f64(&mHPH_t, 2, 2, &fHPH_t[0][0]);

    // Calculate K gain //////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    // H' (Transpose of H_jacobian)
    arm_mat_trans_f64(&mH_jacobian, &mH_trans);

    // H * P_priori * H'
    arm_mat_mult_f64(&mH_jacobian, &mP_priori, &mHPH_t);
    arm_mat_mult_f64(&mHPH_t, &mH_trans, &mHPH_t);

    // H * P_priori * H' + R
    arm_mat_add_f64(&mHPH_t, &mR, &mHPH_t_R);

    // Inverse of H * P_priori * H' + R
    arm_mat_inverse_f64(&mHPH_t_R, &mHPH_t_R_inv);

    // P_priori * H'
    arm_mat_mult_f64(&mP_priori, &mH_trans, &mK_temp);

    // K = P_priori * H' * inv(H * P_priori * H' + R)
    arm_mat_mult_f64(&mK_temp, &mHPH_t_R_inv, &mK);


}

void vEstimationSystemCalculateXposteriori(float64_t z1, float64_t z2){
    
    #if GLOBAL_TEST == 0
        static char cFirstIteration = 1;
        if(cFirstIteration){
            // Initialize all matrices used in the extended kalman filter implementation
            arm_mat_init_f64(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
            arm_mat_init_f64(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
            arm_mat_init_f64(&mK, NUM_STATES, 2, &fK[0][0]);

            cFirstIteration = 0;
            return;
        }
    #endif

    fZ[0][0] = z1;
    fZ[1][0] = z2;

    float64_t fZ_h[2][1];  // z - h
    float64_t fKZ_h[NUM_STATES][1]; // K * (z - h)

    arm_matrix_instance_f64 mZ_h;
    arm_matrix_instance_f64 mKZ_h;

    arm_mat_init_f64(&mZ_h, 2, 1, (float64_t *)fZ_h);
    arm_mat_init_f64(&mKZ_h, NUM_STATES, 1, (float64_t *)fKZ_h);

    // z - h
    for (int i = 0; i < 2; i++) {
        fZ_h[i][0] = fZ[i][0] - fH_nonlinear[i];
    }

    // K * (z - h)
    arm_mat_mult_f64(&mK, &mZ_h, &mKZ_h);

    // x_hat_posteriori = x_hat_priori + K * (z - h)
    arm_mat_add_f64(&mX_hat_priori, &mKZ_h, &mX_hat_posteriori);
}

void vEstimationSystemCalculatePposteriori(){

    #if GLOBAL_TEST == 0
        static char cFirstIteration = 1;
        if(cFirstIteration){
            // Initialize all matrices used in the extended kalman filter implementation
            arm_mat_init_f64(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
            arm_mat_init_f64(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
            arm_mat_init_f64(&mR, 2, 2, &fR[0][0]);
            arm_mat_init_f64(&mK, NUM_STATES, 2, &fK[0][0]);
            arm_mat_init_f64(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);

            cFirstIteration = 0;
            return;
        }
    #endif

    // Calculate P   /////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////
    float64_t fI[NUM_STATES][NUM_STATES];  // Identity matrix
    float64_t fIKH[NUM_STATES][NUM_STATES]; // (I - K * H)
    float64_t fIKH_t[NUM_STATES][NUM_STATES]; // Transpose of (I - K * H)
    float64_t fIKHP[NUM_STATES][NUM_STATES]; // (I - K * H) * P_priori
    float64_t fKRK[NUM_STATES][NUM_STATES]; // K * R * K'

    arm_matrix_instance_f64 mI;
    arm_matrix_instance_f64 mIKH;
    arm_matrix_instance_f64 mIKH_t;
    arm_matrix_instance_f64 mIKHP;
    arm_matrix_instance_f64 mKRK;

    arm_mat_init_f64(&mI, NUM_STATES, NUM_STATES, (float64_t *)fI);
    arm_mat_init_f64(&mIKH, NUM_STATES, NUM_STATES, (float64_t *)fIKH);
    arm_mat_init_f64(&mIKH_t, NUM_STATES, NUM_STATES, (float64_t *)fIKH_t);
    arm_mat_init_f64(&mIKHP, NUM_STATES, NUM_STATES, (float64_t *)fIKHP);
    arm_mat_init_f64(&mKRK, NUM_STATES, NUM_STATES, (float64_t *)fKRK);

    // Identity matrix
    for (int i = 0; i < NUM_STATES; i++) {
        for (int j = 0; j < NUM_STATES; j++) {
            fI[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // I - K * H
    arm_mat_mult_f64(&mK, &mH_jacobian, &mIKH);
    arm_mat_sub_f64(&mI, &mIKH, &mIKH);

    // (I - K * H)' (Transpose)
    arm_mat_trans_f64(&mIKH, &mIKH_t);

    // (I - K * H) * P_priori * (I - K * H)'
    arm_mat_mult_f64(&mIKH, &mP_priori, &mIKHP);
    arm_mat_mult_f64(&mIKHP, &mIKH_t, &mP_posteriori);

    // K * R * K'
    arm_mat_mult_f64(&mK, &mR, &mKRK);
    arm_mat_mult_f64(&mKRK, &mK, &mKRK);

    // Add K * R * K' to P_posteriori
    arm_mat_add_f64(&mP_posteriori, &mKRK, &mP_posteriori);
}

void vEstimationSystemEkfEstimate(float64_t z1, float64_t z2){

    #if GLOBAL_TEST == 0
        static char cFirstIteration = 1;

        if(cFirstIteration){
            // Initialize all matrices used in the extended kalman filter implementation
            arm_mat_init_f64(&mZ, 2, 1, &fZ[0][0]);
            arm_mat_init_f64(&mU, 2, 1, &fU[0][0]);
            arm_mat_init_f64(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
            arm_mat_init_f64(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
            arm_mat_init_f64(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
            arm_mat_init_f64(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
            arm_mat_init_f64(&mQ, NUM_STATES, NUM_STATES, &fQ[0][0]);
            arm_mat_init_f64(&mR, 2, 2, &fR[0][0]);
            arm_mat_init_f64(&mK, NUM_STATES, 2, &fK[0][0]);
            arm_mat_init_f64(&mF_jacobian, NUM_STATES, NUM_STATES, &fF_jacobian[0][0]);
            arm_mat_init_f64(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);
            arm_mat_init_f64(&mH_nonlinear, 2, 1, &fH_nonlinear[0]);

            cFirstIteration = 0;
        }
    #endif

    vEstimationSystemCalculateK();
    vEstimationSystemCalculateXposteriori(z1, z2);
    vEstimationSystemCalculatePposteriori();
    

}

void vEstimationSystemComputeEstimate(float64_t u1, float64_t u2, float64_t z1, float64_t z2){

    static char cFirstIteration = 1;
    if(cFirstIteration){
        // Initialize all matrices used in the extended kalman filter implementation
        arm_mat_init_f64(&mZ, 2, 1, &fZ[0][0]);
        arm_mat_init_f64(&mU, 2, 1, &fU[0][0]);
        arm_mat_init_f64(&mX_hat_posteriori, NUM_STATES, 1, &(fStateVector[0]));
        arm_mat_init_f64(&mX_hat_priori, NUM_STATES, 1, &fX_hat_priori[0]);
        arm_mat_init_f64(&mP_posteriori, NUM_STATES, NUM_STATES, &fP_posteriori[0][0]);
        arm_mat_init_f64(&mP_priori, NUM_STATES, NUM_STATES, &fP_priori[0][0]);
        arm_mat_init_f64(&mQ, NUM_STATES, NUM_STATES, &fQ[0][0]);
        arm_mat_init_f64(&mR, 2, 2, &fR[0][0]);
        arm_mat_init_f64(&mK, NUM_STATES, 2, &fK[0][0]);
        arm_mat_init_f64(&mF_jacobian, NUM_STATES, NUM_STATES, &fF_jacobian[0][0]);
        arm_mat_init_f64(&mH_jacobian, 2, NUM_STATES, &fH_jacobian[0][0]);
        arm_mat_init_f64(&mH_nonlinear, 2, 1, &fH_nonlinear[0]);


        vEstimationSystemCalculateFjacobian(u1, u2);
        vEstimationSystemComputeDynamicModel(u1, u2);
        vEstimationSystemEkfPredict();
        cFirstIteration = 0;
        return;
    }
    
    vEstimationSystemComputeObservation();
    vEstimationSystemEkfEstimate(z1, z2);

    vEstimationSystemCalculateFjacobian(u1, u2);
    vEstimationSystemComputeDynamicModel(u1, u2);
    vEstimationSystemEkfPredict();
}