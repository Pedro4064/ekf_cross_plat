#include "arm_math.h"


#ifndef __ESTIMATION_SYSTEM_H__
#define __ESTIMATION_SYSTEM_H__


#define NUM_STATES 8

typedef struct STATE_ESTIMATE{
    float32_t fX;
    float32_t fX_dot;
    float32_t fX_ddot;
    float32_t fY;
    float32_t fY_dot;
    float32_t fY_ddot;
    float32_t fTheta;
    float32_t fTheta_dot;
}StateEstimate;

typedef union SYSTEM_STATE{
    StateEstimate xStateStruct;
    float32_t fStateVector[8];
}SystemState;

extern float64_t fStateVector[8];
extern  float64_t fX_hat_priori[NUM_STATES];
extern float64_t fP_priori[NUM_STATES][NUM_STATES];
extern float64_t fP_posteriori[NUM_STATES][NUM_STATES];
extern float64_t fF_jacobian[NUM_STATES][NUM_STATES];
extern float64_t fH_jacobian[2][NUM_STATES];
extern float64_t fH_nonlinear[2];
extern float64_t fK[NUM_STATES][2];
extern float64_t fF_P[NUM_STATES][NUM_STATES];     // Intermediate result F * P_posteriori

// extern float32_t fHPH_t[2][2];            // H * P_priori * H'

// void vEstimationSystemInit(SystemState* pSysState, TelemetryData* pTelemData, MotorCommands* pMotorCommands);
extern void vEstimationSystemComputeEstimate(float64_t u1, float64_t u2, float64_t z1, float64_t z2);
extern void vEstimationSystemComputeObservation(void);
extern void vEstimationSYstemCalculateK(void);
extern void vEstimationSystemCalculateXposteriori(float64_t z1, float64_t z2);
extern void vEstimationSystemCalculatePposteriori();

extern void vEstimationSystemComputeDynamicModel(float64_t u1, float64_t u2);
extern void vEstimationSystemEkfPredict(void);

extern void CalculateFjacobian(float64_t u1, float64_t u2);
extern void EstimationStep(float64_t z1, float64_t z2);
extern void PredictionStep(float64_t u1, float64_t u2);
#endif