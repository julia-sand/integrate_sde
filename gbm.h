#ifndef GBM_H
#define GBM_H

double integrate_gbm(double x0, int tsteps, double dt);
void generate_training_data(double *X0, double *XT);
double mse_loss(double *XT, double *X0);

#endif
