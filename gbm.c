#include <stdio.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

/*problem variables*/
#define mu 1
#define sigma 1 
#define neurons_in 2
#define neurons_out 1
#define batch 10


double generate_uniform()
/*returns a U(0,1) random variable*/
{
    return (double)rand() / (double)RAND_MAX;
}


double generate_uniform_shifted(double lim)
/*returns a U(-limit,limit) random variable*/
{   
    
    return (double)((2*lim)*(double)generate_uniform()) - lim;
}

double nrand()
/*takes two uniform random variables and generates a normal random variable using the box muller transform*/
{
    
    return (double)sqrt(-2*log(generate_uniform()))*cos(4*acos(0)*generate_uniform()); 
}


double integrate_gbm(double x0, int tsteps, double dt)
{
   
    double xs[tsteps];
    xs[0] = x0;
    for (int i=1; i<tsteps; i++)
    {   
        xs[i] = xs[i-1] + mu*xs[i-1]*dt + sqrt(2*dt)*nrand();
    }
    return xs[tsteps-1];
}

void generate_training_data(double *X0, double *XT)
{
    srand(time(NULL)); // seed with currtime
    
    double dt = 0.01;
        
    for (int i=0; i<batch;i++)
    {
        X0[i] = generate_uniform(); //sample initial position
        XT[i] = integrate_gbm(X0[i],10,dt); //integrates the auxillary sde process
        //printf("%f\n",XT[i]);
    }   
}


double* make_weights_matrix(int neurons_in_var, int neurons_out_var)
{
    srand(time(NULL)); // seed with currtime

    double lim_glorot = sqrt(6 / ((double) (neurons_in_var + neurons_out_var)));  //compute initialisation scale
    
    double  ** W=( double * * ) malloc ( sizeof ( double * ) * neurons_in_var );     //allocate neurons_in_var rows 
    
    for (int i=0; i<neurons_out_var; i++)
        {
            W[i] = (int *)malloc(sizeof(double)*neurons_out_var); //allocate neurons_out_var cols per row
            for (int j=0; j<neurons_out_var; j++)
        {
            W[i][j] = generate_uniform_shifted(lim_glorot);  //sample initial weights 
            }
        }
    return W;
}

double* forward_pass(int neurons_in_var, int neurons_out_var, double (*X0)[neurons_in_var], double weights[neurons_out_var][neurons_in_var], double bias[neurons_out_var])
{
    double  ** xout =( double * * ) malloc ( sizeof ( double * ) * batch );     //allocate batch rows 
    
    for (int bi=0; bi<batch; bi++)
            {
                xout[bi] = (int *)malloc(sizeof(double)*neurons_out_var); //allocate neurons_out_var cols per row
        
        for (int i=0; i<neurons_out_var; i++)
            {
            for (int j=0; j<neurons_in_var; j++)
        {
            
            xout[bi][i] += weights[i][j] * X0[bi][j]; //sum up the weights 
            }
        
            xout[bi][i] += bias[i]; //add the bias
            xout[bi][i] = tanh(xout[bi][i]); //apply an activation
            }
    }
    return xout;
}

double mse_loss(double *XT, double *X0)
{
    // computes mean squared error between input and output
    
    double temp = 0;
    for (int i=1; i<batch; i++)
    {   
        temp += (double) (pow(XT[i] - X0[i],2)) ;

    }
    {return sqrt(temp/batch);}
}



//driver
void main()
{
    double X0[batch][1] = {{0}} ; //initial 
    double XT[batch] = {0} ; //final

    double bias[1] ={0} ; //layer bias
    double* W1 = make_weights_matrix(1,2);

    generate_training_data(X0, XT);
    
    double* xout = forward_pass(1, 2, X0, W1, bias);
    printf("Loss %f",xout);

    //double* xin = forward_pass(X0,W1,bias);

    //double loss_temp = mse_loss(XT,xin);
    
    //printf("Loss %f",loss_temp);
    
}
