#include <stdio.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

/*problem variables*/
#define mu 1
#define sigma 1 
#define neurons_in 8
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
        X0[i] = generate_uniform();
        XT[i] = integrate_gbm(X0[i],10,dt);
        //printf("%f\n",XT[i]);
    }   
}

void initialise_network_weights(double (*W1)[neurons_out])
{
    srand(time(NULL)); // seed with currtime

    double lim_glorot = sqrt(6 / ((double) (neurons_in + neurons_out)));  //compute initialisation scale

   for (int i=0; i<neurons_in; i++) //use glorot uniform, ie sample initial weights from uniform
    {
        for (int j=0; j<neurons_out; j++)
            {
            W1[i][j] = generate_uniform_shifted(lim_glorot);
    }}
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
    double X0[batch] = {0} ; //initial 
    double XT[batch] = {0} ; //final

    double W1[neurons_in][neurons_out] ={{0}} ; //layer weights

    double bias[neurons_in] ; //layer bias

    initialise_network_weights(W1);
    
    generate_training_data(X0, XT);

    double loss_temp = mse_loss(XT,X0);
    
    printf("Loss %f",loss_temp);
    
}
