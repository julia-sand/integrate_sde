#include <stdio.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

/*problem variables*/
#define mu 1
#define sigma 1 
#define neurons_in 1 
#define neurons_hidden 3
#define neurons_out 1
#define batch 10

/*initialise layer weights and biases*/
double W1[neurons_in][neurons_hidden];
double W2[neurons_hidden][neurons_hidden];
double W3[neurons_hidden][neurons_out];
double bias1[neurons_hidden];
double bias2[neurons_hidden];
double bias3[neurons_out];

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
    }   
}


double* forward_pass(double X0[batch][neurons_in])
{
    double  ** xout =( double * * ) malloc ( sizeof ( double * ) * batch );     //allocate batch rows 
    
    /*pass through first layer*/
    for (int bi=0; bi<batch; bi++)
    {
                xout[bi] = (int *)malloc(sizeof(double)*neurons_hidden); //allocate neurons_hidden cols per row
        for (int i=0; i<neurons_in; i++)
            {
                for (int j=0; j<neurons_hidden; j++)
                {
                    xout[bi][i] += W1[i][j] * X0[bi][i]; //sum up the weights 
                    // printf("x0 %f\n",X0[bi][i]) ;
                    //printf("value %f\n",W1[i][j]) ;
                }
                xout[bi][i] += bias1[i]; //add the bias
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
    return sqrt(temp/batch);
}

//driver
void main()
{   
    double X0[batch][neurons_in] = {{0}} ; //initial 
    double XT[batch][neurons_out] = {{0}} ; //final
    
    double lim_glorot1 = sqrt(6 / ((double) (neurons_in + neurons_hidden)));  //compute initialisation scale
    double lim_glorot2 = sqrt(6 / ((double) (neurons_hidden + neurons_hidden)));  //compute initialisation scale
    double lim_glorot3 = sqrt(6 / ((double) (neurons_hidden + neurons_out)));  //compute initialisation scale

    /*Set initial weights for W1*/
    for (int i1=0; i1<neurons_in;i1++)
    {
        for (int j1=0; j1<neurons_hidden;j1++)
        {
            W1[i1][j1] = generate_uniform_shifted(lim_glorot1);
        }
    }
    /*Set initial weights for W2*/
    for (int i2=0; i2<neurons_hidden;i2++)
    {
        //set biases of first and second layer
        bias1[i2] = generate_uniform_shifted(lim_glorot1);
        bias2[i2] = generate_uniform_shifted(lim_glorot2);
        
        for (int j2=0; j2<neurons_hidden;j2++)
        {
            W2[i2][j2] = generate_uniform_shifted(lim_glorot2);
        }
    }
    /*Set initial weights for W3*/
    for (int i3=0; i3<neurons_hidden;i3++)
    {
        for (int j3=0; j3<neurons_out;j3++)
        {
            W3[i3][j3] = generate_uniform_shifted(lim_glorot3);
        }
    }
    /*Set final layer biases*/
    for (int j3=0; j3<neurons_out;j3++)
    {
        bias3[j3] = generate_uniform_shifted(lim_glorot3);
    }
    generate_training_data(X0, XT);
    double* xout = forward_pass(X0);
}
