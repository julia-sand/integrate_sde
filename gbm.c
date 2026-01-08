#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

/*problem variables*/
#define mu 1
#define sigma 1 

double generate_uniform()
/*returns a U(0,1) random variable*/
{
    return (double)rand() / (double)RAND_MAX;
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

void generate_training_data(int batch, double *X0, double *XT)
{
    srand(time(NULL)); // seed with currtime
    
    double dt = 0.01;
        
    for (int i=0; i<batch;i++)
    {
        X0[i] = generate_uniform();
        XT[i] = integrate_gbm(X0[i],10,dt);
        printf("%f\n",XT[i]);
    }
    
}

void main()
{
    int batch = 20;
    double X0[batch] ; //initial 
    double XT[batch] ; //final
    
    //pointer to 
    generate_training_data(batch, X0, XT);
    printf("%f\n",XT[batch-1]);
}
