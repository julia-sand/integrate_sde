#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
/*Integrating a Geometric Brownian motion in C*/

/*problem variables*/
#define mu 1
#define sigma

double generate_uniform()
/*returns a U(0,1) random variable*/
{
    return (double)rand() / (double)RAND_MAX;
}


double nrand(double u, double v)
/*takes two uniform random variables and generates a normal random variable using the box muller transform*/
{
    return (double)sqrt(-2*log(u))*cos(4*acos(0)*v); 
}


double integrate_gbm(double x0, int tsteps, double dt)
{
   
    double xs[tsteps];
    xs[0] = x0;
    for (int i=1; i<tsteps; i++)
    {   
        xs[i] = xs[i-1] + mu*xs[i-1]*dt + sqrt(2*dt*sigma)*nrand(generate_uniform(),generate_uniform());
    }
    return xs[tsteps-1];
}

int main()
{
    srand(time(NULL)); // seed with currtime
    
    int batch = 20; //batch size 
    double dt = 0.01;
    
    double XSDE[batch] 
    double temp = integrate_gbm(0,1,1,3,dt);
    
    printf("%f", temp);
}
