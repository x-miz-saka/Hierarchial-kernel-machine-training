#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    int i, j, p, n, d, k;
    int *t, *T;
    double *aa, *bb, a, b;
    t = mxGetPr(prhs[0]);       /* row of integers  to look for */
    T = mxGetPr(prhs[1]);       /* rows of integers  to look i */
    p =  mxGetM(prhs[1]);
    n =  mxGetN(prhs[1]);
/*
    int number_of_elements_t = sizeof(t)/sizeof(int *);
    int number_of_elements_T = sizeof(T)/sizeof(int *);
    mexPrintf("Hello, world!%d,%d,%d,%d\n",n,p,number_of_elements_t,number_of_elements_T);
*/
    k = 0;
    a = p;
    b = 0;
    /*Docs Pratik
    n: number of nodes in source_cache
    p: number of base features
    t: target source node
    T: list of source nodes which already exists in the cache
    */
    int storeVal; /*Pratik*/
    for (i=0;i<n;i++) {
        d = 0;
        for (j=0;j<p;j++) {
            storeVal = T[k++];
/*
            if (t[j]!=storeVal)
                d++;
*/
	    /*Pratik: If t is not a child of current i, do not consider i*/
	    if (t[j] < storeVal)
		d = p+1;
	    else
		if (t[j] > storeVal)
		    d++;
		
            
            if (d >= a) { /* stop if more than before */
                k = k + p - 1 - j;
                j = p;
            }
        }
        if (d < a) {
            a = d;
            b = i+1;
        }
        
        if (a==0)   /* stop if exact match */
            i=n;
    }
    
    
    plhs[0]=mxCreateDoubleMatrix(1, 1, 0);
    aa = mxGetPr(plhs[0]);
    plhs[1]=mxCreateDoubleMatrix(1, 1, 0);
    bb = mxGetPr(plhs[1]);
    aa[0]=a;
    bb[0]=b;
    return;
    
}
