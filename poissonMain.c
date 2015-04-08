#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "poissonFunctions.h"


int main(int argc, char **argv)
{
  int rank,		/*MPI rank*/
      mpiSize=1;	/*Number of MPI processes*/
  
 init_mpi(argc, argv,&rank,&mpiSize);

 if( argc < 2 ) {
    printf("need a problem size\n");
    MPI_Finalize();
    return 1;
  }
  
  double pi, h, l_umax;
  int n, m, nn, i, j, *len, *displ,maxLen, f;  
  n = atoi(argv[1]);
  m = n-1; 
  nn = 4*n; 
 
  splitVector(m,mpiSize,&len,&displ,&maxLen); 

  struct Array *diag	= newArray(m,1);
  struct Array *b 	= newArray(m,len[rank]);
  struct Array *bt	= newArray(m,len[rank]);
  struct Array *z	= newArray(nn,1); 
  struct Array *send	= newArray(maxLen*mpiSize,maxLen);
  struct Array *receive	= newArray(maxLen*mpiSize,maxLen);

  createDatatype(b,maxLen);

  h 	= 1./(double)n;
  pi	= 4.*atan(1.);
  f	= 1;
  /*struct timeval start, end;
  gettimeofday(&start,NULL);
*/
  for(i=0;i<diag->rows;i++){
    diag->data[i] = 2.*(1.-cos((i+1)*pi/(double)n));
  } 

  #pragma omp parallel for schedule(static) 
  for(i=0;i<b->size;i++){
    b->data[i] = h*h*f;
  }
  
  #pragma omp parallel for schedule(static)
  for(i=0;i<b->cols;i++){
    fst_(&(b->data[b->rows*i]), &b->rows, z->data,&nn);
  }

  foldMatrix(b, send);
 
  MPI_Alltoall(&send->data,1,transpose_select_t,&receive->data,1,transpose_insert_t,MPI_COMM_WORLD);
  
  unfoldMatrix(bt,receive,len,mpiSize,maxLen,rank);

  #pragma omp parallel for schedule(static)
  for(i=0;i<len[rank];i++){
    fstinv_(&bt->data[i*bt->rows],&bt->rows,z->data,&nn);
  }

  #pragma omp parallel for schedule(static) private(i)
  for(j=0;j<len[rank];j++){
    for(i=0;i<bt->rows;i++){
      bt->data[j*bt->rows+i]=bt->data[j*bt->rows+i]/(diag->data[i]+diag->data[j]);
    }
  }

  #pragma omp parallel for schedule(static)
  for(i=0;i<len[rank];i++){
    fst_(&bt->data[i*b->rows], &bt->rows, z->data,&nn); 
  }

  foldMatrix(bt, send);

  MPI_Alltoall(&send->data,1,transpose_select_t,&receive->data,1,transpose_insert_t,MPI_COMM_WORLD);
 
  unfoldMatrix(b,receive,len,mpiSize,maxLen,rank);
  
  #pragma omp parallel for schedule(static)
  for(i=0;i<len[rank];i++){
    fstinv_(&b->data[i*b->rows],&b->rows,z->data,&nn);
  }
 
  l_umax = 0.0;
  
  #pragma omp parallel for schedule(static) reduction(max:l_umax)
  for(i=0;i<b->size;i++){
    if(l_umax<b->data[i]) l_umax=b->data[i];
  }

  if(rank==0){
    double umax = 0.0;
    struct Array *umaxArray = newArray(mpiSize,1);
    MPI_Gather(&l_umax,1,MPI_DOUBLE,&umaxArray->data,mpiSize,MPI_DOUBLE,0,MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static) reduction(max:umax)
    for(i=0;i<umaxArray->rows;i++){
       if(umax<umaxArray->data[i]){
         umax=umaxArray->data[i];
       }
    } 

     printf (" umax = %e \n",umax);
     /*gettimeofday(&end,NULL);
     print_time(start,end);
     */
  }

  freeArray(b);
  freeArray(bt);
  freeArray(diag);
  freeArray(z);
  freeArray(send);
  freeArray(receive);
 
  free(len);
  free(displ);

  freeDatatype();

  MPI_Finalize();

 return 0;
}
