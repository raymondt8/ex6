#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "poissonFunctions.h"

/*Global variables*/
int rank,		/*MPI rank*/
    grid_size,		/*Number of MPI processes*/
    dims[2]={0,0},	/*Dimention of MPI grid*/
    coords[2],		/*Coordinate of the rank in MPI grid*/
    periods[2] = {0,0}; /*Periodicity of grid*/

int main(int argc, char **argv)
{
 init_mpi(argc, argv,rank,grid_size,dims,coords,periods);
 if(grid_size==0) grid_size =1;/*For debugging and with only 1 processor*/ 
 if( argc < 2 ) {
    printf("need a problem size\n");
    return 1;
  }
  
  double pi, h, l_umax;
  int n, m, nn, i, j, cols, l_diag_rows; 
  n = atoi(argv[1]);
  m = n-1; 
  nn = 4*n; 
  
  struct Array *len	= newArray(grid_size);
  struct Array *displ	= newArray(grid_size);  
  
  splitVector(m,grid_size,) 
  cols 	= m/grid_size;	/*number of coulums of b to each processor*/
  l_diag_rows =m/grid_size; 
  struct Array *diag = newArray(m,1);
  struct Array *l_diag = newArray(l_diag_rows,1);
  struct Array *b 	= newArray(m,cols);
  struct Array *bt	= newArray(m,cols);
  struct Array *z	= newArray(nn,1); 

  createDatatype(b,&grid_size);

  h 	= 1./(double)n;
  pi	= 4.*atan(1.);

  /*struct timeval start, end;
  gettimeofday(&start,NULL);
*/
  for(i=0;i<diag->rows;i++){
    diag->data[i] = 2.*(1.-cos((i+1)*pi/(double)n));
} 

/*
 #pragma omp for schedule(static)
  for(i=0+rank*l_diag->rows;i<((rank+1)*l_diag->rows);i++){
    *(l_diag->data+i) = 2.*(1.-cos((i+1)*pi/(double)n));
  }
  MPI_Allgather(l_diag->data,l_diag->rows,MPI_DOUBLE,diag->data,diag->size,MPI_DOUBLE,cart_comm);
*/
  #pragma omp parallel for schedule(static) 
  for(i=0;i<m;i++){
    b->data[i] = h*h;
  }
  
  #pragma omp parallel for schedule(static)
  for(i=0;i<b->cols;i++){
    fst_(b->data+b->rows*i, &b->rows, z->data,&nn); 
  }
  /*Transpose b by sending/receiving rows of the columns in each local b; eg (l_b_0 = {1,2,a,b})+(l_b_1 = {3,4,c,d}) -> b_all = {1,a,3,c,2,b,4,d}*/
  MPI_Alltoall(b->data,grid_size,transpose_select_t,bt->data,1,transpose_insert_t,cart_comm);

  #pragma omp parallel for schedule(static)
  for(i=0;i<m;i++){
    fstinv_(bt->data+i*bt->rows,&bt->rows,z->data,&nn);
  }
  #pragma omp parallel for schedule(static) private(i)
  for(j=0;j<m;j++){
    for(i=0;i<cols;i++){
      *(bt->data + j*bt->rows + i)=*(bt->data + j*bt->rows + i)/(*(diag->data+i)+*(diag->data+j));
    }
  }
  #pragma omp parallel for schedule(static)
  for(i=0;i<cols;i++){
    fst_(bt->data+i*b->rows, &bt->rows, z->data,&nn); 
  }

  MPI_Alltoall(bt->data,grid_size,transpose_select_t,b->data,1,transpose_insert_t,cart_comm);
  
  #pragma omp parallel for schedule(static)
  for(i=0;i<m;i++){
    fstinv_(b->data+i*b->rows,&b->rows,z->data,&nn);
  }
 
  l_umax = 0.0;
  
  #pragma omp parallel for schedule(static) reduction(max:l_umax)
  for(i=0;i<b->size;i++){
    if(l_umax<*(b->data +i)) l_umax=*(b->data+i);
  }
  if(rank==0){
    double umax = 0.0;
    struct Array *umaxArray = newArray(grid_size,1);
    MPI_Gather(&l_umax,1,MPI_DOUBLE,umaxArray->data,grid_size,MPI_DOUBLE,0,cart_comm);
    #pragma omp parallel for schedule(static) reduction(max:umax)
    for(i=0;i<grid_size;i++){
       if(umax<*(umaxArray->data +i)) umax=*(umaxArray->data+i);
    } 
     printf (" umax = %e \n",umax);
     /*gettimeofday(&end,NULL);
     print_time(start,end);
     */
  }

  freeArray(b);
  freeArray(bt);
  freeArray(l_diag);
  freeArray(diag);
  freeArray(z);
  freeDatatype();


  MPI_Finalize();

 return 0;
}
