#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <assert.h>
#include "poissonFunctions.h"

void init_mpi(int argc,char** argv, int *rank, int *mpiSize)
{
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, mpiSize);
   MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

struct Array *newArray(int rows, int cols)
{
  /*For this application, the matrix is stored columnwise sutch that the array consists of the n'th column followed by the (n+1)'th column in the memory. This is beause of the extended use of columns in the methods applyed in this poisson-solver*/ 
  struct Array *array=malloc(sizeof(struct Array));
  assert(array!=NULL);
  array->rows = rows;
  array->cols = cols; 
  array->size = rows*cols;
  array->data = (double*)malloc(rows*cols*sizeof(double)); 
  int i; /* must use rows in place of i since c90 forbids mixed declaration ans code */
  for(i=0; i<array->size;i++){
     array->data[i]=0.0;
  }
  return array;
}
void freeArray(struct Array* array)
{
  assert(array!=NULL);
  free(array->data);
  free(array);
}
void createDatatype(struct Array *array,int maxLen)
{
  MPI_Type_vector(maxLen*maxLen,1,array->rows,MPI_DOUBLE,&transpose_select_t);
  MPI_Type_commit(&transpose_select_t);
  
  MPI_Type_vector(maxLen,maxLen,array->rows,MPI_DOUBLE,&transpose_insert_t);
  MPI_Type_commit(&transpose_insert_t);
}
void freeDatatype(){
  MPI_Type_free(&transpose_select_t);
  MPI_Type_free(&transpose_insert_t);
}
/*splitvector is copied from common.c, small changes*/
void splitVector(int globLen, int size, int** len, int** displ,int* maxLen)
{
  *maxLen=0;
  int i;
  *len = calloc(size,sizeof(int));
  *displ = calloc(size,sizeof(int));
  for (i=0;i<size;++i) {
    (*len)[i] = globLen/size;
    if (globLen % size && i >= (size - globLen % size))
    (*len)[i]++;
    if (i < size-1)
    (*displ)[i+1] = (*displ)[i]+(*len)[i];
  }
  for(i=0;i<size;i++){
    if(maxLen < len[i]) maxLen = len[i];
  }
}
void foldMatrix(struct Array *array,struct Array *fold){
  int i,j,k=0;
  for(i=0;i<array->rows;i++){
    for(j=0;j<fold->cols;j++){
      if(j<array->cols){
        fold->data[k] = array->data[j*array->rows + i];
        k++;
      }else{
        k++;
    }
  }
}
}
void unfoldMatrix(struct Array *array, struct Array *unfold,int *len,int mpiSize,int maxLen,int rank){
  int i,p,j,l;
  i=0;
  for(l=0;l<len[rank];l++){
    for(p=0;p<mpiSize;p++){
      for(j=0;j<len[p];j++){
        array->data[i] = unfold->data[unfold->rows*l+p*maxLen+j];
        i+=1;
      }
    }
  }
}
