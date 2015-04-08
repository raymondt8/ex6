#ifndef POISSONFUNCTION_H
#define	POISSONFUNCTIONS_H

#include <stdlib.h>
#include <omp.h>
#include <mpi.h>

/*Cartesian communicator*/
MPI_Comm cart_comm; 

/*MPI datatype*/
MPI_Datatype	transpose_select_t,
		transpose_insert_t;
		
struct Array{
  int rows;
  int cols;
  int size;
  double *data;
};
struct Array *newArray(int rows,int cols);

void init_mpi(int argc, char **argv, int *rank,int *mpiSize);
void fst_(double *v, int *n, double *w, int *nn);
void fstinv_(double *v, int *n, double *w, int *nn);
void freeArray(struct Array* array);
void createDatatype(struct Array* array,int maxLen);
void freeDatatype();
void splitVector(int globLen, int size, int** len, int** displ,int* maxLen);
void foldMatrix(struct Array *array,struct Array *fold);
void unfoldMatrix(struct Array *array, struct Array *unfold,int *len,int mpiSize,int maxLen,int rank);
#endif
