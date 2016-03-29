// hello_mpi.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <iomanip>
#include <mpi.h>

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);



	MPI_Finalize();
	return 0;
}

