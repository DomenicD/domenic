#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>


int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);

	// This can be done elegently by using MPI_Scatter and MPI_Gather.
	// Use this for reference:
	// http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
	// Use modern c++ IO:
	// http://www.cplusplus.com/doc/tutorial/files/

	const int N = 300;
	int i, target;
	int b[N]; // the entire array of integers
    /** External file "b.data" has the target value on the first line.
	 *  The remaining 300 lines of b.data have the values for the b array.
	 */
	FILE* data;
	data = fopen("random_numbers.txt", "r");
	// File found.data will contain the indices of b where the target is.
	FILE* results;
	results = fopen("found.data", "w");
	// Read-in the target
	fscanf(data, "%d", &target);
	// Read-in b array 
	for (i = 0; i < N; i++) {
		fscanf(data, "%d", &b[i]);
	}
	// Search the b array and output the target locations
	for (i = 0; i < N; i++) {
		if (b[i] == target) {
			fprintf(results, "%d\n", i);
		}
	}
	fclose(data); fclose(results);

	return 0;
}

