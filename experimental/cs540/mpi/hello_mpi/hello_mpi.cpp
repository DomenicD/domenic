// hello_mpi.cpp : Defines the entry point for the console application.
//

#include <iomanip>
#include <iostream>
#include <mpi.h>

void basic_send(int size, int rank) {
  if (rank == 0) {
    int value = 42;
    for (size_t i = 1; i < size; i++) {
      MPI_Request request;
      std::cout << "Ready to send " << rank << "-->" << i << std::endl;
      MPI_Isend(&value, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
      std::cout << "Data send     " << rank << "-->" << i << std::endl;
    }
  } else {
    int value = -1;
    MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::cout << rank << " received from 0 the value " << value << std::endl;
  }
}

void broadcast_send(int rank) {
  int value = rank == 0 ? 42 : -1;
  MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    std::cout << "Rank " << rank << " received from 0 the value " << value
              << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Rank " << rank << " is done working" << std::endl;
}

void scatter_reduce(int size, int rank) {
  const int itemsPerProcess = 10;
  const int count = size * itemsPerProcess;
  int *data = new int[count];

  if (rank == 0) {
    for (size_t i = 0; i < count; i++) {
      data[i] = i + 1;
    }
  }

  int *localData = new int[itemsPerProcess];

  MPI_Scatter(data, itemsPerProcess, MPI_INT, localData, itemsPerProcess,
              MPI_INT, 0, MPI_COMM_WORLD);

  int localSum = 0;
  for (size_t i = 0; i < itemsPerProcess; i++) {
    localSum += localData[i];
  }

  int globalSum;
  MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Total sum = " << globalSum << std::endl;
  }

  delete[] data;
  delete[] localData;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // basic_send(size, rank);
  // broadcast_send(rank);
  scatter_reduce(size, rank);

  MPI_Finalize();
  return 0;
}
