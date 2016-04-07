#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

const int TARGET_INDICES_SEND = 1;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int process_count, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // This can be done elegently by using MPI_Scatter and MPI_Gather.
  // Use this for reference:
  // http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
  // Use modern c++ IO:
  // http://www.cplusplus.com/doc/tutorial/files/

  int items_per_process = -1;
  int target_value;
  std::vector<int> numbers;
  if (rank == 0) {
    std::string file_name = "random_numbers.txt";
    std::ifstream data_file(file_name, std::ios_base::in);

    if (!data_file.is_open()) {
      std::cout << "Unable to open file " << file_name << std::endl;
    } else {
      data_file >> target_value;

      int value;
      while (data_file >> value) {
        numbers.push_back(value);
      }
      items_per_process = numbers.size() / process_count;
    }
  }

  MPI_Bcast(&target_value, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&items_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (items_per_process < 0) {
    if (rank == 0) {
      std::cout << "FAILURE EXIT" << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  int *local_numbers = new int[items_per_process];

  MPI_Scatter(&numbers[0], items_per_process, MPI_INT, local_numbers,
              items_per_process, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_target_indices;
  for (size_t i = 0; i < items_per_process; i++) {
    if (local_numbers[i] == target_value) {
      local_target_indices.push_back(i * items_per_process + i);
    }
  }

  if (rank == 0) {
    std::vector<int> target_indices;
    target_indices.insert(target_indices.begin(), local_target_indices.begin(),
                          local_target_indices.end());
    std::vector<MPI_Status> target_send_statuses;

    int *process_targets = new int[items_per_process];
    for (size_t i = 1; i < process_count; i++) {
      MPI_Status status;
      int count;
      MPI_Probe(i, TARGET_INDICES_SEND, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_INT, &count);
      MPI_Recv(process_targets, count, MPI_INT, i, TARGET_INDICES_SEND,
               MPI_COMM_WORLD, &status);
      target_indices.insert(target_indices.begin() + target_indices.size(),
                            process_targets, process_targets + count);
    }
    // TODO(domenic): Need to print out the target_indices.
  } else {
    // TODO(domenic): Need to send messaged to the root node.
  }

  return 0;
}
