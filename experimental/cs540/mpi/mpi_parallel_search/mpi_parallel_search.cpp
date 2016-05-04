#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

const std::string FILE_NAME = "random_numbers.txt";
const std::string SMALL_FILE_NAME = "random_numbers_small.txt";
const int TARGET_INDICES_SEND = 1;
const int ROOT = 0;

// Uses c++11. Need to include flag -std=c++11 when compiling.
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int process_count, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int items_per_process = -1;
  int target_value = 0;
  std::vector<int> numbers;
  // Have root read in the file.
  if (rank == ROOT) {
    std::string file_name = FILE_NAME;
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

  // Send the target_value and items_per_process to every processor.
  MPI_Bcast(&target_value, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(&items_per_process, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

  if (items_per_process < 0) {
    if (rank == ROOT) {
      std::cout << "FAILURE EXIT" << std::endl;
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  int *local_numbers = new int[items_per_process];

  // Send each processor a part of the array.
  MPI_Scatter(&numbers.front(), items_per_process, MPI_INT, local_numbers,
              items_per_process, MPI_INT, ROOT, MPI_COMM_WORLD);

  // Search for the target_value.
  std::vector<int> local_target_indices;
  for (size_t i = 0; i < items_per_process; i++) {
    if (local_numbers[i] == target_value) {
      local_target_indices.push_back(rank * items_per_process + i);
    }
  }

  // Consolidate the results.
  if (rank == ROOT) {
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
    delete[] process_targets;

    for (int index : target_indices) {
      std::cout << index << std::endl;
    }
  } else {
    MPI_Ssend(&local_target_indices.front(), local_target_indices.size(),
              MPI_INT, ROOT, TARGET_INDICES_SEND, MPI_COMM_WORLD);
  }

  delete[] local_numbers;
  MPI_Finalize();
  return 0;
}
