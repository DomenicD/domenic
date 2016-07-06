#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <sstream>
#include <string>

const int DO_WORK_CMD = 1;
const int WORK_COMPLETE_CMD = 2;
const int ROOT = 0;
const int MAX_PROCESSOR_MEMORY = 1024;

std::string get_timestamp();

// Uses c++11. Need to include flag -std=c++11 when compiling.
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  if (argc != 2) {
    MPI_Finalize();
    std::cout << "This program takes one argument, which is the name of "
              << "the file to be sorted." << std::endl;
    exit(EXIT_FAILURE);
  }

  int process_pool, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &process_pool);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string file_name = argv[1];
  std::string temp_file_name = "temp_" + get_timestamp() + ".txt";
  MPI_File sorted_file;
  MPI_File temp_file;
  // We will do an "in place" sorting of the numbers in the file provided.
  MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_WRONLY,
                MPI_INFO_NULL, &sorted_file);
  // For the temp file we will create it, read and write from it, then have it
  // deleted when we are finished.
  MPI_File_open(MPI_COMM_WORLD, temp_file_name.c_str(),
                MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &temp_file);

  // Stage 1: Use OMP to sort chunks of the file.
  int work_complete = 0;
  MPI_Offset file_offset;
  MPI_Offset done_signal = -1;
  if (rank == ROOT) {
    MPI_Offset file_size;
    MPI_File_get_size(sorted_file, &file_size);
    MPI_Offset file_offset = 0;
    std::map<int, bool> idle_processors;
    for (int i = 1; i < process_pool; i++) {
      idle_processors[i] = true;
    }

    while (file_offset < file_size) {
      for (const auto kv : idle_processors) {
        if (kv.second) {
          MPI_Send(&file_offset, 1, MPI_OFFSET, kv.first, DO_WORK_CMD,
                   MPI_COMM_WORLD);
          file_offset += MAX_PROCESSOR_MEMORY;
        }
      }
      MPI_Status status;
      MPI_Recv(&work_complete, 1, MPI_INT, MPI_ANY_SOURCE, WORK_COMPLETE_CMD,
               MPI_COMM_WORLD, &status);
      idle_processors[status.MPI_TAG] = true;
    }
    // Wait for all the processors to finish their work and then tell them
    // that we are done with this stage.
    for (const auto kv : idle_processors) {
      if (!kv.second) {
        MPI_Recv(&work_complete, 1, MPI_INT, kv.first, WORK_COMPLETE_CMD,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        idle_processors[kv.first] = true;
      }
      MPI_Send(&done_signal, 1, MPI_OFFSET, kv.first, DO_WORK_CMD,
               MPI_COMM_WORLD);
    }
  } else {
    bool isDone = false;
    while (!isDone) {
      MPI_Recv(&file_offset, 1, MPI_OFFSET, ROOT, DO_WORK_CMD, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      // When no more work is to be done, file_offset will be to done_signal.
      if (file_offset == done_signal) {
        isDone = true;
      } else {
        // Do OMP merge sort on section of the file.
        MPI_Send(&work_complete, 1, MPI::BOOL, ROOT, WORK_COMPLETE_CMD,
                 MPI_COMM_WORLD);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Stage 2: Sort the parts of the file "in place" with some memory buffering.

  std::cout << file_name << std::endl;

  MPI_Finalize();
  getchar();
  return 0;
}

std::string get_timestamp() {
  auto timestamp = std::chrono::system_clock::now().time_since_epoch();
  std::stringstream ss;
  ss << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp)
            .count();
  return ss.str();
}