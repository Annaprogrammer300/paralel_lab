#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

// Создание директории
void createDirectory(const fs::path& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

// Генерация матрицы
void generateMatrix(const fs::path& filepath, int size) {
    ofstream file(filepath);
    if (!file) {
        cerr << "ERROR: Failed to open file " << filepath << endl;
        return;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << rand() % 100 << " ";
        }
        file << "\n";
    }
    file.close();
}

// Чтение матрицы из файла
vector<vector<int>> readMatrix(const fs::path& filepath) {
    ifstream file(filepath);
    if (!file) {
        cerr << "ERROR: Failed to open file " << filepath << endl;
        return {};
    }

    vector<vector<int>> matrix;
    string line;
    while (getline(file, line)) {
        vector<int> row;
        istringstream iss(line);
        int num;
        while (iss >> num) {
            row.push_back(num);
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

// Запись результата
void writeResult(const fs::path& filepath, const vector<vector<int>>& matrix) {
    ofstream file(filepath);
    if (!file) {
        cerr << "ERROR: Failed to open file " << filepath << endl;
        return;
    }

    for (const auto& row : matrix) {
        for (int val : row) {
            file << val << " ";
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(time(0) + world_rank);

    fs::path base_path = "C:/Users/user/Desktop/paralel_lab/lab3/matrix_results_mpi";

    if (world_rank == 0 && world_size == 1) {
        createDirectory(base_path);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    vector<int> sizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

    for (int size : sizes) {
        fs::path size_dir = base_path / to_string(size);

        if (world_rank == 0 && world_size == 1) {
            createDirectory(size_dir);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        fs::path matrix1_path = size_dir / "matrix1.txt";
        fs::path matrix2_path = size_dir / "matrix2.txt";
        fs::path result_path = size_dir / "result.txt";

        vector<vector<int>> A, B;

        if (world_rank == 0) {
            cout << "\nMatrix size: " << size << "x" << size << endl;

            if (world_size == 1) {
                cout << "Generating matrices..." << endl;
                generateMatrix(matrix1_path, size);
                generateMatrix(matrix2_path, size);

                cout << "Reading matrices..." << endl;
                A = readMatrix(matrix1_path);
                B = readMatrix(matrix2_path);
            }
            else {
                // Если процессов > 1, просто генерируем случайные матрицы
                A.resize(size, vector<int>(size));
                B.resize(size, vector<int>(size));
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        A[i][j] = rand() % 100;
                        B[i][j] = rand() % 100;
                    }
                }
            }
        }

        // Передаём размер всем процессам
        MPI_Barrier(MPI_COMM_WORLD);

        // Рассылаем матрицу B всем процессам
        vector<int> B_flat;
        if (world_rank == 0) {
            for (const auto& row : B) {
                B_flat.insert(B_flat.end(), row.begin(), row.end());
            }
        }

        if (world_rank != 0) {
            B_flat.resize(size * size);
        }
        MPI_Bcast(B_flat.data(), size * size, MPI_INT, 0, MPI_COMM_WORLD);

        B.resize(size, vector<int>(size));
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                B[i][j] = B_flat[i * size + j];
            }
        }

        int rows_per_proc = size / world_size;
        int extra_rows = size % world_size;

        int my_rows = rows_per_proc + (world_rank < extra_rows ? 1 : 0);
        int my_offset = world_rank * rows_per_proc + min(world_rank, extra_rows);

        vector<int> A_local(my_rows * size);

        if (world_rank == 0) {
            for (int p = 0; p < world_size; p++) {
                int proc_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
                int proc_offset = p * rows_per_proc + min(p, extra_rows);
                if (p == 0) {
                    for (int i = 0; i < proc_rows; i++) {
                        for (int j = 0; j < size; j++) {
                            A_local[i * size + j] = A[proc_offset + i][j];
                        }
                    }
                }
                else {
                    vector<int> temp(proc_rows * size);
                    for (int i = 0; i < proc_rows; i++) {
                        for (int j = 0; j < size; j++) {
                            temp[i * size + j] = A[proc_offset + i][j];
                        }
                    }
                    MPI_Send(temp.data(), proc_rows * size, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            MPI_Recv(A_local.data(), my_rows * size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto start_time = MPI_Wtime();

        vector<int> result_local(my_rows * size, 0);

        for (int i = 0; i < my_rows; i++) {
            for (int j = 0; j < size; j++) {
                int sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += A_local[i * size + k] * B[k][j];
                }
                result_local[i * size + j] = sum;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto end_time = MPI_Wtime();

        double elapsed_time = end_time - start_time;

        // Сбор результатов
        vector<vector<int>> result(size, vector<int>(size, 0));
        if (world_rank == 0) {
            for (int i = 0; i < my_rows; i++) {
                for (int j = 0; j < size; j++) {
                    result[i][j] = result_local[i * size + j];
                }
            }
            for (int p = 1; p < world_size; p++) {
                int proc_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
                int proc_offset = p * rows_per_proc + min(p, extra_rows);

                vector<int> temp(proc_rows * size);
                MPI_Recv(temp.data(), proc_rows * size, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < proc_rows; i++) {
                    for (int j = 0; j < size; j++) {
                        result[proc_offset + i][j] = temp[i * size + j];
                    }
                }
            }

            cout << "Elapsed time: " << elapsed_time * 1000 << " ms" << endl;

            if (world_size == 1) {
                cout << "Saving result..." << endl;
                writeResult(result_path, result);
            }

            // Записываем время в общий файл
            fs::path global_time_path = base_path / "all_execution_times.txt";
            ofstream global_time_file(global_time_path, ios::app);
            if (global_time_file.is_open()) {
                global_time_file << "Matrix size: " << size << "x" << size
                    << ", Processes: " << world_size
                    << ", Time: " << elapsed_time * 1000 << " ms" << endl;
                global_time_file.close();
            }
        }
        else {
            MPI_Send(result_local.data(), my_rows * size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
