#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Генерация случайной матрицы
void generateMatrix(vector<vector<int>>& matrix, int size) {
    matrix.resize(size, vector<int>(size));
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            matrix[i][j] = rand() % 100;
}

// Плоское представление матрицы
void flattenMatrix(const vector<vector<int>>& matrix, vector<int>& flat) {
    int size = matrix.size();
    flat.resize(size * size);
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            flat[i * size + j] = matrix[i][j];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(time(0) + world_rank);

    vector<int> sizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };

    for (int size : sizes) {
        vector<vector<int>> A, B;
        vector<int> B_flat;

        if (world_rank == 0) {
            generateMatrix(A, size);
            generateMatrix(B, size);
            flattenMatrix(B, B_flat);
        }

        // Рассылаем матрицу B
        if (world_rank != 0)
            B_flat.resize(size * size);

        MPI_Bcast(B_flat.data(), size * size, MPI_INT, 0, MPI_COMM_WORLD);

        // Восстановить B из плоского вида
        B.resize(size, vector<int>(size));
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                B[i][j] = B_flat[i * size + j];

        // Распределение строк матрицы A
        int rows_per_proc = size / world_size;
        int extra_rows = size % world_size;
        int my_rows = rows_per_proc + (world_rank < extra_rows ? 1 : 0);
        int my_offset = world_rank * rows_per_proc + min(world_rank, extra_rows);

        vector<int> A_local(my_rows * size);

        if (world_rank == 0) {
            for (int p = 0; p < world_size; ++p) {
                int proc_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
                int proc_offset = p * rows_per_proc + min(p, extra_rows);

                if (p == 0) {
                    for (int i = 0; i < proc_rows; ++i)
                        for (int j = 0; j < size; ++j)
                            A_local[i * size + j] = A[proc_offset + i][j];
                }
                else {
                    vector<int> temp(proc_rows * size);
                    for (int i = 0; i < proc_rows; ++i)
                        for (int j = 0; j < size; ++j)
                            temp[i * size + j] = A[proc_offset + i][j];
                    MPI_Send(temp.data(), proc_rows * size, MPI_INT, p, 0, MPI_COMM_WORLD);
                }
            }
        }
        else {
            MPI_Recv(A_local.data(), my_rows * size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // Локальное перемножение
        vector<int> result_local(my_rows * size, 0);
        for (int i = 0; i < my_rows; ++i) {
            for (int j = 0; j < size; ++j) {
                int sum = 0;
                for (int k = 0; k < size; ++k) {
                    sum += A_local[i * size + k] * B[k][j];
                }
                result_local[i * size + j] = sum;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        // Сбор результата
        if (world_rank == 0) {
            vector<int> result(size * size, 0);

            // Сохраняем свои результаты
            for (int i = 0; i < my_rows; ++i)
                for (int j = 0; j < size; ++j)
                    result[(my_offset + i) * size + j] = result_local[i * size + j];

            for (int p = 1; p < world_size; ++p) {
                int proc_rows = rows_per_proc + (p < extra_rows ? 1 : 0);
                int proc_offset = p * rows_per_proc + min(p, extra_rows);

                vector<int> temp(proc_rows * size);
                MPI_Recv(temp.data(), proc_rows * size, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                for (int i = 0; i < proc_rows; ++i)
                    for (int j = 0; j < size; ++j)
                        result[(proc_offset + i) * size + j] = temp[i * size + j];
            }

            cout << "Matrix size: " << size << "x" << size
                << ", Processes: " << world_size
                << ", Time: " << elapsed_time * 1000 << " ms" << endl;
        }
        else {
            MPI_Send(result_local.data(), my_rows * size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
