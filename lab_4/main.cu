#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <locale.h>
#include <windows.h>
#include <cuda_runtime.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

using namespace std;

// �������� ����������
void createDirectory(const fs::path& path) {
    if (!fs::exists(path)) {
        try {
            if (fs::create_directories(path)) {
                cout << "����� �������: " << path << endl;
            }
            else {
                cout << "����� ��� ����������: " << path << endl;
            }
        }
        catch (const fs::filesystem_error& e) {
            cerr << "������ ��� �������� �����: " << e.what() << endl;
        }
    }
    else {
        cout << "����� ��� ����������: " << path << endl;
    }
}

// ��������� �������
void generateMatrix(const fs::path& filepath, int size) {
    ofstream file(filepath);
    if (!file) {
        cerr << "������: �� ������� ������� ���� " << filepath << endl;
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

// ������ ������� �� �����
vector<vector<int>> readMatrix(const fs::path& filepath) {
    ifstream file(filepath);
    if (!file) {
        cerr << "������: �� ������� ������� ���� " << filepath << endl;
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

// ���� CUDA ��� ��������� ������
__global__ void matrixMultiplyKernel(int* a, int* b, int* c, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

// ��������� ������ � �������������� CUDA
vector<vector<int>> multiplyMatricesCUDA(const vector<vector<int>>& a, const vector<vector<int>>& b, int threadsPerBlock) {
    int size = a.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

    // �������� ������ �� �����
    int* h_a = new int[size * size];
    int* h_b = new int[size * size];
    int* h_c = new int[size * size];

    // ����������� ������� � ���������� �������
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_a[i * size + j] = a[i][j];
            h_b[i * size + j] = b[i][j];
        }
    }

    // �������� ������ �� ����������
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size * size * sizeof(int));
    cudaMalloc(&d_b, size * size * sizeof(int));
    cudaMalloc(&d_c, size * size * sizeof(int));

    // �������� ������ �� ����������
    cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // �������� ��������� �����
    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGrid((size + threadsPerBlock - 1) / threadsPerBlock, (size + threadsPerBlock - 1) / threadsPerBlock);

    // ��������� ����
    matrixMultiplyKernel << <blocksPerGrid, threadsPerBlockDim >> > (d_a, d_b, d_c, size);

    // �������� ��������� ������� �� ����
    cudaMemcpy(h_c, d_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    // ����������� ��������� ������� � ��������� ������
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = h_c[i * size + j];
        }
    }

    // ����������� ������
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return result;
}

// ������ ����������
void writeResult(const fs::path& filepath, const vector<vector<int>>& matrix) {
    ofstream file(filepath);
    if (!file) {
        cerr << "������: �� ������� ������� ���� " << filepath << endl;
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

// ������ ������� � ����
void writeTimingResults(const fs::path& filepath, int size, int threads, long duration) {
    ofstream file(filepath, ios::app);
    if (!file) {
        cerr << "������: �� ������� ������� ���� ��� ������ ������� " << filepath << endl;
        return;
    }
    file << "������ �������: " << size << "x" << size << ", �������: " << threads
        << ", �����: " << duration << " ��" << endl;
    file.close();
}

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);
    setlocale(LC_ALL, "Russian");
    srand(time(0));

    // ��������� ����������� CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "������: �� ������� ���������� CUDA" << endl;
        return 1;
    }
    cout << "������� ��������� CUDA: " << deviceCount << endl;

    // ������� ���� ��� ���������� ������
    fs::path base_path = "D:/Code/lessons/year_3/semak_2/paralel_lab/lab_4/matrix_results";
    createDirectory(base_path);

    // ���� ��� ������ �������
    fs::path timing_file = base_path / "check.txt";
    ofstream timing_file_stream(timing_file);
    timing_file_stream.close(); // ������� ����� ����� ������� ����� ������

    vector<int> thread_counts = { 1, 2, 4, 8 }; // (CUDA)

    // ��������� ��� �������� �� 100 �� 1000 � ����� 100
    for (int size = 100; size <= 1000; size += 100) {
        // ������� ����� ��� �������� �������
        fs::path size_dir = base_path / to_string(size);
        createDirectory(size_dir);

        // ���������� ���� � ������
        fs::path matrix1_path = size_dir / "matrix1.txt";
        fs::path matrix2_path = size_dir / "matrix2.txt";
        fs::path result_path = size_dir / "result.txt";

        cout << "\n��������� ������ " << size << "x" << size << "..." << endl;

        // 1. ��������� ������
        cout << "��������� ������..." << endl;
        generateMatrix(matrix1_path, size);
        generateMatrix(matrix2_path, size);

        // 2. ������ ������
        cout << "������ ������..." << endl;
        auto matrix1 = readMatrix(matrix1_path);
        auto matrix2 = readMatrix(matrix2_path);

        if (matrix1.empty() || matrix2.empty()) {
            cerr << "������: �� ������� ��������� �������" << endl;
            continue;
        }

        // ������� ���������� �������
        for (int threads : thread_counts) {
            // 3. ������������ � ������� ������� (CUDA)
            cout << "��������� ������ (CUDA) � " << threads << " ��������..." << endl;
            auto start = chrono::high_resolution_clock::now();
            auto result = multiplyMatricesCUDA(matrix1, matrix2, threads);
            auto end = chrono::high_resolution_clock::now();

            // 4. ������ ����������
            cout << "������ ����������..." << endl;
            writeResult(result_path, result);

            // ��������� ���������
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            writeTimingResults(timing_file, size, threads, duration.count());

            cout << "����� ����������: " << duration.count() << " ��" << endl;
        }
    }

    return 0;
}
