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

// Создание директории
void createDirectory(const fs::path& path) {
    if (!fs::exists(path)) {
        try {
            if (fs::create_directories(path)) {
                cout << "Папка создана: " << path << endl;
            }
            else {
                cout << "Папка уже существует: " << path << endl;
            }
        }
        catch (const fs::filesystem_error& e) {
            cerr << "Ошибка при создании папки: " << e.what() << endl;
        }
    }
    else {
        cout << "Папка уже существует: " << path << endl;
    }
}

// Генерация матрицы
void generateMatrix(const fs::path& filepath, int size) {
    ofstream file(filepath);
    if (!file) {
        cerr << "ОШИБКА: Не удалось открыть файл " << filepath << endl;
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
        cerr << "ОШИБКА: Не удалось открыть файл " << filepath << endl;
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

// Ядро CUDA для умножения матриц
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

// Умножение матриц с использованием CUDA
vector<vector<int>> multiplyMatricesCUDA(const vector<vector<int>>& a, const vector<vector<int>>& b, int threadsPerBlock) {
    int size = a.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

    // Выделяем память на хосте
    int* h_a = new int[size * size];
    int* h_b = new int[size * size];
    int* h_c = new int[size * size];

    // Преобразуем матрицы в одномерные массивы
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            h_a[i * size + j] = a[i][j];
            h_b[i * size + j] = b[i][j];
        }
    }

    // Выделяем память на устройстве
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size * size * sizeof(int));
    cudaMalloc(&d_b, size * size * sizeof(int));
    cudaMalloc(&d_c, size * size * sizeof(int));

    // Копируем данные на устройство
    cudaMemcpy(d_a, h_a, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // Настроим параметры блока
    dim3 threadsPerBlockDim(threadsPerBlock, threadsPerBlock);
    dim3 blocksPerGrid((size + threadsPerBlock - 1) / threadsPerBlock, (size + threadsPerBlock - 1) / threadsPerBlock);

    // Запускаем ядро
    matrixMultiplyKernel << <blocksPerGrid, threadsPerBlockDim >> > (d_a, d_b, d_c, size);

    // Копируем результат обратно на хост
    cudaMemcpy(h_c, d_c, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    // Преобразуем результат обратно в двумерный вектор
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[i][j] = h_c[i * size + j];
        }
    }

    // Освобождаем память
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return result;
}

// Запись результата
void writeResult(const fs::path& filepath, const vector<vector<int>>& matrix) {
    ofstream file(filepath);
    if (!file) {
        cerr << "ОШИБКА: Не удалось открыть файл " << filepath << endl;
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

// Запись времени в файл
void writeTimingResults(const fs::path& filepath, int size, int threads, long duration) {
    ofstream file(filepath, ios::app);
    if (!file) {
        cerr << "ОШИБКА: Не удалось открыть файл для записи времени " << filepath << endl;
        return;
    }
    file << "Размер матрицы: " << size << "x" << size << ", Потоков: " << threads
        << ", Время: " << duration << " мс" << endl;
    file.close();
}

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);
    setlocale(LC_ALL, "Russian");
    srand(time(0));

    // Проверяем доступность CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cerr << "ОШИБКА: Не найдены устройства CUDA" << endl;
        return 1;
    }
    cout << "Найдено устройств CUDA: " << deviceCount << endl;

    // Базовый путь для сохранения файлов
    fs::path base_path = "D:/Code/lessons/year_3/semak_2/paralel_lab/lab_4/matrix_results";
    createDirectory(base_path);

    // Файл для записи времени
    fs::path timing_file = base_path / "stats.txt";
    ofstream timing_file_stream(timing_file);
    timing_file_stream.close(); // Очистка файла перед записью новых данных

    vector<int> thread_counts = { 1, 2, 4, 8 }; // Количество потоков в блоке (CUDA)

    // Тестируем для размеров от 100 до 1000 с шагом 100
    for (int size = 100; size <= 1000; size += 100) {
        // Создаем папку для текущего размера
        fs::path size_dir = base_path / to_string(size);
        createDirectory(size_dir);

        // Генерируем пути к файлам
        fs::path matrix1_path = size_dir / "matrix1.txt";
        fs::path matrix2_path = size_dir / "matrix2.txt";
        fs::path result_path = size_dir / "result.txt";

        cout << "\nОбработка матриц " << size << "x" << size << "..." << endl;

        // 1. Генерация матриц
        cout << "Генерация матриц..." << endl;
        generateMatrix(matrix1_path, size);
        generateMatrix(matrix2_path, size);

        // 2. Чтение матриц
        cout << "Чтение матриц..." << endl;
        auto matrix1 = readMatrix(matrix1_path);
        auto matrix2 = readMatrix(matrix2_path);

        if (matrix1.empty() || matrix2.empty()) {
            cerr << "ОШИБКА: Не удалось прочитать матрицы" << endl;
            continue;
        }

        // Перебор количества потоков
        for (int threads : thread_counts) {
            // 3. Перемножение с замером времени (CUDA)
            cout << "Умножение матриц (CUDA) с " << threads << " потоками..." << endl;
            auto start = chrono::high_resolution_clock::now();
            auto result = multiplyMatricesCUDA(matrix1, matrix2, threads);
            auto end = chrono::high_resolution_clock::now();

            // 4. Запись результата
            cout << "Запись результата..." << endl;
            writeResult(result_path, result);

            // Сохраняем результат
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            writeTimingResults(timing_file, size, threads, duration.count());

            cout << "Время выполнения: " << duration.count() << " мс" << endl;
        }
    }

    return 0;
}
