#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <locale.h>
#include <windows.h>

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

// Умножение матриц
vector<vector<int>> multiplyMatrices(const vector<vector<int>>& a, const vector<vector<int>>& b) {
    int size = a.size();
    vector<vector<int>> result(size, vector<int>(size, 0));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
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



int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);
    setlocale(LC_ALL, "Russian");
    srand(time(0));

    // Базовый путь для сохранения файлов
    fs::path base_path = "C:/Users/user/Desktop/paralel_lab/lab1/matrix_results";
    createDirectory(base_path);

    vector<pair<int, long>> results;

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

        // 3. Перемножение с замером времени
        cout << "Умножение матриц..." << endl;
        auto start = chrono::high_resolution_clock::now();
        auto result = multiplyMatrices(matrix1, matrix2);
        auto end = chrono::high_resolution_clock::now();

        // 4. Запись результата
        cout << "Запись результата..." << endl;
        writeResult(result_path, result);

        // Сохраняем результат
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        results.emplace_back(size, duration.count());

        cout << "Время выполнения: " << duration.count() << " мс" << endl;
    }

    // Выводим красивые результаты
    //printResults(results);

    return 0;
}