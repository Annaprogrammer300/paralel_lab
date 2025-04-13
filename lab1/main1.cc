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

// �������� ����������
void createDirectory(const fs::path& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
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

// ��������� ������
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



int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);
    setlocale(LC_ALL, "Russian");
    srand(time(0));

    // ������� ���� ��� ���������� ������
    fs::path base_path = "C:/Users/user/Desktop/paralel_lab/lab1/matrix_results";
    createDirectory(base_path);

    vector<pair<int, long>> results;

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

        // 3. ������������ � ������� �������
        cout << "��������� ������..." << endl;
        auto start = chrono::high_resolution_clock::now();
        auto result = multiplyMatrices(matrix1, matrix2);
        auto end = chrono::high_resolution_clock::now();

        // 4. ������ ����������
        cout << "������ ����������..." << endl;
        writeResult(result_path, result);

        // ��������� ���������
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        results.emplace_back(size, duration.count());

        cout << "����� ����������: " << duration.count() << " ��" << endl;
    }

    // ������� �������� ����������
    //printResults(results);

    return 0;
}