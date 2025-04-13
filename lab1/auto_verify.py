import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

def verify_single_matrix(dir_path):
    """Проверяет одну директорию с матрицами"""
    try:
        size = int(dir_path.name)
        m1_file = dir_path / "matrix1.txt"
        m2_file = dir_path / "matrix2.txt"
        res_file = dir_path / "result.txt"

        if not m1_file.exists() or not m2_file.exists() or not res_file.exists():
            print(f"Files missing in {dir_path}")
            return False, None

        m1 = np.loadtxt(m1_file)
        m2 = np.loadtxt(m2_file)
        cpp_res = np.loadtxt(res_file)

        np_res = np.matmul(m1, m2)

        if np.array_equal(cpp_res, np_res):
            print(f"{size}x{size}: Verification PASSED")
            return True, size
        else:
            diff = np.abs(cpp_res - np_res)
            print(f"{size}x{size}: Verification FAILED")
            print(f"    Max diff: {np.max(diff):.2f}, Avg diff: {np.mean(diff):.2f}")
            return False, size

    except Exception as e:
        print(f"{dir_path}: Error - {str(e)}")
        return False, None


def parse_time_from(parse_file):
    """Извлекает время выполнения из файла"""
    times = {}
    with open(parse_file) as f:
        for line in f:
            if "Обработка матриц" in line:
                current_size = int(line.split()[2].split('x')[0])
            elif "Время выполнения:" in line:
                time_ms = int(line.split()[2])
                times[current_size] = time_ms
    return times


def plot_performance(times, output_file="performance_plot.png"):
    """Строит график производительности"""
    sizes = sorted(times.keys())
    times_ms = [times[size] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_ms, 'o-', label='Время выполнения')


    plt.title('Зависимость времени умножения матриц от их размера')
    plt.xlabel('Размер матрицы (N x N)')
    plt.ylabel('Время (мс)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    print(f"\nГрафик сохранён как {output_file}")


def find_and_verify_all(base_dir="matrix_results"):
    """Находит и проверяет все результаты автоматически"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found!")
        return False

    print(f"\n🔍 Starting automatic verification in {base_path}")

    total = 0
    passed = 0
    sizes = []

    # Ищем все поддиректории с числовыми именами
    for dir_path in base_path.iterdir():
        if dir_path.is_dir() and re.match(r'^\d+$', dir_path.name):
            total += 1
            result, size = verify_single_matrix(dir_path)
            if result:
                passed += 1
            if size:
                sizes.append(size)

    # Парсим лог и строим график
    parse_file = Path("stats.txt")
    if parse_file.exists():
        times = parse_time_from(parse_file)
        plot_performance(times)
    else:
        print("файл не найден, график не построен")

    print(f"\nVerification summary:")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    return passed == total


if __name__ == "__main__":

    base_dir = sys.argv[1] if len(sys.argv) > 1 else "matrix_results"

    success = find_and_verify_all(base_dir)
    exit(0 if success else 1)