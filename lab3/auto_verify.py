import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
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


def parse_time_from(log_file):
    """Извлекает время выполнения из файла stats.txt с поддержкой дробного времени"""
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Новый паттерн под твои логи
    pattern = r'Matrix size: (\d+)x\d+, Processes: (\d+), Time: ([\d\.]+) ms'

    data = []
    for line in content.split('\n'):
        match = re.search(pattern, line)
        if match:
            size = int(match.group(1))
            threads = int(match.group(2))
            time_ms = float(match.group(3))  # <-- здесь теперь float!
            data.append({
                'size': size,
                'threads': threads,
                'time_sec': time_ms / 1000  # переводим в секунды
            })

    return pd.DataFrame(data)


def plot_performance(df, output_file="performance_plot.png"):
    """Строит график производительности с улучшенным отображением времени"""
    plt.figure(figsize=(12, 8))

    # Уникальные значения потоков и размеров
    thread_counts = sorted(df['threads'].unique())
    sizes = sorted(df['size'].unique())

    # Цвета для каждого количества потоков
    colors = plt.cm.viridis(np.linspace(0, 1, len(thread_counts)))

    # Строим график для каждого количества потоков
    for i, threads in enumerate(thread_counts):
        thread_data = df[df['threads'] == threads]
        plt.plot(thread_data['size'], thread_data['time_sec'],
                 'o-', color=colors[i], linewidth=2.5,
                 markersize=8, label=f'{threads} процессов')

    # Настройки графика
    plt.title('Зависимость времени умножения матриц от их размера', fontsize=14, pad=20)
    plt.xlabel('Размер матрицы (N x N)', fontsize=12)
    plt.ylabel('Время выполнения (секунды)', fontsize=12)

    # Настройка оси Y для лучшего отображения времени
    max_time = df['time_sec'].max()
    y_ticks = np.arange(0, max_time + max_time / 10, max_time / 10)  # 10 делений на оси Y
    plt.yticks(y_ticks, [f"{t:.1f} с" for t in y_ticks])  # Добавляем единицы измерения

    # Форматирование оси X
    plt.xticks(sizes, [f"{s}" for s in sizes], rotation=45)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Количество процессов', fontsize=10,
               title_fontsize=11, framealpha=0.9)

    # Улучшаем читаемость
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def find_and_verify_all(base_dir="matrix_results_mpi"):
    """Находит и проверяет все результаты автоматически"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} not found!")
        return False

    print(f"\n Starting automatic verification in {base_path}")

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
    parse_file = Path("matrix_results_mpi/all_execution_times.txt")
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
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "matrix_results_mpi"

    success = find_and_verify_all(base_dir)
    exit(0 if success else 1)