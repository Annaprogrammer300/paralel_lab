import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import glob


def parse_slurm_file(file_path):
    """Парсит один SLURM-файл с результатами"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.match(
                r'Matrix size: (\d+)x\d+, Processes: (\d+), Time: ([\d.]+) ms',
                line.strip()
            )
            if match:
                data.append({
                    'size': int(match.group(1)),
                    'processes': int(match.group(2)),
                    'time_sec': float(match.group(3)) / 1000  # Конвертируем в секунды
                })
    return pd.DataFrame(data)


def plot_execution_time(slurm_files):
    """Строит график времени выполнения"""
    plt.figure(figsize=(12, 8))

    # Собираем данные из всех файлов
    all_data = pd.DataFrame()
    for file in slurm_files:
        df = parse_slurm_file(file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data.empty:
        print("Нет данных для построения графиков!")
        return

    # Уникальные значения процессов
    process_counts = sorted(all_data['processes'].unique())

    # Цветовая схема
    colors = plt.cm.viridis(np.linspace(0, 1, len(process_counts)))

    # Строим график для каждого количества процессов
    for i, proc in enumerate(process_counts):
        proc_data = all_data[all_data['processes'] == proc]
        plt.plot(proc_data['size'], proc_data['time_sec'],
                 'o-', color=colors[i], linewidth=2.5,
                 markersize=8, label=f'{proc} процессов')

    # Настройки графика
    plt.title('Зависимость времени умножения матриц от их размера', fontsize=14, pad=20)
    plt.xlabel('Размер матрицы (N x N)', fontsize=12)
    plt.ylabel('Время выполнения (секунды)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Количество процессов', fontsize=10,
               title_fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('mpi_performance_plot.png', dpi=150)
    plt.close()
    print("График сохранен в mpi_execution_time.png")


def main():
    # Находим все SLURM-файлы
    slurm_files = glob.glob('super/slurm-*.txt')

    if not slurm_files:
        print("Не найдено SLURM-файлов (slurm-*.txt) в текущей директории!")
        return

    print(f"Найдено SLURM-файлов: {len(slurm_files)}")
    plot_execution_time(slurm_files)


if __name__ == "__main__":
    main()