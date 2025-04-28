import subprocess
import sys
import os

def run_mpi_program(executable, num_processes, input_data=None):
    try:
        print(f"\nЗапуск {num_processes} процессов...")

        # Открываем процесс
        process = subprocess.Popen(
            ["mpiexec", "-n", str(num_processes), executable],
            stdin=subprocess.PIPE,
            stdout=sys.stdout,  # Пишем сразу в stdout
            stderr=sys.stderr,  # И ошибки сразу в stderr
            text=True,
            encoding="utf-8",
        )

        # Отправляем данные на ввод, если нужно
        if input_data:
            process.communicate(input=input_data)
        else:
            process.wait()

        if process.returncode != 0:
            print(f"Процесс завершился с ошибкой: код {process.returncode}")

    except FileNotFoundError:
        print(f"Ошибка: файл {executable} не найден!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    executable = "out/build/x64-Debug/MPI_Project.exe"
    processes_to_run = [1, 2, 4, 6, 8]
    input_data = "1000\n"  # Ввод размера матрицы автоматически

    if not os.path.exists(executable):
        print(f"Ошибка: исполняемый файл {executable} не существует!")
        sys.exit(1)

    for num_processes in processes_to_run:
        run_mpi_program(executable, num_processes, input_data=input_data)


