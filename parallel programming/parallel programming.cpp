#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cassert>
#include <omp.h>  // Подключение OpenMP

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64; // Порог для переключения на стандартное умножение

// Функция для генерации случайной матрицы размера n x n (n = 2^k)
Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10; // случайные числа от 0 до 9
    return A;
}

// Функция для стандартного умножения двух матриц с использованием OpenMP
Matrix standardMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Функция для сложения матриц с использованием OpenMP
Matrix add(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

// Функция для вычитания матриц с использованием OpenMP
Matrix subtract(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Рекурсивное умножение матриц методом Штрассена с использованием OpenMP-задач
Matrix strassenMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();

    // Если размер матрицы меньше или равен порогу, используем стандартное умножение
    if (n <= STRASSEN_THRESHOLD) {
        return standardMultiply(A, B);
    }

    int newSize = n / 2;
    // Инициализируем подматрицы
    Matrix A11(newSize, vector<int>(newSize));
    Matrix A12(newSize, vector<int>(newSize));
    Matrix A21(newSize, vector<int>(newSize));
    Matrix A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize));
    Matrix B12(newSize, vector<int>(newSize));
    Matrix B21(newSize, vector<int>(newSize));
    Matrix B22(newSize, vector<int>(newSize));

    // Разбиваем матрицы A и B на 4 подматрицы
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // Вычисляем 7 промежуточных матриц по алгоритму Штрассена с параллельными задачами
    Matrix M1, M2, M3, M4, M5, M6, M7;

#pragma omp task shared(M1) firstprivate(A11, A22, B11, B22)
    {
        M1 = strassenMultiply(add(A11, A22), add(B11, B22));
    }
#pragma omp task shared(M2) firstprivate(A21, A22, B11)
    {
        M2 = strassenMultiply(add(A21, A22), B11);
    }
#pragma omp task shared(M3) firstprivate(A11, B12, B22)
    {
        M3 = strassenMultiply(A11, subtract(B12, B22));
    }
#pragma omp task shared(M4) firstprivate(A22, B21, B11)
    {
        M4 = strassenMultiply(A22, subtract(B21, B11));
    }
#pragma omp task shared(M5) firstprivate(A11, A12, B22)
    {
        M5 = strassenMultiply(add(A11, A12), B22);
    }
#pragma omp task shared(M6) firstprivate(A21, A11, B11, B12)
    {
        M6 = strassenMultiply(subtract(A21, A11), add(B11, B12));
    }
#pragma omp task shared(M7) firstprivate(A12, A22, B21, B22)
    {
        M7 = strassenMultiply(subtract(A12, A22), add(B21, B22));
    }

#pragma omp taskwait

    // Собираем итоговую матрицу из полученных результатов
    Matrix C11 = add(subtract(add(M1, M4), M5), M7);
    Matrix C12 = add(M3, M5);
    Matrix C21 = add(M2, M4);
    Matrix C22 = add(subtract(add(M1, M3), M2), M6);

    Matrix C(n, vector<int>(n, 0));
    // Объединяем четыре подматрицы в итоговую матрицу C
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }

    return C;
}

// Функция для валидации: проверяет равенство двух матриц
bool validateMatrices(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (B.size() != n)
        return false;
    for (int i = 0; i < n; i++) {
        if (A[i].size() != B[i].size())
            return false;
        for (int j = 0; j < n; j++) {
            if (A[i][j] != B[i][j])
                return false;
        }
    }
    return true;
}

// Функция для замера времени работы алгоритма
template<typename Func, typename... Args>
double measureTime(Func func, Args&&... args) {
    auto start = chrono::high_resolution_clock::now();
    func(forward<Args>(args)...);
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double, milli>(end - start).count();
}

int main() {
    // Устанавливаем размер матриц: n = 2^k, например k=7 -> n=128
    int k = 10; // можно изменить значение k
    int n = 1 << k;

    // Генерация случайных матриц
    Matrix A = generateRandomMatrix(n);
    Matrix B = generateRandomMatrix(n);

  

    // Умножение алгоритмом Штрассена с замером времени
    Matrix C_strassen;
    auto start_strassen = chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
    #pragma omp single nowait
        {
            C_strassen = strassenMultiply(A, B);
        }
    }
    auto end_strassen = chrono::high_resolution_clock::now();
    double time_strassen = chrono::duration<double, milli>(end_strassen - start_strassen).count();

    cout << "Strassen time: " << time_strassen << " ms\n";



    // Стандартное умножение с замером времени
    auto start_std = chrono::high_resolution_clock::now();
    Matrix C_std = standardMultiply(A, B);
    auto end_std = chrono::high_resolution_clock::now();
    double time_std = chrono::duration<double, milli>(end_std - start_std).count();


    // вывод результатов
    cout << "Matrix shape: " << n << "x" << n << "\n";
    cout << "Standard time: " << time_std << " ms\n";

    // Валидация: сравнение результатов
    bool valid = validateMatrices(C_std, C_strassen);
    cout << "Standard equals Strassen: " << (valid ? "yes" : "no") << "\n";

    return 0;
}
