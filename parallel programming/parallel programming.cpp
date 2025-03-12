#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cassert>

using namespace std;
using Matrix = vector<vector<int>>;

// Функция для генерации случайной матрицы размера n x n (n = 2^k)
Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10; // случайные числа от 0 до 9
    return A;
}

// Функция для стандартного умножения двух матриц
Matrix standardMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Вспомогательные функции для сложения и вычитания матриц
Matrix add(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix subtract(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Функция для рекурсивного умножения матриц методом Штрассена
Matrix strassenMultiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));

    // Базовый случай: если размер 1x1, то обычное умножение
    if (n == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return C;
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

    // Вычисляем 7 промежуточных матриц по алгоритму Штрассена
    Matrix M1 = strassenMultiply(add(A11, A22), add(B11, B22));
    Matrix M2 = strassenMultiply(add(A21, A22), B11);
    Matrix M3 = strassenMultiply(A11, subtract(B12, B22));
    Matrix M4 = strassenMultiply(A22, subtract(B21, B11));
    Matrix M5 = strassenMultiply(add(A11, A12), B22);
    Matrix M6 = strassenMultiply(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassenMultiply(subtract(A12, A22), add(B21, B22));

    // Собираем итоговую матрицу из полученных результатов
    Matrix C11 = add(subtract(add(M1, M4), M5), M7);
    Matrix C12 = add(M3, M5);
    Matrix C21 = add(M2, M4);
    Matrix C22 = add(subtract(add(M1, M3), M2), M6);

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
    // Устанавливаем размер матриц: n = 2^k, например k=3 -> n=8
    int k = 10; // можно изменить значение k
    int n = 1 << k;

    // Генерация случайных матриц
    Matrix A = generateRandomMatrix(n);
    Matrix B = generateRandomMatrix(n);

    // Стандартное умножение с замером времени
    auto start_std = chrono::high_resolution_clock::now();
    Matrix C_std = standardMultiply(A, B);
    auto end_std = chrono::high_resolution_clock::now();
    double time_std = chrono::duration<double, milli>(end_std - start_std).count();

    // Умножение алгоритмом Штрассена с замером времени
    auto start_strassen = chrono::high_resolution_clock::now();
    Matrix C_strassen = strassenMultiply(A, B);
    auto end_strassen = chrono::high_resolution_clock::now();
    double time_strassen = chrono::duration<double, milli>(end_strassen - start_strassen).count();

    // Валидация: сравнение результатов
    bool valid = validateMatrices(C_std, C_strassen);

    cout << "Размер матрицы: " << n << "x" << n << "\n";
    cout << "Время стандартного умножения: " << time_std << " ms\n";
    cout << "Время умножения Штрассена: " << time_strassen << " ms\n";
    cout << "Результаты совпадают: " << (valid ? "Да" : "Нет") << "\n";

    return 0;
}
