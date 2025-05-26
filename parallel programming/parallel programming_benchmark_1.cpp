#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <omp.h>  

using namespace std;
using Matrix = vector<vector<int>>;

const int STRASSEN_THRESHOLD = 64;

Matrix generateRandomMatrix(int n) {
    Matrix A(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = rand() % 10;
    return A;
}

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

Matrix strassenSequential(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD) return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
        A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
        B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

    for (int i = 0; i < newSize; i++)
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

    Matrix M1 = strassenSequential(add(A11, A22), add(B11, B22));
    Matrix M2 = strassenSequential(add(A21, A22), B11);
    Matrix M3 = strassenSequential(A11, subtract(B12, B22));
    Matrix M4 = strassenSequential(A22, subtract(B21, B11));
    Matrix M5 = strassenSequential(add(A11, A12), B22);
    Matrix M6 = strassenSequential(subtract(A21, A11), add(B11, B12));
    Matrix M7 = strassenSequential(subtract(A12, A22), add(B21, B22));

    Matrix C(n, vector<int>(n));
    for (int i = 0; i < newSize; i++)
        for (int j = 0; j < newSize; j++) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    return C;
}

Matrix strassenParallel(const Matrix& A, const Matrix& B) {
    int n = A.size();
    if (n <= STRASSEN_THRESHOLD) return standardMultiply(A, B);

    int newSize = n / 2;
    Matrix A11(newSize, vector<int>(newSize)), A12(newSize, vector<int>(newSize)),
        A21(newSize, vector<int>(newSize)), A22(newSize, vector<int>(newSize));
    Matrix B11(newSize, vector<int>(newSize)), B12(newSize, vector<int>(newSize)),
        B21(newSize, vector<int>(newSize)), B22(newSize, vector<int>(newSize));

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

    Matrix M1, M2, M3, M4, M5, M6, M7;

#pragma omp parallel sections
    {
#pragma omp section
        M1 = strassenParallel(add(A11, A22), add(B11, B22));

#pragma omp section
        M2 = strassenParallel(add(A21, A22), B11);

#pragma omp section
        M3 = strassenParallel(A11, subtract(B12, B22));

#pragma omp section
        M4 = strassenParallel(A22, subtract(B21, B11));

#pragma omp section
        M5 = strassenParallel(add(A11, A12), B22);

#pragma omp section
        M6 = strassenParallel(subtract(A21, A11), add(B11, B12));

#pragma omp section
        M7 = strassenParallel(subtract(A12, A22), add(B21, B22));
    }

    Matrix C(n, vector<int>(n));
#pragma omp parallel for collapse(2)
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }
    return C;
}

void benchmarkPowersOfTwo() {
    ofstream outFile("benchmark_powers_of_two.csv");
    outFile << "Size,Standard (ms),Sequential Strassen (ms),Parallel Strassen (ms)\n";

    vector<int> sizes;
    for (int i = 1; i <= 13; i++) {
        sizes.push_back(1 << i);  // 2, 4, 8, ..., 8192
    }

    for (int n : sizes) {
        cout << "Benchmarking " << n << "x" << n << "..." << endl;
        Matrix A = generateRandomMatrix(n);
        Matrix B = generateRandomMatrix(n);

        double time_std, time_seq, time_par;

        // Standard multiply
        auto start_std = chrono::high_resolution_clock::now();
        Matrix C_std = standardMultiply(A, B);
        auto end_std = chrono::high_resolution_clock::now();
        time_std = chrono::duration<double, milli>(end_std - start_std).count();

        // Sequential Strassen
        auto start_seq = chrono::high_resolution_clock::now();
        Matrix C_seq = strassenSequential(A, B);
        auto end_seq = chrono::high_resolution_clock::now();
        time_seq = chrono::duration<double, milli>(end_seq - start_seq).count();

        // Parallel Strassen
        omp_set_num_threads(4); // Установка количества потоков
        auto start_par = chrono::high_resolution_clock::now();
        Matrix C_par = strassenParallel(A, B);
        auto end_par = chrono::high_resolution_clock::now();
        time_par = chrono::duration<double, milli>(end_par - start_par).count();

        outFile << n << "," << time_std << "," << time_seq << "," << time_par << "\n";
    }

    outFile.close();
    cout << "Benchmark results saved to benchmark_powers_of_two.csv" << endl;
}

void benchmarkExtendedPowersOfTwo() {
    ofstream outFile("benchmark_extended_powers_of_two.csv");
    outFile << "Size,Standard (ms),Sequential Strassen (ms),Parallel Strassen (ms)\n";

    vector<int> sizes;
    for (int i = 1; i <= 10; i++) {
        sizes.push_back(1 << i);  // 2, 4, 8, ..., 1024
    }

    // Добавляем дополнительные степени двойки между существующими
    vector<int> extendedSizes;
    for (int i = 0; i < sizes.size() - 1; i++) {
        extendedSizes.push_back(sizes[i]);
        extendedSizes.push_back((sizes[i] + sizes[i + 1]) / 2);
    }
    extendedSizes.push_back(sizes.back());

    for (int n : extendedSizes) {
        // Округляем до ближайшей степени двойки для Strassen
        int strassenSize = 1;
        while (strassenSize < n) strassenSize <<= 1;
        if (strassenSize - n > n - strassenSize / 2 && strassenSize / 2 >= STRASSEN_THRESHOLD) {
            strassenSize /= 2;
        }

        cout << "Benchmarking " << n << "x" << n << " (Strassen uses " << strassenSize << "x" << strassenSize << ")..." << endl;

        Matrix A = generateRandomMatrix(n);
        Matrix B = generateRandomMatrix(n);

        double time_std, time_seq, time_par;

        // Standard multiply
        auto start_std = chrono::high_resolution_clock::now();
        Matrix C_std = standardMultiply(A, B);
        auto end_std = chrono::high_resolution_clock::now();
        time_std = chrono::duration<double, milli>(end_std - start_std).count();

        // Sequential Strassen
        auto start_seq = chrono::high_resolution_clock::now();
        Matrix C_seq;
        if (n == strassenSize) {
            C_seq = strassenSequential(A, B);
        }
        else {
            // Дополняем матрицы до степени двойки
            Matrix A_padded(strassenSize, vector<int>(strassenSize, 0));
            Matrix B_padded(strassenSize, vector<int>(strassenSize, 0));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A_padded[i][j] = A[i][j];
                    B_padded[i][j] = B[i][j];
                }
            }
            Matrix C_padded = strassenSequential(A_padded, B_padded);
            C_seq = Matrix(n, vector<int>(n));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C_seq[i][j] = C_padded[i][j];
                }
            }
        }
        auto end_seq = chrono::high_resolution_clock::now();
        time_seq = chrono::duration<double, milli>(end_seq - start_seq).count();

        // Parallel Strassen
        omp_set_num_threads(12);
        auto start_par = chrono::high_resolution_clock::now();
        Matrix C_par;
        if (n == strassenSize) {
            C_par = strassenParallel(A, B);
        }
        else {
            Matrix A_padded(strassenSize, vector<int>(strassenSize, 0));
            Matrix B_padded(strassenSize, vector<int>(strassenSize, 0));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A_padded[i][j] = A[i][j];
                    B_padded[i][j] = B[i][j];
                }
            }
            Matrix C_padded = strassenParallel(A_padded, B_padded);
            C_par = Matrix(n, vector<int>(n));
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    C_par[i][j] = C_padded[i][j];
                }
            }
        }
        auto end_par = chrono::high_resolution_clock::now();
        time_par = chrono::duration<double, milli>(end_par - start_par).count();

        outFile << n << "," << time_std << "," << time_seq << "," << time_par << "\n";
    }

    outFile.close();
    cout << "Benchmark results saved to benchmark_extended_powers_of_two.csv" << endl;
}

int main() {
    benchmarkPowersOfTwo();
    benchmarkExtendedPowersOfTwo();
    return 0;
}