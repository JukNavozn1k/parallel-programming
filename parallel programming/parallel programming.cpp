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
    // #pragma omp parallel for collapse(3)
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
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix subtract(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<int>(n, 0));
    // #pragma omp parallel for collapse(2)
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

    // Сборка результата
    Matrix C(n, vector<int>(n));
    // #pragma omp parallel for collapse(2)
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
int main() {
    vector<int> sizes = { 8, 16, 32, 64, 128 };
    ofstream file_std("standard_multiply.txt");
    ofstream file_seq("strassen_sequential.txt");
    ofstream file_par("strassen_parallel.txt");

    omp_set_num_threads(12);

    for (int n : sizes) {
        Matrix A = generateRandomMatrix(n);
        Matrix B = generateRandomMatrix(n);

        auto start = chrono::high_resolution_clock::now();
        standardMultiply(A, B);
        auto end = chrono::high_resolution_clock::now();
        file_std << n << " " << chrono::duration<double, milli>(end - start).count() << "\n";

        start = chrono::high_resolution_clock::now();
        strassenSequential(A, B);
        end = chrono::high_resolution_clock::now();
        file_seq << n << " " << chrono::duration<double, milli>(end - start).count() << "\n";

        start = chrono::high_resolution_clock::now();
        strassenParallel(A, B);
        end = chrono::high_resolution_clock::now();
        file_par << n << " " << chrono::duration<double, milli>(end - start).count() << "\n";
    }

    file_std.close();
    file_seq.close();
    file_par.close();

    return 0;
}