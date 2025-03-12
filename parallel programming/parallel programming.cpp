#include <iostream>
#include <omp.h>

int main() {
    const int size = 1000000;
    int* array = new int[size];

    // Инициализация массива
#pragma omp parallel for
    for (int i = 0; i < size; i++) {
        array[i] = i;
    }

    // Параллельное вычисление суммы элементов массива
    long long sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    std::cout << "Sum of elements: " << sum << std::endl;

    delete[] array;
    return 0;
}
