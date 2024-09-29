#include <iostream>
#include <complex>
#include <vector>
using namespace std;

//прототип функции cgemv из BLAS для умножения матрицы на вектор
//y = alpha * A * x + beta * y
extern "C" void cgemv_(const char* trans, const int* m, const int* n,
    const complex<float> *alpha, const complex<float> *A,
    const int* lda, const complex<float> *x,
    const int* incx, const complex<float> *beta,
    complex<float> *y, const int* incy);

//прототип функции caxpy из BLAS для сложения векторов
//y = alpha * x + y
extern "C" void caxpy_(const int* n, const complex<float>*alpha,
    const complex<float>*x, const int* incx,
    complex<float>*y, const int* incy);

int main() {
    setlocale(LC_ALL, "rus");

    int m = 3; //количество строк
    int n = 3; //количество столбцов

    //матрица A
    vector<complex<float>> A = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, //первый столбец
        {7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f},
        {13.0f, 14.0f}, {15.0f, 16.0f}, {17.0f, 18.0f}
    };

    //вектор b
    vector<complex<float>> b = {
        {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    };

    //вектор x (результат)
    vector<complex<float>> x(m, { 0.0f, 0.0f });

    //параметры для функции cgemv (умножение комплексной матрицы на комплексный вектор)
    const char trans = 'N'; //не транспонировать
    const complex<float> alpha = { 1.0f, 0.0f }; //коэффициент перед матрицей A
    const complex<float> beta = { 0.0f, 0.0f };  //коэффициент перед вектором x
    const int lda = n;        //Leading dimension (ведущая размерность) для A
    const int incb = 1;       //шаг между элементами вектора b
    const int incx = 1;       //шаг между элементами вектора x

    //умножение матрицы A на вектор b: x = alpha * A * b + beta * x
    cgemv_(&trans, &m, &n, &alpha, A.data(), &lda, b.data(), &incb, &beta, x.data(), &incx);

    // Вывод результата
    cout << "Результат умножения комплексной матрицы на комплексный вектор (x = A * b):\n";
    for (const auto& val : x) {
        cout << val.real() << " + (" << val.imag() << ") * i\n";
    }
    cout << endl;

    //умножение матрицы A на вектор b: x = alpha * A * b + beta * x
    caxpy_(&n, &alpha, b.data(), &incb, x.data(), &incx);

    // Вывод результата
    cout << "Результат сложения векторов b и x (x = b + x):\n";
    for (const auto& val : x) {
        cout << val.real() << " + (" << val.imag() << ") * i\n";
    }
    cout << endl;

    return 0;
}