#include <iostream>
#include <complex>
#include <vector>
using namespace std;

//прототип функции zgemv из BLAS для умножения матрицы на вектор
//y = alpha * A * x + beta * y
extern "C" void zgemv_(const char* trans, const int* m, const int* n,
    const complex<double> *alpha, const complex<double> *A,
    const int* lda, const complex<double> *x,
    const int* incx, const complex<double> *beta,
    complex<double> *y, const int* incy);

//прототип функции zaxpy из BLAS для сложения векторов
//y = alpha * x + y
extern "C" void zaxpy_(const int* n, const complex<double>*alpha,
    const complex<double>*x, const int* incx,
    complex<double>*y, const int* incy);

//прототип функции zgeev из LAPACK для вычисления собственных значений, собственных векторов
extern "C" void zgeev_(const char* jobvl, const char* jobvr, const int* n,
    complex<double>*a, const int* lda,
    complex<double>*w, complex<double>*vl, const int* ldvl,
    complex<double>*vr, const int* ldvr, complex<double>*work,
    const int* lwork, double* rwork, int* info);

int main() {
    setlocale(LC_ALL, "rus");

    int m = 3; //количество строк
    int n = 3; //количество столбцов

    //матрица A
    vector<complex<double>> A = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, //первый столбец
        {7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f},
        {13.0f, 14.0f}, {15.0f, 16.0f}, {17.0f, 18.0f}
    };

    //вектор b
    vector<complex<double>> b = {
        {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}
    };

    //вектор x (результат)
    vector<complex<double>> x(m, { 0.0f, 0.0f });

    //параметры для функции cgemv (умножение комплексной матрицы на комплексный вектор)
    const char trans = 'N'; //не транспонировать
    const complex<double> alpha = { 1.0f, 0.0f }; //коэффициент
    const complex<double> beta = { 0.0f, 0.0f };  //коэффициент
    const int lda = n;        //Leading dimension (ведущая размерность) для A
    const int incb = 1;       //шаг между элементами вектора b
    const int incx = 1;       //шаг между элементами вектора x

    //умножение матрицы A на вектор b: x = alpha * A * b + beta * x
    zgemv_(&trans, &m, &n, &alpha, A.data(), &lda, b.data(), &incb, &beta, x.data(), &incx);

    // Вывод результата
    cout << "Результат умножения комплексной матрицы на комплексный вектор (x = A * b):\n";
    for (const auto& val : x) {
        cout << val.real() << " + (" << val.imag() << ") * i\n";
    }
    cout << endl;

    //сложение векторов b и x: x = alpha * b + x
    zaxpy_(&n, &alpha, b.data(), &incb, x.data(), &incx);

    // Вывод результата
    cout << "Результат сложения векторов b и x (x = b + x):\n";
    for (const auto& val : x) {
        cout << val.real() << " + (" << val.imag() << ") * i\n";
    }
    cout << endl;

    //параметры для функции zgeev (поиск собственных значений и векторов)
    const char jobvl = 'V';  //вычислять левые собственные векторы
    const char jobvr = 'V';  //вычислять левые собственные векторы

    vector<complex<double>> w(n); //собственные значения

    vector<complex<double>> vl(n * n); //левые собственные векторы
    vector<complex<double>> vr(n * n); //правые собственные векторы

    //рабочие массивы
    vector<complex<double>> work(2 * n);
    vector<double> rwork(2 * n);
    int lwork = 2 * n; //размер рабочего массива

    int info;

    //поиск собственных значений и собственных векторов
    zgeev_(&jobvl, &jobvr, &n, A.data(), &lda, w.data(), vl.data(), &n, vr.data(), &n, work.data(), &lwork, rwork.data(), &info);

    if (info == 0) {
        cout << "Собственные значения:\n";
        for (int i = 0; i < n; ++i) {
            cout << "lambda_" << i + 1 << " = " << w[i] << endl;
        }

        // Вывод правых собственных векторов
        cout << "\nПравые собственные векторы:\n";
        for (int i = 0; i < n; ++i) {
            cout << "v_" << i + 1 << " = [";
            for (int j = 0; j < n; ++j) {
                cout << vr[j + i * n]; // Правый собственный вектор хранится по столбцам
                if (j != n - 1) cout << ", ";
            }
            cout << "]" << endl;
        }

        cout << "\nЛевые собственные векторы:\n";
        for (int i = 0; i < n; ++i) {
            cout << "v_" << i + 1 << " = [";
            for (int j = 0; j < n; ++j) {
                cout << vl[j + i * n];
                if (j != n - 1) cout << ", ";
            }
            cout << "]" << endl;
        }
    }
    else {
        cerr << "Ошибка в zgeev, код: " << info << endl;
    }

    return 0;
}