#include <iostream>
#include <fstream>
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

//прототип функции zscal из BLAS для умножения вектора на скаляр
extern "C" void zscal_(const int* n, const complex<double>*alpha, complex<double>*x, const int* incx);

//прототип функции zgeev из LAPACK для вычисления собственных значений, собственных векторов
extern "C" void zgeev_(const char* jobvl, const char* jobvr, const int* n,
    complex<double>*a, const int* lda,
    complex<double>*w, complex<double>*vl, const int* ldvl,
    complex<double>*vr, const int* ldvr, complex<double>*work,
    const int* lwork, double* rwork, int* info);

void findEigen(complex<double>* A, const int* n, complex<double>* w, complex<double>* vl, complex<double>* vr)
{
    //параметры для функции zgeev (поиск собственных значений и векторов)
    const char jobvl = 'V';  //вычислять левые собственные векторы
    const char jobvr = 'V';  //вычислять правые собственные векторы

    //рабочие массивы
    int lwork = 2 * (*n); //размер рабочего массива
    vector<complex<double>> work(lwork);
    vector<double> rwork(lwork);

    int info;

    //поиск собственных значений и собственных векторов
    zgeev_(&jobvl, &jobvr, n, A, n, w, vl, n, vr, n, work.data(), &lwork, rwork.data(), &info);

    if (info != 0)
    {
        cerr << "Ошибка в zgeev, код: " << info << endl;
    }
}

//функция для чтения матрицы и вектора из файла
void readFile(int& n, vector<complex<double>>& matrix, vector<complex<double>>& vect) {
    ifstream f("input.txt");

    if (!f.is_open()) {
        cerr << "Не удалось открыть файл!" << endl;
        return;
    }

    f >> n;

    matrix.resize(n * n);
    vect.resize(n);

    for (int i = 0; i < n * n; ++i) {
        double real, imag;
        f >> real >> imag;
        matrix[i] = complex<double>(real, imag);
    }

    for (int i = 0; i < n; ++i) {
        double real, imag;
        f >> real >> imag;
        vect[i] = std::complex<double>(real, imag);
    }

    f.close();

    vector<complex<double>> transposed(n*n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[j * n + i] = matrix[i * n + j];
        }
    }

    matrix = transposed;
}

int main() {
    setlocale(LC_ALL, "rus");

    int n;

    //матрица A
    vector<complex<double>> A;

    //вектор b
    vector<complex<double>> b;

    readFile(n, A, b);

    //вектор x (результат)
    vector<complex<double>> x(n, { 0.0f, 0.0f });

    //параметры для функции cgemv (умножение комплексной матрицы на комплексный вектор)
    const char trans = 'N'; //не транспонировать
    const complex<double> alpha = { 1.0f, 0.0f }; //коэффициент
    const complex<double> beta = { 0.0f, 0.0f };  //коэффициент
    const int lda = n;        //Leading dimension (ведущая размерность) для A
    const int incb = 1;       //шаг между элементами вектора b
    const int incx = 1;       //шаг между элементами вектора x

    //умножение матрицы A на вектор b: x = alpha * A * b + beta * x
    zgemv_(&trans, &n, &n, &alpha, A.data(), &lda, b.data(), &incb, &beta, x.data(), &incx);

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

    vector<complex<double>> w(n); //собственные значения

    vector<complex<double>> vl(n * n); //левые собственные векторы
    vector<complex<double>> vr(n * n); //правые собственные векторы

    findEigen(A.data(), &n, w.data(), vl.data(), vr.data());

    cout << "Собственные значения:\n";
    for (int i = 0; i < n; ++i) {
        cout << "lambda_" << i + 1 << " = " << w[i] << endl;
    }

    cout << "\nПравые собственные векторы:\n";
    for (int i = 0; i < n; ++i) {
        cout << "v_" << i + 1 << " = [";
        for (int j = 0; j < n; ++j) {
            cout << vr[i * n + j]; //хранится по столбцам
            if (j != n - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    cout << "\nЛевые собственные векторы:\n";
    for (int i = 0; i < n; ++i) {
        cout << "vl_" << i + 1 << " = [";
        for (int j = 0; j < n; ++j) {
            cout << vl[i * n + j];
            if (j != n - 1) cout << ", ";
        }
        cout << "]" << endl;
    }

    cout << "\n\nПроверка\n";
    for (int i = 0; i < n; ++i) {
        vector<complex<double>> vect(n);

        for (int j = 0; j < n; j++)
        {
            vect[j] = vr[i * n + j];
        }

        //умножение матрицы A на вектор vect: x = alpha * A * vect + beta * x
        zgemv_(&trans, &n, &n, &alpha, A.data(), &lda, vect.data(), &incb, &beta, x.data(), &incx);

        // Вывод результата
        cout << "A * v_" << i + 1 << ":\n";
        for (const auto& val : x) {
            cout << val.real() << " + (" << val.imag() << ") * i\n";
        }
        cout << endl;


        zscal_(&n, &(w[i]), vect.data(), &incx);

        // Вывод результата
        cout << "lambda_" << i + 1 << " * v_" << i + 1 << ":\n";
        for (const auto& val : x) {
            cout << val.real() << " + (" << val.imag() << ") * i\n";
        }
        cout << "-----------------------------------------------" << endl;
    }

    return 0;
}