#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <malloc.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
using std::vector;
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include "inverse.h"
//using namespace boost::numeric::ublas;

//функция вывода матрицы
void printMatrix (boost::numeric::ublas::matrix<double> m, int sizeY, int sizeX)
{
  //sizeY - количество строк
  //sizeX - количество столбцов
    printf("---------------------------------------------\n");
    for(int i = 0; i < sizeY; ++i){
        for(int j = 0; j<sizeX; ++j){
            std::cout<<std::setprecision(6)<<m(i,j)<<'\t';
        }
        std::cout<<std::endl;
    }
    printf("---------------------------------------------\n");
}
// Функция заполнения матрицы нулями
void matrixInit (boost::numeric::ublas::matrix<double> &m, int size)
{
    for(int i = 0; i < size; ++i){
        for(int j = 0; j<size; ++j){
            m(i,j) = 0.0;
        }
    }
}

// решаем задачу от 0 до 1
//задаем функцию y'' = f(x,y,y')
// v = y'
double func (double x, double y, double v)
{
    //return sin(4*x)+y*v*v;
    //return cos(4*x);
    //return cos(y);
    return exp(-y);
}
//Производная по у
double func_y (double x, double y, double v)
{
    return -exp(-y);
}
//Производная по y'
double func_v (double x, double y, double v)
{
    return 0;
}

//решение задачи Коши для вычисления коэффициентов матрицы Якоби
// p' = q
// q' = a*p + b*q
vector<double> Cauchy_problem (vector <double> &y, vector <double> &v, int N, double p0, double q0, double x0, double h)
{
    vector <double> p(N, 0.0);
    vector <double> q(N, 0.0);
    p[0] = p0;
    q[0] = q0;
    double t1, t2, k1, k2;
    double x;
    double a, b;
    x = x0;
    for (int i = 0; i < N-1; ++i)
    {
        a = func_y(x,y[i],v[i]);
        b = func_v(x,y[i],v[i]);

        k1 = h * q[i];
        t1 = h * (a * p[i] + b * q[i] );

        k2 = h * (q[i] + t1);
        t2 = h * (a * (p[i]+k1) + b * (q[i] + t1));

        p[i + 1] = p[i] + (k1+k2)/2;
        q[i + 1] = q[i] + (t1+t2)/2;
        x += h;
    }
    vector <double> arr(2);
    arr[0] = p[N -1];
    arr[1] = q[N -1];
    return arr;
}

int main(int argc, char *argv[])
{
    MPI_Status status;
    int rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS){ printf("Error starting MPI program. Terminating.\n"); MPI_Abort(MPI_COMM_WORLD, 1);  }
    int rank, size;
    rc = MPI_Comm_size(MPI_COMM_WORLD, &size); if (rc != MPI_SUCCESS){ MPI_Abort(MPI_COMM_WORLD, 1); }
    rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank); if (rc != MPI_SUCCESS){ MPI_Abort(MPI_COMM_WORLD, 1); }
    double A1, B1, A2, B2, U1, U2;
    double k1, k2, t1, t2;

    // начальные условия
    // A1 * y'(0) + B1*y(0) = U1
    // A2 * y'(1) + B2*y(1) = U2
    A1 = 0.0; B1 = 1.0; U1 = 1.0; A2 = 0.0; B2 = 1.0; U2 = 0.7;
    // псевдогенератор
    srand(time(NULL));  srandom(rank * 7);
    int N = 100; // разбиение отрезка для одного процесса
    double eps = 1e-3;
    double h = (1.0 / size) / N; // шаг для каждого процесса
    double x0 = 1.0 / size * rank; // начальная точка для каждого процесса
    double p0, q0; // начальные условия для вспомогательной задачи Коши
    vector <double> y (N+1, 0.0);
    vector <double> v (N+1, 0.0);

    if (rank == 0)
    {
        printf("Number of processes = %d\n", size);
        printf("Step (h) = %f\n", h);
        printf("Grid for one process (N) = %d\n", N);
        printf("%.2f * y'(0) + %.2f * y(0) = %.2f\n", A1,B1,U1);
        printf("%.2f * y'(1) + %.2f * y(1) = %.2f\n", A2,B2,U2);
        int iter = 0; // количество итераций

        //пристрелочный параметр гамма для нулевого процесса
        double gamma = (double)rand() / RAND_MAX * 0.05;
        while(1) {
          iter += 1;
          // y(0) = gamma else  y'(0) = gamma
          //Задаем начальные условия
          if (abs(A1) > abs(B1))
          {
              y[0] = gamma;
              v[0] = (U1 - B1 * gamma) / A1;
              p0 = 1;
              q0 = -B1 / A1;
          }
          else
          {
            v[0] = gamma;
            y[0] = (U1 - A1 * gamma) / B1;
            p0 = -A1 / B1;
            q0 = 1;
          }
          //задача Коши основная
          double x;
            x = x0;
            for (int i = 0; i < N; ++i)
            {
                k1 = h * v[i];
                t1 = h * func(x, y[i], v[i]);
                k2 = h * (v[i] + t1);
                t2 = h * func(x, y[i] + k1, v[i] + t1);
                y[i+1] = y[i] + (k1 + k2) / 2.0;
                v[i+1] = v[i] + (t1 + t2) / 2.0;
                x += h;
           }

          vector <double> buff (2, 0.0);
          buff = Cauchy_problem (y, v, N + 1, p0, q0, x0, h);  // коэффициенты матрицы Якоби

          boost::numeric::ublas::matrix<double> Jacobi (2*size-1, 2*size-1);
          matrixInit (Jacobi,2*size-1); // обнуление матрицы

          Jacobi(0,0) = buff[0];
          Jacobi(0,1) = -1;
          Jacobi(1,0) = buff[1];
          Jacobi(1,2) = -1;
          double *recv_buff_1_0 = new double[2];
          double *recv_buff_0_1 = new double[2];
          for (int i = 1; i < size; ++i)
          {
            //Производные по кси получаем
            rc = MPI_Recv(recv_buff_1_0, 2, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, &status); if (rc != MPI_SUCCESS) {MPI_Abort(MPI_COMM_WORLD, 1);}
            //Производные по ета получаем
            rc = MPI_Recv(recv_buff_0_1, 2, MPI_DOUBLE, i, 101, MPI_COMM_WORLD, &status); if (rc != MPI_SUCCESS) {MPI_Abort(MPI_COMM_WORLD, 1);}

            if (status.MPI_SOURCE == size - 1)
            {
                Jacobi(2 * size - 2, 2 * size - 3) = A2 * recv_buff_1_0[1] + B2 * recv_buff_1_0[0];
                Jacobi(2 * size - 2, 2 * size - 2) = A2 * recv_buff_0_1[1] + B2 * recv_buff_0_1[0];
            }
            else
            {
                Jacobi(i * 2, (i + 1) * 2 - 1) = -1;
                Jacobi(i * 2, (i + 1) * 2 - 3) = recv_buff_1_0[0];
                Jacobi(i * 2, (i + 1) * 2 - 2) = recv_buff_0_1[0];
                Jacobi(i * 2 + 1, (i + 1) * 2) = -1;
                Jacobi(i * 2 + 1, (i + 1) * 2 - 3) = recv_buff_1_0[1];
                Jacobi(i * 2 + 1, (i + 1) * 2 - 2) = recv_buff_0_1[1];
            }
        } // конец матрицы Якоби



        boost::numeric::ublas::matrix<double> R (2*size-1,1); // вектор невязок
        boost::numeric::ublas::matrix<double> Vars_old (2*size-1,1); // старые пристрелочные параметры
        boost::numeric::ublas::matrix<double> Vars_new (2*size-1,1); // новые пристрелочные параметры
        Vars_old(0,0) = gamma;
        R(0,0) = y[N];
        R(1,0) = v[N];

        // первые два - пристрелочные параметры, вторые два  - полученные в конце решения
        double *recv_buff = new double[4];
        for( int i = 1; i < size; ++i ){
          rc = MPI_Recv( recv_buff, 4, MPI_DOUBLE, i, 999, MPI_COMM_WORLD, &status); if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }
          Vars_old(2 * i - 1, 0) = recv_buff[0];
          Vars_old(2 * i, 0) = recv_buff[1];
          if (i != size-1)
          {
            R(2 * i - 2, 0) -= recv_buff[0];
            R(2 * i - 1, 0) -= recv_buff[1];
            R(2 * i, 0) = recv_buff[2]; // y
            R(2 * i + 1, 0) = recv_buff[3]; // y'
          }
          else {
            R(2 * size - 4, 0) -= recv_buff[0];
            R(2 * size - 3, 0) -= recv_buff[1];
            R(2 * size - 2, 0) = A2 * recv_buff[3] + B2 * recv_buff[2] - U2;
          }
        }

        //критерий остановки (находим максимум в векторе невязок)
        double MAX = -999;
        for( int i = 0; i < 2 * size - 1; ++i )
          if( abs(R(i,0)) > MAX ) MAX = abs(R(i,0));
        // если попали в условие выхода отправляем 1, и записываем файл
        if( MAX < eps )
        {
          printf("End of work.\nIterations = %d\n", iter);
          //отправка завершения работы
          int send_end = 1;
          for( int i = 1; i < size; ++i)
            rc = MPI_Send(&send_end, 1, MPI_INT, i, 555, MPI_COMM_WORLD);
          delete [] recv_buff; delete [] recv_buff_1_0; delete [] recv_buff_0_1;
          // принимаю и записываю в файл
          std::ofstream fout;
          fout.open("answer.txt"); // связываем объект с файлом
          double x_ans = 0;
          for (int i = 0; i < N + 1; ++i)
          {
            fout << x_ans << ", " << y[i] << "," << std::endl;
            x_ans += h;
          }

          for( int i = 1; i < size; ++i) {
            double* recv_y = new double[N+1];
            rc = MPI_Recv(recv_y, N+1, MPI_DOUBLE, i, 666, MPI_COMM_WORLD, &status); if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }

            for (int j = 1; j < N + 1; ++j)
            {
              fout << x_ans << ", " <<recv_y[j] << "," << std::endl;
              x_ans += h;
            }
            delete [] recv_y;
          }
          fout.close();
          printf("File is ready.\n");
          MPI_Finalize();
          return 0;
        }
        else
        {
          //не попали в условие выхода, отправляем всем 2
          int send_cont = 2;
          for (int i = 1; i < size; ++i)
            rc = MPI_Send(&send_cont, 1, MPI_INT, i, 555, MPI_COMM_WORLD);
        }

        // считаем обратную матрицу
        bool flag = false;
        boost::numeric::ublas::matrix<double> Jacobi_inverse(2*size-1,2*size-1); matrixInit(Jacobi_inverse, 2*size-1);
        Jacobi_inverse = gjinverse<double>(Jacobi,flag);

        // Метод Ньютона
        Vars_new = Vars_old - boost::numeric::ublas::prod(Jacobi_inverse, R);

        //рассылка параметров процессам
        double *send_buff = new double[2];
        for (int i = 1; i < size; ++i)
        {
          send_buff[0] = Vars_new(2*i-1,0);
          send_buff[1] = Vars_new(2*i,0);
          rc = MPI_Send(send_buff, 2, MPI_DOUBLE, i, 101, MPI_COMM_WORLD); if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }
        }
        gamma = Vars_new(0,0);
        delete [] send_buff;
        delete [] recv_buff;
        delete [] recv_buff_1_0;
        delete [] recv_buff_0_1;

        if (iter == 1e3)
        {
          printf("Iterations = 1e3, its bad\n");
        }
      } // while 1

    } // rank=0

    if (rank > 0)
    {
        //задание пристрелочных параметров
        double ksi = (double)rand() / RAND_MAX * 0.05;
        double eta = (double)rand() / RAND_MAX * 0.05;
        while (1)
        {
            y[0] = ksi;
            v[0] = eta;
            double x;
                x = x0;
                for (int i = 0; i < N; ++i)
                {
                  k1 = h * v[i];
                  t1 = h * func(x, y[i], v[i]);

                  k2 = h * (v[i] + t1);
                  t2 = h * func(x, y[i] + k1, v[i] + t1);

                  y[i+1] = y[i] + (k1 + k2) / 2.0;
                  v[i+1] = v[i] + (t1 + t2) / 2.0;
                  x += h;
                }

            vector <double> buff_1_0_tmp(2); vector <double> buff_0_1_tmp(2);

            buff_1_0_tmp = Cauchy_problem(y, v, N + 1, 1, 0, x0, h); // Производная по кси
            buff_0_1_tmp = Cauchy_problem(y, v, N + 1, 0, 1, x0, h); // Производная по ета

            // переделываем данные для отправки
            double *buff_1_0 = new double[2];
            double *buff_0_1 = new double[2];
            buff_1_0[0] = buff_1_0_tmp[0]; buff_1_0[1] = buff_1_0_tmp[1]; buff_0_1[0] = buff_0_1_tmp[0]; buff_0_1[1] = buff_0_1_tmp[1];

            // отправка нулевому процессу коэффициентов матрицы Якоби
            rc = MPI_Send(buff_1_0, 2, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);if( rc != MPI_SUCCESS ){ MPI_Abort(MPI_COMM_WORLD, 1); }
            rc = MPI_Send(buff_0_1, 2, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);if( rc != MPI_SUCCESS ){ MPI_Abort(MPI_COMM_WORLD, 1); }
            double *send_buff = new double[4];

            send_buff[0] = ksi;
            send_buff[1] = eta;
            send_buff[2] = y[N];
            send_buff[3] = v[N];

            // отправка данных для вектора невязок 999
            rc = MPI_Send(send_buff, 4, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }

            //проверка на завершение работы 555
            int recv_end = 10;
            rc = MPI_Recv(&recv_end, 1, MPI_INT, 0, 555, MPI_COMM_WORLD, &status); if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }
            //если приняли сигнал на завершение
            if( recv_end == 1 ) {
              double* send_y = new double[N+1];
              for( int i = 0; i < N+1; ++i )
                send_y[i] = y[i];
              rc = MPI_Send(send_y, N+1, MPI_DOUBLE, 0, 666, MPI_COMM_WORLD); if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }
              delete [] send_buff;
              delete [] buff_1_0;
              delete [] buff_0_1;
              delete [] send_y;
              MPI_Finalize();
              return 0;
            }

            //получение новых параметров
            double *main_recv_buff = new double[2];
            rc = MPI_Recv(main_recv_buff, 2, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &status); if( rc != MPI_SUCCESS ) { MPI_Abort( MPI_COMM_WORLD, 1); }

            ksi = main_recv_buff[0];
            eta = main_recv_buff[1];

            delete [] main_recv_buff;
            delete [] send_buff;
            delete [] buff_1_0;
            delete [] buff_0_1;
        } // while 1
    } //rank > 0

    return 0;
}
