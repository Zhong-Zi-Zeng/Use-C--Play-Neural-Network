#include <iostream>
#include <math.h>
#include <algorithm>
#include <vector>
#include <time.h>

using namespace std;

template <class T, size_t ROW, size_t COL>
class Matrix{
public:
    T matrix[ROW][COL];
    size_t row = ROW;
    size_t col = COL;


    Matrix(){
        random_initial();
    }

    Matrix(double fill_value){
        generate_mat(fill_value);
    }


    void generate_mat(double fill_value) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                matrix[r][c] = fill_value;
            }
        }
    }

    void random_initial() {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                matrix[r][c] = 2 * rand() / (RAND_MAX + 1.0) - 1;
            }
        }
    }

    void show_mat(){
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                printf("%6f ", matrix[r][c]);
            }
            cout << endl;
        }
        cout << endl;
    }


    Matrix *operator+(double other) {
        Matrix *result = new Matrix<double, ROW, COL> ();

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result.matrix[r][c] = matrix[r][c] + other;
            }
        }

        return result;
    }


    Matrix operator-() {
        Matrix<double, ROW, COL> result {0};

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result.matrix[r][c] = -matrix[r][c];
            }
        }
        return result;
    }
};



int main(){
    srand(2);

    cout << rand();

    return 0;
}