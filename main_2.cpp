#include <iostream>
#include <math.h>
#include <algorithm>
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
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i][j] = fill_value;
            }
        }
    }

    void random_initial() {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i][j] = 2 * rand() / (RAND_MAX + 1.0) - 1;
            }
        }
    }

};







int main(){
    srand(time(NULL));


    Matrix<double, 3> w {};
    cout << w.matrix[0][2] << endl;
    cout << w.col;


    return 0;
}