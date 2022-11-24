#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <math.h>
#include <time.h>

using namespace std;


class Matrix {
public:
    double **matrix;
    int row = 0;
    int col = 0;

    Matrix() {
        _declare_mem(0, 0);
    }

    template<size_t row, size_t col>
    Matrix(double (&arr)[row][col]) {
        /*!
         * 將傳遞進來的2維陣列轉換為matrix類別
         * @tparam row
         * @tparam col
         * @param arr
         */
        _declare_mem(row, col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                Matrix::matrix[r][c] = arr[r][c];
            }
        }
    }

    Matrix(int row, int col) {
        _declare_mem(row, col);
    }

    Matrix(int row, int col, double fill_value) {
        _declare_mem(row, col);
        fill(fill_value);
    }

    ~Matrix() {
//        cout << "delete: " << matrix << endl;
        delete matrix;
    }

    void _declare_mem(int row, int col){
        this->row = row;
        this->col = col;
        matrix = new double *[row];
        for (int r = 0; r < row; r++) {
            matrix[r] = new double[col]();
        }
    }

    unique_ptr<Matrix> operator+(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        auto result = make_unique<Matrix>(Matrix::row, Matrix::col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][0] + other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] + other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][c] + other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][0] + other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] + other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] + other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] + other.matrix[r][c];
                }
            }
        }

        return result;
    }

    unique_ptr<Matrix> operator-(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        auto result = make_unique<Matrix>(Matrix::row, Matrix::col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][0] - other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] - other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][c] - other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][0] - other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] - other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] - other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] - other.matrix[r][c];
                }
            }
        }

        return result;
    }

    unique_ptr<Matrix> operator*(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        auto result = make_unique<Matrix>(Matrix::row, Matrix::col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][0] * other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] * other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][c] * other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][0] * other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] * other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] * other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] * other.matrix[r][c];
                }
            }
        }

        return result;
    }

    unique_ptr<Matrix> operator/(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        auto result = make_unique<Matrix>(Matrix::row, Matrix::col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][0] / other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] / other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[0][c] / other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][0] / other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] / other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] / other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result->matrix[r][c] = Matrix::matrix[r][c] / other.matrix[r][c];
                }
            }
        }

        return result;
    }

    unique_ptr<Matrix> operator+(double num) {
        // 初始化結果
        auto result = std::make_unique<Matrix>(Matrix::row, Matrix::col);
        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result->matrix[r][c] = Matrix::matrix[r][c] + num;
            }
        }
        return result;
    }

    unique_ptr<Matrix> operator-(double num) {
        // 初始化結果
        auto result = std::make_unique<Matrix>(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result->matrix[r][c] = Matrix::matrix[r][c] - num;
            }
        }

        return result;
    }

    unique_ptr<Matrix> operator*(double num) {
        // 初始化結果
        auto result = std::make_unique<Matrix>(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result->matrix[r][c] = Matrix::matrix[r][c] * num;
            }
        }

        return result;
    }

    unique_ptr<Matrix> operator/(double num) {
        // 初始化結果
        auto result = std::make_unique<Matrix>(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result->matrix[r][c] = Matrix::matrix[r][c] / num;
            }
        }

        return result;
    }

    unique_ptr<Matrix> dot(Matrix &other) {
        /*!
         * 將兩個matrix進行點積運算，且在點積前檢查維度是否正確
         * @param Matrix other
         * @return Matrix
         */

        assert((Matrix::col == other.row));

        // 初始化結果
        auto result = std::make_unique<Matrix>(Matrix::row, Matrix::col);

        for (int i = 0; i < Matrix::row; i++) {
            for (int j = 0; j < other.col; j++) {
                for (int k = 0; k < other.row; k++) {
                    result->matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }

        return result;
    }

    Matrix transpose() {
        /*!
         * 將matrix進行轉置
         * @return Matrix
         */

        // 初始化結果
        Matrix result(Matrix::row, Matrix::col, 0);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++)
                result.matrix[j][i] = Matrix::matrix[i][j];
        }

        return result;
    }

    static unique_ptr<Matrix> ln(unique_ptr<Matrix> other){
        return ln(*other);
    }

    static unique_ptr<Matrix> ln(Matrix &_matrix) {
        /*!
         * 計算傳入matrix的ln值
         * @param _matrix
         * @return Matrix result
         */
        // 初始化結果
        auto result = std::make_unique<Matrix>(_matrix.row, _matrix.col);


        for (int r = 0; r < _matrix.row; r++) {
            for (int c = 0; c < _matrix.col; c++)
                result->matrix[r][c] = log(_matrix.matrix[r][c]);
        }

        return result;
    }

    static unique_ptr<Matrix> exp(unique_ptr<Matrix> other){
        return exp(*other);
    }

    static unique_ptr<Matrix> exp(Matrix &_matrix) {
        /*!
         * 計算傳入matrix的exp值
         * @param _matrix
         * @return Matrix result
         */
        // 初始化結果
        auto result = std::make_unique<Matrix>(_matrix.row, _matrix.col);

        for (int r = 0; r < _matrix.row; r++) {
            for (int c = 0; c < _matrix.col; c++)
                result->matrix[r][c] = std::exp(_matrix.matrix[r][c]);
        }

        return result;
    }

    void operator=(unique_ptr<Matrix> other) {
//        cout << "delete: " << matrix << endl;
        delete matrix;
        _declare_mem(other->row, other->col);

        for (int r = 0; r < Matrix::row; r++) {
            memcpy(matrix[r], other->matrix[r], col * sizeof(double));
        }
    }

    double mean() {
        /*!
         *  對matrix所有元素求平均值
         *  @return double result
         */
        double result = 0;

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++)
                result += Matrix::matrix[r][c];
        }

        return result / (row * col);
    }

    double sum() {
        /*!
         *  對matrix所有元素求和
         *  @return double result
         */

        double result = 0;

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++)
                result += Matrix::matrix[r][c];
        }

        return result;
    }

    double maximize() {
        double max = Matrix::matrix[0][0];

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++)
                max = (Matrix::matrix[r][c] > max) ? Matrix::matrix[r][c] : max;
        }
        return max;
    }

    void _check_shape(Matrix *other) {
        /*!
          *  檢查維度是否正確，規則如下:
          *     1. (2, 3) and (2, 3) = (2, 3)
          *     2. (2, 3) and (2, 1) = (2, 3)
          *     3. (2, 3) and (1, 3) = (2, 3)
          *     4. (2, 1) and (2, 3) = (2, 3)
          *     5. (1, 3) and (2, 3) = (2, 3)
          *
          * @param Matrix other
        */

        assert((Matrix::row == other->row && Matrix::col == other->col) ||
               (Matrix::row == other->row && (Matrix::col == 1 || other->col == 1)) ||
               ((Matrix::row == 1 || other->row == 1) && (Matrix::col == other->col)));

    }

    void random() {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                matrix[r][c] = 2 * rand() / (RAND_MAX + 1.0) - 1;
            }
        }
    }

    void fill(double value) {
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                matrix[r][c] = value;
            }
        }
    }

    void show_matrix() {
        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                cout << setw(8) << Matrix::matrix[r][c] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
};


/*!
*  Overload unique_ptr的運算符函式
 *  (1.) unique_ptr<Matrix> + Matrix
 *  (2.) unique_ptr<Matrix> + double
 *  (3.) double + Matrix
*/
template<class T>
unique_ptr<Matrix> operator+(unique_ptr<Matrix> matrix, T other){
    return  *matrix + other;
}

template<class T>
unique_ptr<Matrix> operator-(unique_ptr<Matrix> matrix, T other){
    return  *matrix - other;
}

template<class T>
unique_ptr<Matrix> operator*(unique_ptr<Matrix> matrix, T other){
    return  *matrix * other;
}

template<class T>
unique_ptr<Matrix> operator/(unique_ptr<Matrix> matrix, T other){
    return  *matrix / other;
}

unique_ptr<Matrix> operator+(double i, Matrix &matrix) {
    // 初始化結果
    auto result = std::make_unique<Matrix>(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result->matrix[r][c] = matrix.matrix[r][c] + i;
        }
    }
    return result;
}

unique_ptr<Matrix> operator-(double i, Matrix &matrix) {
    // 初始化結果
    auto result = std::make_unique<Matrix>(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result->matrix[r][c] = i - matrix.matrix[r][c];
        }
    }
    return result;
}

unique_ptr<Matrix> operator*(double i, Matrix &matrix) {
    // 初始化結果
    auto result = std::make_unique<Matrix>(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result->matrix[r][c] = matrix.matrix[r][c] * i;
        }
    }
    return result;
}

unique_ptr<Matrix> operator/(double i, Matrix &matrix) {
    // 初始化結果
    auto result = std::make_unique<Matrix>(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result->matrix[r][c] = i / matrix.matrix[r][c];
        }
    }
    return result;
}


int main() {
    Matrix a(1000, 1000, 4);
    Matrix b(1000, 1000, 4);
    Matrix c;

    clock_t  start = clock();
    c = a.dot(b);
    clock_t  end = clock();
    cout << "Run time:" << double(end-start)/CLOCKS_PER_SEC << endl;


    return 0;
}
