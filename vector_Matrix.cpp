#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <math.h>
#include <time.h>
#include <vector>

using namespace std;

typedef vector<double> _1D_MATRIX;
typedef vector<vector<double>> _2D_MATRIX;
typedef vector<vector<vector<double>>> _3D_MATRIX;
typedef vector<vector<vector<vector<double>>>> _4D_MATRIX;


class Matrix {
public:
    _2D_MATRIX matrix;
    int row = 0;
    int col = 0;

    Matrix() {}

    template<size_t row, size_t col>
    Matrix(double (&arr)[row][col]) {
        /*!
         * 將傳遞進來的2維陣列轉換為matrix類別
         * @tparam row
         * @tparam col
         * @param arr
         */
        _initial(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                matrix[r][c] = arr[r][c];
            }
        }
    }

    Matrix(int row, int col) {
        _initial(row, col);
    }

    Matrix(int row, int col, double fill_value) {
        _initial(row, col);
        fill(fill_value);
    }

    void _initial(int row, int col) {
        this->row = row;
        this->col = col;
        matrix = _2D_MATRIX(row, _1D_MATRIX(col, 0));
    }

    Matrix operator+(Matrix &other) {

        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(row, other.row);
        int max_col = max(col, other.col);

        // 初始化結果
        Matrix result(row, col);

        if (row == 1 && col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][0] + other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] + other.matrix[0][0];
                }
            }
        } else if (row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][c] + other.matrix[r][c];
                }
            }
        } else if (col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][0] + other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] + other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] + other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] + other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator+(Matrix &&other) {
        return *this + other;
    }

    Matrix operator-(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(row, other.row);
        int max_col = max(col, other.col);

        // 初始化結果
        Matrix result(row, col);

        if (row == 1 && col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][0] - other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] - other.matrix[0][0];
                }
            }
        } else if (row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][c] - other.matrix[r][c];
                }
            }
        } else if (col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][0] - other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] - other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] - other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] - other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator-(Matrix &&other) {
        return *this - other;
    }

    Matrix operator*(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(row, other.row);
        int max_col = max(col, other.col);

        // 初始化結果
        Matrix result(row, col);

        if (row == 1 && col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][0] * other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] * other.matrix[0][0];
                }
            }
        } else if (row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][c] * other.matrix[r][c];
                }
            }
        } else if (col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][0] * other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] * other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] * other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] * other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator*(Matrix &&other) {
        return *this * other;
    }

    Matrix operator/(Matrix &other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(row, other.row);
        int max_col = max(col, other.col);

        // 初始化結果
        Matrix result(row, col);

        if (row == 1 && col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][0] / other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] / other.matrix[0][0];
                }
            }
        } else if (row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[0][c] / other.matrix[r][c];
                }
            }
        } else if (col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][0] / other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] / other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] / other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = matrix[r][c] / other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator/(Matrix &&other) {
        return *this / other;
    }

    Matrix operator+(double num) {
        // 初始化結果
        Matrix result(row, col);
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result.matrix[r][c] = matrix[r][c] + num;
            }
        }
        return result;
    }

    Matrix operator-(double num) {
        // 初始化結果
        Matrix result(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result.matrix[r][c] = matrix[r][c] - num;
            }
        }

        return result;
    }

    Matrix operator*(double num) {
        // 初始化結果
        Matrix result(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result.matrix[r][c] = matrix[r][c] * num;
            }
        }

        return result;
    }

    Matrix operator/(double num) {
        // 初始化結果
        Matrix result(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result.matrix[r][c] = matrix[r][c] / num;
            }
        }

        return result;
    }

    Matrix dot(Matrix &other) {
        /*!
         * 將兩個matrix進行點積運算，且在點積前檢查維度是否正確
         * @param Matrix other
         * @return Matrix
         */

        assert((col == other.row));

        // 初始化結果
        Matrix result(row, other.col);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < other.col; j++) {
                for (int k = 0; k < other.row; k++) {
                    result.matrix[i][j] += matrix[i][k] * other.matrix[k][j];
                }
            }
        }

        return result;
    }

    Matrix dot(Matrix &&other){
        return this->dot(other);
    }

    Matrix transpose() {
        /*!
         * 將matrix進行轉置
         * @return Matrix
         */

        // 初始化結果
        Matrix result(row, col);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++)
                result.matrix[j][i] = matrix[i][j];
        }

        return result;
    }

    template<class T>
    static Matrix ln(T &&_matrix) {
        /*!
         * 計算傳入matrix的ln值
         * @param _matrix
         * @return Matrix result
         */
        // 初始化結果
        Matrix result(_matrix.row, _matrix.col);

        for (int r = 0; r < _matrix.row; r++) {
            for (int c = 0; c < _matrix.col; c++)
                result.matrix[r][c] = std::log(_matrix.matrix[r][c]);
        }

        return result;
    }

    template<class T>
    static Matrix exp(T &&_matrix) {
        /*!
         * 計算傳入matrix的exp值
         * @param _matrix
         * @return Matrix result
         */
        // 初始化結果
        Matrix result(_matrix.row, _matrix.col);

        for (int r = 0; r < _matrix.row; r++) {
            for (int c = 0; c < _matrix.col; c++)
                result.matrix[r][c] = std::exp(_matrix.matrix[r][c]);
        }

        return result;
    }

    Matrix mean(int axis, bool keep_dim = false) {
        /*!
         *  計算matrix指定軸的元素平均值，keep_dim只在axis=1時有影響。
         *  ex:
         *      axis = 1
         *
         *      matrix = [[5, 2, 2],
         *                [1, 1, 1]]
         *
         *      keep_dim = true
         *      result = [[3],
         *                [1]]
         *
         *      keep_dim = false
         *      result = [3, 1]
         *
         *  @param axis 指定軸
         *  @param keep_dim 是否保持形狀
         *  @return double result
         */

        if (axis == 0){
            Matrix result(1, col);
            for (int i = 0; i < col; i++) {
                for (int j = 0; j < row; j++) {
                    result.matrix[0][i] += matrix[j][i];
                }
                result.matrix[0][i] /= row;
            }
            return result;

        }else if(axis == 1 && keep_dim){
            Matrix result(row, 1);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.matrix[i][0] += matrix[i][j];
                }
                result.matrix[i][0] /= col;
            }
            return result;

        }else{
            Matrix result(1, row);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.matrix[0][i] += matrix[i][j];
                }
                result.matrix[0][i] /= col;
            }
            return result;
        }
    }

    double mean() {
        /*!
         *  對matrix所有元素求平均值
         *  @return double result
         */
        double result = 0;

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++)
                result += matrix[r][c];
        }

        return result / (row * col);
    }

    Matrix sum(int axis, bool keep_dim = false) {
        /*!
         *  計算matrix指定軸的元素和，keep_dim只在axis=1時有影響。
         *  ex:
         *      axis = 1
         *
         *      matrix = [[1, 2, 1],
         *                [0, 1, 1]]
         *
         *      keep_dim = true
         *      result = [[4],
         *                [2]]
         *
         *      keep_dim = false
         *      result = [4, 2]
         *
         *  @param axis 指定軸
         *  @param keep_dim 是否保持形狀
         *  @return double result
         */

        if (axis == 0){
            Matrix result(1, col);
            for (int i = 0; i < col; i++) {
                for (int j = 0; j < row; j++) {
                    result.matrix[0][i] += matrix[j][i];
                }
            }
            return result;

        }else if(axis == 1 && keep_dim){
            Matrix result(row, 1);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.matrix[i][0] += matrix[i][j];
                }
            }
            return result;

        }else{
            Matrix result(1, row);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.matrix[0][i] += matrix[i][j];
                }
            }
            return result;
        }
    }

    double sum() {
        /*!
         * 對Matrix所有元素求總和
         * @return
         */
        double result = 0;

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++)
                result += matrix[r][c];
        }

        return result;
    }

    double maximize() {
        double max = matrix[0][0];

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++)
                max = (matrix[r][c] > max) ? matrix[r][c] : max;
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

        assert((row == other->row && col == other->col) ||
               (row == other->row && (col == 1 || other->col == 1)) ||
               ((row == 1 || other->row == 1) && (col == other->col)));

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
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                cout << setw(8) << matrix[r][c] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
};


/*!
*  Overload double的運算符函式
 *  (1.) double + Matrix
*/
Matrix operator+(double i, Matrix matrix) {
    // 初始化結果
    Matrix result(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result.matrix[r][c] = matrix.matrix[r][c] + i;
        }
    }
    return result;
}

Matrix operator-(double i, Matrix matrix) {
    // 初始化結果
    Matrix result(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result.matrix[r][c] = i - matrix.matrix[r][c];
        }
    }
    return result;
}

Matrix operator*(double i, Matrix matrix) {
    // 初始化結果
    Matrix result(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result.matrix[r][c] = matrix.matrix[r][c] * i;
        }
    }
    return result;
}

Matrix operator/(double i, Matrix matrix) {
    // 初始化結果
    Matrix result(matrix.row, matrix.col);

    for (int r = 0; r < matrix.row; r++) {
        for (int c = 0; c < matrix.col; c++) {
            result.matrix[r][c] = i / matrix.matrix[r][c];
        }
    }
    return result;
}

int main() {

    Matrix a(5000, 5000, 4);
    Matrix b(5000, 5000, 4);

    clock_t  start = clock();
    Matrix d = Matrix::ln(a + 1);
    clock_t  end = clock();
    cout << "Run time:" << double(end-start)/CLOCKS_PER_SEC << endl;

    return 0;
}
