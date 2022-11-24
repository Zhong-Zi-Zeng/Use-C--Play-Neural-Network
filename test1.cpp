#include <cassert>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <time.h>

using namespace std;

class Matrix {
public:
    double **matrix;
    int row = 0;
    int col = 0;

    Matrix() {
        matrix = (double **) malloc(sizeof(double *) * row);
    }

    template<size_t row, size_t col>
    Matrix(double (&arr)[row][col]) {
        /*!
         * 將傳遞進來的2維陣列轉換為matrix類別
         * @tparam row
         * @tparam col
         * @param arr
         */
        this->row = row;
        this->col = col;
        matrix = (double **) malloc(sizeof(double *) * row);

        for (int r = 0; r < Matrix::row; r++) {
            matrix[r] = (double *) malloc(sizeof(double) * Matrix::col);
            for (int c = 0; c < Matrix::col; c++) {
                Matrix::matrix[r][c] = arr[r][c];
            }

        }
    }

    Matrix(int row, int col) {
        this->row = row;
        this->col = col;
        matrix = (double **) malloc(sizeof(double *) * row);
        random();
    }

    Matrix(int row, int col, double fill_value) {
        this->row = row;
        this->col = col;
        matrix = (double **) malloc(sizeof(double *) * row);
        fill(fill_value);
    }

    ~Matrix(){
        cout << "clear:";
        cout << matrix << " ";
        cout << row << " ";
        cout << col << endl;

        for (int r = 0; r < row; r++) {
            free(Matrix::matrix[r]);
        }
        free(Matrix::matrix);
    }

    Matrix operator+(Matrix other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        Matrix result(max_row, max_col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][0] + other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] + other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][c] + other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][0] + other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] + other.matrix[0][c];
                }
            }
        } else if(other.col == 1){
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] + other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] + other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator-(Matrix other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        Matrix result(max_row, max_col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][0] - other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] - other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][c] - other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][0] - other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] - other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] - other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] - other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator*(Matrix other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        Matrix result(max_row, max_col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][0] * other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] * other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][c] * other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][0] * other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] * other.matrix[0][c];
                }
            }
        } else if (other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] * other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] * other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator/(Matrix other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        Matrix result(max_row, max_col, 0);

        if (Matrix::row == 1 && Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][0] / other.matrix[r][c];
                }
            }
        } else if (other.row == 1 && other.col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] / other.matrix[0][0];
                }
            }
        } else if (Matrix::row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[0][c] / other.matrix[r][c];
                }
            }
        } else if (Matrix::col == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][0] / other.matrix[r][c];
                }
            }
        } else if (other.row == 1) {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] / other.matrix[0][c];
                }
            }
        } else if(other.col == 1){
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] / other.matrix[r][0];
                }
            }
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] / other.matrix[r][c];
                }
            }
        }

        return result;
    }

    Matrix operator+(double other) {
        // 初始化結果
        Matrix result(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result.matrix[r][c] = Matrix::matrix[r][c] + other;
            }
        }

        return result;
    }

    Matrix operator-(double other) {
        // 初始化結果
        Matrix result(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result.matrix[r][c] = Matrix::matrix[r][c] - other;
            }
        }

        return result;
    }

    Matrix operator*(double other) {
        // 初始化結果
        Matrix result(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result.matrix[r][c] = Matrix::matrix[r][c] * other;
            }
        }

        return result;
    }

    Matrix operator/(double other) {
        // 初始化結果
        Matrix result(Matrix::row, Matrix::col);

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++) {
                result.matrix[r][c] = Matrix::matrix[r][c] / other;
            }
        }

        return result;
    }

    Matrix dot(Matrix other) {
        /*!
         * 將兩個matrix進行點積運算，且在點積前檢查維度是否正確
         * @param Matrix other
         * @return Matrix
         */

        assert((Matrix::col == other.row));

        // 初始化結果
        Matrix result(Matrix::row, other.col, 0);

        for (int i = 0; i < Matrix::row; i++) {
            for (int j = 0; j < other.col; j++) {
                for (int k = 0; k < other.row; k++) {
                    result.matrix[i][j] += Matrix::matrix[i][k] * other.matrix[k][j];
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
        Matrix result(Matrix::col, Matrix::row, 0);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++)
                result.matrix[j][i] = Matrix::matrix[i][j];
        }

        return result;


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

    Matrix mean(int axis = 0) {
        /*!
         * 向指定軸求平均。
         * ex:
         *     matrix = [[1, 1, 1],
         *                [2, 1, 3]]
         *
         *     1. axis = 0
         *        result = [1.5, 1, 2]
         *
         *     2. axis = 1
         *        matrix = [1, 2]
         * @param axis
         * @return Matrix result
         */

        switch (axis) {
            case 0: {
                Matrix result(1, Matrix::col, 0);

                for (int c = 0; c < Matrix::col; c++) {
                    for (int r = 0; r < Matrix::row; r++) {
                        result.matrix[0][c] += Matrix::matrix[r][c];
                    }
                    result.matrix[0][c] /= Matrix::row;
                }
                return result;
            }

            case 1: {
                Matrix result(1, Matrix::row, 0);

                for (int r = 0; r < Matrix::row; r++) {
                    for (int c = 0; c < Matrix::col; c++) {
                        result.matrix[0][r] += Matrix::matrix[r][c];
                    }
                    result.matrix[0][r] /= Matrix::col;
                }
                return result;
            }
        }
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

    Matrix sum(int axis) {
        /*!
         * 向指定軸求和。
         * ex:
         *     matrix = [[1, 1, 1],
         *               [2, 1, 3]]
         *
         *     1. axis = 0
         *        result = [3, 2, 4]
         *
         *     2. axis = 1
         *        matrix = [3, 6]
         * @param axis
         * @return Matrix result
         */

        switch (axis) {
            case 0: {
                Matrix result(1, Matrix::col, 0);

                for (int c = 0; c < Matrix::col; c++) {
                    for (int r = 0; r < Matrix::row; r++) {
                        result.matrix[0][c] += Matrix::matrix[r][c];
                    }
                }
                return result;
            }

            case 1: {
                Matrix result(1, Matrix::row, 0);

                for (int r = 0; r < Matrix::row; r++) {
                    for (int c = 0; c < Matrix::col; c++) {
                        result.matrix[0][r] += Matrix::matrix[r][c];
                    }
                }
                return result;
            }
        }
    }

    double maximize() {
        double max = Matrix::matrix[0][0];

        for (int r = 0; r < Matrix::row; r++) {
            for (int c = 0; c < Matrix::col; c++)
                max = (Matrix::matrix[r][c] > max) ? Matrix::matrix[r][c] : max;
        }
        return max;
    }

    static Matrix ln(Matrix _matrix) {
        /*!
         * 計算傳入matrix的ln值
         * @param _matrix
         * @return Matrix result
         */
        Matrix result(_matrix.row, _matrix.col, 0);

        for (int r = 0; r < _matrix.row; r++) {
            for (int c = 0; c < _matrix.col; c++)
                result.matrix[r][c] = log(_matrix.matrix[r][c]);
        }

        return result;
    }

    static Matrix exp(Matrix _matrix) {
        /*!
         * 計算傳入matrix的exp值
         * @param _matrix
         * @return Matrix result
         */
        Matrix result(_matrix.row, _matrix.col, 0);

        for (int r = 0; r < _matrix.row; r++) {
            for (int c = 0; c < _matrix.col; c++)
                result.matrix[r][c] = std::exp(_matrix.matrix[r][c]);
        }

        return result;
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
            matrix[r] = (double *) malloc(sizeof(double) * col);
            for (int c = 0; c < col; c++) {
                matrix[r][c] = 2 * rand() / (RAND_MAX + 1.0) - 1;
            }
        }
    }

    void fill(double value) {
        for (int r = 0; r < row; r++) {
            matrix[r] = (double *) malloc(sizeof(double) * col);
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

Matrix operator+(double i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = m.matrix[r][c] + i;
        }
    }
    return result;
}

Matrix operator-(double i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = i - m.matrix[r][c];
        }
    }
    return result;
}

Matrix operator*(double i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = m.matrix[r][c] * i;
        }
    }
    return result;
}

Matrix operator/(double i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = i / m.matrix[r][c];
        }
    }
    return result;
}

// ==============================================================================
// -- 隱藏層 ---------------------------------------------------------------------
// ==============================================================================
class Layer {
public:
    int output_shape;  // 輸入維度
    int input_shape;  // 輸出維度
    bool use_bias; // 是否使用偏置值
    string layer_name; // 網路層名稱
    Matrix *w;  // Weight
    Matrix *b;  // Bias
    Matrix *x;  // 輸入
    Matrix *u;  // 未使用激活函式前的輸出
    Matrix *y;  // 使用激活函式後的輸出
    Matrix *d_w; // Weight的梯度
    Matrix *d_b; // Bias的梯度

    virtual void set_weight_bias() = 0;  // 初始化權重與偏置值

    virtual Matrix FP(Matrix x, bool training) = 0;

    virtual Matrix BP(Matrix x, Matrix label, bool training) = 0;

};

// ==============隱藏層==============
class BaseLayer : public Layer {
public:
    BaseLayer(int input_shape, int output_shape, bool use_bias = true) {
        init(input_shape, output_shape, use_bias);
    }

    BaseLayer(int output_shape, bool use_bias = true) {
        init(output_shape, use_bias);
    }

    void init(int input_shape, int output_shape, bool use_bias = true) {
        this->input_shape = input_shape;
        this->output_shape = output_shape;
        this->use_bias = use_bias;
        layer_name = "BaseLayer";
    }

    void init(int output_shape, bool use_bias = true) {
        init(input_shape, output_shape, use_bias);
    }

    void set_weight_bias() override {
        w = new Matrix(input_shape, output_shape);
        b = new Matrix(1, output_shape);
    }

    Matrix FP(Matrix x, bool training) override {}

    Matrix BP(Matrix x, Matrix label, bool training) override{}
};

class Car{
public:
    int door = 4;
    int speed = 100;
    string car_name;

    Car(string car_name){
        this->car_name = car_name;
        cout << "Create " << car_name << " " << this << endl;
    }

    ~Car(){
        cout << "Delete "<< car_name << " " << this << endl;
    }
};



int main() {
    auto **m = new double* [2];

    for(int i = 0; i < 2; i++){
        m[i] = new double [3];
    }



    return 0;
}