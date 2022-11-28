#include <cassert>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include <fstream>
#include<Windows.h>


using namespace std;
typedef vector<double> _1D_MATRIX;
typedef vector<vector<double>> _2D_MATRIX;
typedef vector<vector<vector<double>>> _3D_MATRIX;
typedef vector<vector<vector<vector<double>>>> _4D_MATRIX;

class Matrix {
public:
    _2D_MATRIX matrix;
    _4D_MATRIX matrix_4d;
    bool is_4d = false;
    int batch = 0;
    int channel = 0;
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
        _initial_2d(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                matrix[r][c] = arr[r][c];
            }
        }
    }

    template<size_t batch, size_t channel, size_t row, size_t col>
    Matrix(double (&arr)[batch][channel][row][col]) {
        /*!
         * 將傳遞進來的4維陣列轉換為matrix類別
         * @tparam batch
         * @tparam channel
         * @tparam col
         * @tparam col
         * @param arr
         */
        _initial_4d(batch, channel, row, col);
        for (int b = 0; b < batch; b++) {
            for (int ch = 0; ch < channel; ch++) {
                for (int r = 0; r < row; r++) {
                    for (int c = 0; c < col; c++) {
                        matrix_4d[b][ch][r][c] = arr[b][ch][r][c];
                    }
                }
            }
        }
    }

    Matrix(int batch, int channel, int row, int col) {
        _initial_4d(batch, channel, row, col);
    }

    Matrix(int row, int col) {
        _initial_2d(row, col);
    }

    Matrix(int row, int col, double fill_value) {
        _initial_2d(row, col);
        fill(fill_value);
    }

    void _initial_2d(int row, int col) {
        this->row = row;
        this->col = col;
        matrix = _2D_MATRIX(row, _1D_MATRIX(col, 0));
    }

    void _initial_4d(int batch, int channel, int row, int col) {
        this->batch = batch;
        this->channel = channel;
        this->row = row;
        this->col = col;
        this->is_4d = true;
        matrix_4d = _4D_MATRIX(batch, _3D_MATRIX(channel, _2D_MATRIX(row, _1D_MATRIX(col, 0))));
    };

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

    Matrix dot(Matrix &&other) {
        return this->dot(other);
    }

    Matrix transpose() {
        /*!
         * 將matrix進行轉置
         * @return Matrix
         */

        // 初始化結果
        Matrix result(col, row);

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

        if (axis == 0) {
            Matrix result(1, col);
            for (int i = 0; i < col; i++) {
                for (int j = 0; j < row; j++) {
                    result.matrix[0][i] += matrix[j][i];
                }
                result.matrix[0][i] /= row;
            }
            return result;

        } else if (axis == 1 && keep_dim) {
            Matrix result(row, 1);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.matrix[i][0] += matrix[i][j];
                }
                result.matrix[i][0] /= col;
            }
            return result;

        } else {
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

        if (axis == 0) {
            Matrix result(1, col);
            for (int i = 0; i < col; i++) {
                for (int j = 0; j < row; j++) {
                    result.matrix[0][i] += matrix[j][i];
                }
            }
            return result;

        } else if (axis == 1 && keep_dim) {
            Matrix result(row, 1);
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result.matrix[i][0] += matrix[i][j];
                }
            }
            return result;

        } else {
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
        if (is_4d) {
            for (int b = 0; b < batch; b++) {
                for (int ch = 0; ch < channel; ch++) {
                    for (int r = 0; r < row; r++) {
                        for (int c = 0; c < col; c++) {
                            cout << matrix_4d[b][ch][r][c] << " ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;
            }

        } else {
            for (int r = 0; r < row; r++) {
                for (int c = 0; c < col; c++) {
                    cout << setw(2) << matrix[r][c] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
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


Matrix im2col(Matrix img, int k_size, int output_height, int output_width, int stride = 1) {

    Matrix result(img.channel * k_size * k_size, img.batch * output_height * output_width);

    for (int c = 0; c < result.col; c++) {
        for (int r = 0; r < result.row; r++) {
            int img_batch = c / (output_height * output_width); // 目前到哪張圖
            int img_channel = r / (k_size * k_size); // 目前到哪個通道
            int img_r =
                    (r / k_size) % k_size + (c / output_width) % output_width + (c / stride) % stride; // 對應到原圖的row座標
            int img_c = r % k_size + c % output_height + c % stride;  // 對應到原圖的col座標
            result.matrix[r][c] = img.matrix_4d[img_batch][img_channel][img_r][img_c];
        }
    }

    return result;
};

Matrix find_maximize(Matrix &x, int output_channel, int output_height, int output_width, int k_size){
    Matrix result(x.row / (k_size * k_size), x.col);

    for (int c=0; c < x.col; c ++){
        double max_num = x.matrix[0][c];

        for (int r=0; r < x.row; r++){
            if (x.matrix[r][c] > max_num){
                max_num = x.matrix[r][c];
            }

            if ((r + 1) % (k_size * k_size) == 0){
                max_num = x.matrix[r][c];
                result.matrix[r / (k_size * k_size)][c] = max_num;
            }
        }
    }

    return result;
}

Matrix reshape_4d(Matrix &img, int output_channel, int output_height, int output_width, int k_size) {
    /** @brief 將點積完的二維矩陣reshape回四維
      * @param img 二維圖像 (filters, batch * output_height * output_width)
      * @return 轉換完畢後的四維圖片. (batch_size, channel, output_height, output_width).
      * */

    int batch_size = img.col / (output_height * output_width);
    Matrix result(batch_size, output_channel, output_height, output_width);

    for (int r = 0; r < img.row; r++) {
        for (int c = 0; c < img.col; c++) {
            int img_batch = c / (output_height * output_width); // 對應到四維圖像的哪張圖
            int img_r = (c / (int) output_height) % (int) output_height; // 對應到四維圖像的row座標
            int img_c = c % (int) output_width; // 對應到四維圖像的col座標

            result.matrix_4d[img_batch][r][img_r][img_c] = img.matrix[r][c];
        }
    }

    return result;
};

int main() {
    // 訓練資料
    double img[1][2][4][4] = {{{{0, 1, 2, 3},
                                {4, 5, 6, 7},
                                {8, 9, 10, 11},
                                {12, 13, 14, 15}},

                               {{16, 17, 18, 19},
                                {20, 21, 22, 23},
                                {24, 25, 26, 27},
                                {28, 29, 30, 31}}}};


    Matrix train_x_matrix(img);

    // 轉換成2維矩陣形式
    Matrix x = im2col(train_x_matrix, 2, 2, 2, 2);
    x.show_matrix();

    // 挑出最大值
    Matrix r = find_maximize(x, 2, 2, 2, 2);
    r.show_matrix();

    // 再轉為4d
    Matrix r_4d = reshape_4d(r, 2, 2, 2, 2);
    r_4d.show_matrix();



    return 0;
}