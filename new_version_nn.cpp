#include <cassert>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <fstream>


using namespace std;
typedef vector<double> _1D_MATRIX;
typedef vector<vector<double>> _2D_MATRIX;
typedef vector<vector<vector<double>>> _3D_MATRIX;
typedef vector<vector<vector<vector<double>>>> _4D_MATRIX;

void show_1d_matrix(_1D_MATRIX matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        cout << matrix[i] << " ";
    }
    cout << endl;
}

// ==============================================================================
// -- Matrix -----------------------------------------------------------------------
// ==============================================================================
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
                matrix[r][c] = (2 * rand() / (RAND_MAX + 1.0) - 1) / 20.;
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
                    cout << setw(8) << matrix[r][c] << " ";
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


// ==============================================================================
// -- 讀取影像、Label -------------------------------------------------------------
// ==============================================================================
void isFileExists(std::ifstream &file, string file_name) {
    if (!file.good()) {
        cout << file_name << "File dose not exist." << endl;
        exit(0);
    }
}

Matrix *load_images(string file_name, int length) {
    char pix[784]; // 用來暫存二進制資料
    Matrix *img_matrix = new Matrix(length, 1, 28, 28); // 存放影像
    std::ifstream file;
    file.open(file_name, ios::binary); // 用二進制方式讀取檔案
    isFileExists(file, file_name);  // 確認檔案是否存在

    // 先拿出前面16bytes不必要的資訊
    char p[16];
    file.read(p, 16);

    // 讀取影像
    for (int b = 0; b < length; b++) {
        file.read(pix, 784);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                img_matrix->matrix_4d[b][0][r][c] = (unsigned char) pix[r * 28 + c] / 255.;
            }
        }
    }

    // 關閉檔案
    file.close();

    return img_matrix;
}

Matrix *load_label(string file_name, int length) {
    char label[1];// 用來暫存二進制資料
    Matrix *label_matrix = new Matrix(length, 1);  // 存放label
    std::ifstream file;
    file.open(file_name, ios::binary); // 用二進制方式讀取檔案
    isFileExists(file, file_name); // 確認檔案是否存在

    // 先拿出前面8bytes不必要的資訊
    char p[8];
    file.read(p, 8);

    // 讀取label
    for (int i = 0; i < length; i++) {
        file.read(label, 1);
        label_matrix->matrix[i][0] = (unsigned char) label[0];
    }

    // 關閉檔案
    file.close();

    return label_matrix;
}

Matrix *one_hot_code(Matrix &label) {
    Matrix *one_hot_code_matrix = new Matrix(label.row, 10);

    for (int i = 0; i < label.row; i++) {
        one_hot_code_matrix->matrix[i][label.matrix[i][0]] = 1.;
    }
    return one_hot_code_matrix;
}

// ==============================================================================
// -- 激活函式抽象類 --------------------------------------------------------------
// ==============================================================================
class ActivationFunc {
public:
    virtual Matrix undiff(Matrix &m) = 0;

    virtual Matrix diff(Matrix &m, Matrix &label) = 0;
};

// ==============sigmoid==============
class sigmoid : public ActivationFunc {
public:
    Matrix undiff(Matrix &m) override {
        Matrix result = 1. / (1 + Matrix::exp(-1 * m));

        return result;
    }

    Matrix diff(Matrix &m, Matrix &label) override {
        Matrix y = undiff(m);
        Matrix result = y * (1. - y);

        return result;
    }

};

// ==============relu==============
class relu : public ActivationFunc {
public:
    Matrix undiff(Matrix &m) override {
        Matrix result(m.row, m.col);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.col; c++) {
                result.matrix[r][c] = (m.matrix[r][c] > 0) ? m.matrix[r][c] : 0;
            }
        }

        return result;
    }

    Matrix diff(Matrix &m, Matrix &label) override {
        Matrix result(m.row, m.col);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.col; c++) {
                result.matrix[r][c] = (m.matrix[r][c] > 0) ? 1 : 0;
            }
        }
        return result;
    }
};

// ==============linear==============
class linear : public ActivationFunc {
public:
    Matrix undiff(Matrix &m) override {
        return m;
    }

    Matrix diff(Matrix &m, Matrix &label) override {
        return Matrix(m.row, m.col, 1);
    }
};

// ==============softmax==============
class softmax : public ActivationFunc {
public:
    Matrix undiff(Matrix &m) override {
        // 把所有元素減去最大值
        double max_num = m.maximize();
        Matrix new_m = m - max_num;

        // 對每一列求exp總和
        Matrix exp_sum = Matrix::exp(new_m).sum(1, true);

        // 將所有元素都以exp為底
        Matrix exp_m = Matrix::exp(new_m);

        // 將每一列都除上剛剛的exp_sum
        Matrix result(exp_m.row, exp_m.col);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.col; c++) {
                result.matrix[r][c] = exp_m.matrix[r][c] / exp_sum.matrix[r][0];
            }
        }
        return result;
    }

    Matrix diff(Matrix &m, Matrix &label) override {
        /*
         * 使用softmax的話，其反向傳播公式如下:
         *
         * if i == j:
         *     yi x (1 - yi)
         *
         * else:
         *     -yi x yj
         *
         */
        // 先求回softmax後的結果
        Matrix y = undiff(m);
        int row_size = label.row;
        int col_size = label.col;
        Matrix result(row_size, col_size);

        for (int r = 0; r < row_size; r++) {
            // 先找出label所對應的類別
            int maximize_index = 0;
            for (int l = 0; l < col_size; l++) {
                if (label.matrix[r][l] == 1) {
                    maximize_index = l;
                    break;
                }
            }

            // 判斷目前softmax輸出的索引是否跟label一樣
            for (int c = 0; c < col_size; c++) {
                if (c == maximize_index) {
                    result.matrix[r][c] = y.matrix[r][c] * (1. - y.matrix[r][c]);
                } else {
                    result.matrix[r][c] = -y.matrix[r][c] * y.matrix[r][maximize_index];
                }
            }
        }

        return result;
    }

};

// ==============================================================================
// -- 損失函式 --------------------------------------------------------------------
// ==============================================================================
class LossFunc {
public:
    virtual double undiff(Matrix pre, Matrix label) = 0;

    virtual Matrix diff(Matrix pre, Matrix label) = 0;

};

// ==============均方誤差損失函式==============
class MSE : public LossFunc {
public:
    double undiff(Matrix pre, Matrix label) override {
        Matrix o1 = pre - label;
        Matrix o2 = o1 * o1;
        double o3 = o2.sum() / 2;

        return o3;
    }

    Matrix diff(Matrix pre, Matrix label) override {
        Matrix o1 = pre - label;
        return o1;
    }
};

// ==============Binary cross entropy==============
class Binary_cross_entropy : public LossFunc {
public:
    double undiff(Matrix pre, Matrix label) override {
        /*
         * 公式如下：
         * -Di * ln(Yi) - (1 - Di) * ln(1 - Yi)
         */
        Matrix left_loss = -1 * label * Matrix::ln(pre);
        Matrix right_loss = (1 - label) * Matrix::ln(1 - pre);

        Matrix loss_mat = left_loss - right_loss;
        double loss = loss_mat.sum();

        return loss;
    }

    Matrix diff(Matrix pre, Matrix label) override {
        /*
         * 公式如下：
         * (Yi - Di) / [Yi * (1 - Yi)]
         */

        Matrix loss_mat = (pre - label) / (pre * (1 - pre));
        return loss_mat;
    }
};

// ==============Categorical cross entropy==============
class Categorical_crosse_entropy : public LossFunc {
    double undiff(Matrix pre, Matrix label) override {
        /*
         * 公式如下 (Add 1e-7 Avoid ln(0)):
         * - sum(Di * ln(Yi + 0.0000001))
         */
        Matrix loss_mat = label * Matrix::ln(pre + 1e-7);
        double loss = loss_mat.sum();

        return loss;
    }

    Matrix diff(Matrix pre, Matrix label) override {
        int row_size = label.row;
        int col_size = label.col;
        Matrix result(row_size, col_size);

        for (int r = 0; r < row_size; r++) {
            // 用來存放label對應的類別
            int cls;

            // 找出label對應的類別
            for (int c = 0; c < col_size; c++) {
                if (label.matrix[r][c] == 1) {
                    cls = c;
                    break;
                }
            }

            // -ln(yi) 微分結果剛好等於 - 1 / yi，把每一列都填入此數值
            for (int c = 0; c < col_size; c++) {
                result.matrix[r][c] = -1 / pre.matrix[r][cls];
            }
        }

        return result;
    }
};

// ==============================================================================
// -- 神經網路層 ---------------------------------------------------------------------
// ==============================================================================
class Layer {
public:
    _1D_MATRIX input_shape;  // 輸入維度 1. 純數值 2. (C, H, W)
    _1D_MATRIX output_shape;  // 輸出維度 1. 純數值 2. (C, H, W)
    int filters; // 卷積核數量
    int k_size; // 卷積核大小
    int stride; // 步長
    bool use_bias; // 是否使用偏置值
    string layer_name; // 網路層名稱
    ActivationFunc *activation; // 激活函式
    Matrix w;  // Weight
    Matrix b;  // Bias
    Matrix x;  // 輸入
    Matrix u;  // 未使用激活函式前的輸出
    Matrix y;  // 使用激活函式後的輸出
    Matrix d_w; // Weight的梯度
    Matrix d_b; // Bias的梯度

    virtual void set_weight_bias() = 0;  // 初始化權重與偏置值

    virtual Matrix FP(Matrix x, bool training) = 0;

    virtual Matrix BP(Matrix x, Matrix label, bool training) = 0;

};

// ==============隱藏層==============
class BaseLayer : public Layer {
public:
    BaseLayer(int input_shape, int output_shape, ActivationFunc *activation, bool use_bias = true) {
        init(_1D_MATRIX{(double) input_shape}, _1D_MATRIX{(double) output_shape}, activation, use_bias);
    }

    BaseLayer(int output_shape, ActivationFunc *activation, bool use_bias = true) {
        init(_1D_MATRIX{(double) output_shape}, activation, use_bias);
    }

    void init(_1D_MATRIX input_shape, _1D_MATRIX output_shape, ActivationFunc *activation = new sigmoid,
              bool use_bias = true) {
        this->input_shape = input_shape;
        this->output_shape = output_shape;
        this->use_bias = use_bias;
        this->activation = activation;
        layer_name = "BaseLayer";
    }

    void init(_1D_MATRIX output_shape, ActivationFunc *activation, bool use_bias = true) {
        init(input_shape, output_shape, activation, use_bias);
    }

    void set_weight_bias() override {
        w = Matrix(input_shape[0], output_shape[0]);
        b = Matrix(1, output_shape[0]);
        w.random();
        b.random();
    }

    Matrix FP(Matrix x, bool training) override {
        this->x = x;
        u = x.dot(w);

        if (use_bias) {
            u = u + b;
        }

        y = (*activation).undiff(u);

        return y;
    }

    Matrix BP(Matrix delta, Matrix label, bool training) override {
        delta = delta * (*activation).diff(u, label);
        d_w = x.transpose().dot(delta);
        d_b = delta.sum(0);
        Matrix d_x = delta.dot(w.transpose());

        return d_x;
    }
};

// ==============Dropout層==============
class Dropout : public Layer {
public:
    double prob; // 丟棄概率

    Dropout(double prob) {
        this->prob = prob;
        layer_name = "DropoutLayer";
    }

    void set_weight_bias() override {
        this->output_shape = input_shape;
    }

    Matrix FP(Matrix x, bool training) override {
        this->x = x;

        // 如果不是在訓練過程，則直接返回輸入值
        if (!training) {
            x = x * (1. - prob);
            return x;
        } else {
            // 初始化權重
            if (w.row == 0) {
                // 取得輸入x的shape
                w = Matrix(x.row, x.col, 0);
            }

            // 設置權重，若隨機數小於設置概率則將權重設為0，否則為1
            for (int r = 0; r < w.row; r++) {
                for (int c = 0; c < w.col; c++) {
                    double rand_num = rand() / (RAND_MAX + 1.0);
                    w.matrix[r][c] = (rand_num < prob) ? 0 : 1;
                }
            }
            // 將輸入與w相乘
            y = x * w;
            return y;
        }
    }

    Matrix BP(Matrix delta, Matrix label, bool training) override {
        if (training == false) {
            return delta;
        } else {
            return delta * w;
        }
    }
};

// ==============ConvolutionLayer層==============
class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer(_1D_MATRIX input_shape, int filters, int k_size, int stride, ActivationFunc *activation,
                     bool use_bias = true) {
        init(input_shape, filters, k_size, stride, activation, use_bias);
    };

    ConvolutionLayer(int filters, int k_size, int stride, ActivationFunc *activation, bool use_bias = true) {
        init(filters, k_size, stride, activation, use_bias);
    }

    void init(_1D_MATRIX input_shape, int filters, int k_size, int stride, ActivationFunc *activation,
              bool use_bias = true) {
        this->input_shape = input_shape;
        this->filters = filters;
        this->k_size = k_size;
        this->stride = stride;
        this->activation = activation;
        this->use_bias = use_bias;
    };

    void init(int filters, int k_size, int stride, ActivationFunc *activation, bool use_bias = true) {
        init(input_shape, filters, k_size, stride, activation, use_bias);
    };

    void set_weight_bias() override {
        double output_height = (input_shape[1] - k_size) / stride + 1;
        double output_width = (input_shape[2] - k_size) / stride + 1;
        this->output_shape = _1D_MATRIX{(double) filters, output_height, output_width};

        w = Matrix(filters, input_shape[0] * k_size * k_size);
        b = Matrix(filters, 1);
        w.random();
        b.random();
    };

    Matrix FP(Matrix x, bool training) override {
        this->x = im2col(x);

        u = w.dot(this->x);

        if (use_bias) {
            u = u + b;
        }

        y = (*activation).undiff(u);

        return reshape_4d(y);
    };

    Matrix BP(Matrix x, Matrix label, bool training) override {
        Matrix delta_2d = reshape_2d(x);

        delta_2d = delta_2d * (*activation).diff(u, label);

        d_w = delta_2d.dot(this->x.transpose());
        d_b = delta_2d.sum(1, true);

        delta_2d = w.transpose().dot(delta_2d);
        delta_2d = col2im(delta_2d);

        return delta_2d;
    };

    Matrix im2col(Matrix &img) {
        /** @brief 將輸入的四維圖片轉為二維矩陣形式.
          * @param img 四維圖像 (B, C, H, W)
          * @return 轉換完畢後的二維矩陣. */

        Matrix result(img.channel * k_size * k_size, img.batch * output_shape[1] * output_shape[2]);

        for (int c = 0; c < result.col; c++) {
            for (int r = 0; r < result.row; r++) {
                int img_batch = c / (output_shape[1] * output_shape[2]); // 目前到哪張圖
                int img_channel = r / (k_size * k_size); // 目前到哪個通道
                int img_r = (r / k_size) % k_size + (c / (int) output_shape[2]) % (int) output_shape[2] +
                            (c / stride) % stride; // 對應到原圖的row座標
                int img_c = r % k_size + c % (int) output_shape[1] + c % stride;  // 對應到原圖的col座標
                result.matrix[r][c] = img.matrix_4d[img_batch][img_channel][img_r][img_c];
            }
        }
        return result;
    };

    Matrix col2im(Matrix &delta) {
        /** @brief 將輸入的二維圖片轉為四維矩陣形式.
          * @param img 二維梯度圖像 (channel * k_size * k_size, batch_size * output_height * output_width)
          * @return 轉換完畢後的四維矩陣. (batch, channel, output_height, output_width)*/
        int batch_size = delta.col / (output_shape[1] * output_shape[2]);
        Matrix result(batch_size, input_shape[0], input_shape[1], input_shape[2]);

        for (int c = 0; c < delta.col; c++) {
            for (int r = 0; r < delta.row; r++) {
                int img_batch = c / (output_shape[1] * output_shape[2]); // 目前到哪張圖
                int img_channel = r / (k_size * k_size); // 目前到哪個通道
                int img_r = (r / k_size) % k_size + (c / (int) output_shape[2]) % (int) output_shape[2] +
                            (c / stride) % stride; // 對應到原圖的row座標
                int img_c = r % k_size + c % (int) output_shape[1] + c % stride;  // 對應到原圖的col座標

                result.matrix_4d[img_batch][img_channel][img_r][img_c] = delta.matrix[r][c];
            }
        }

        return result;
    };

    Matrix reshape_4d(Matrix &img) {
        /** @brief 將點積完的二維矩陣reshape回四維
          * @param img 二維圖像 (filters, batch * output_height * output_width)
          * @return 轉換完畢後的四維圖片. (batch_size, channel, output_height, output_width).
          * */

        int batch_size = img.col / (output_shape[1] * output_shape[2]);
        Matrix result(batch_size, filters, output_shape[1], output_shape[2]);

        for (int r = 0; r < img.row; r++) {
            for (int c = 0; c < img.col; c++) {
                int img_batch = c / (output_shape[1] * output_shape[2]); // 對應到四維圖像的哪張圖
                int img_r = (c / (int) output_shape[1]) % (int) output_shape[1]; // 對應到四維圖像的row座標
                int img_c = c % (int) output_shape[2]; // 對應到四維圖像的col座標

                result.matrix_4d[img_batch][r][img_r][img_c] = img.matrix[r][c];
            }
        }

        return result;
    };

    Matrix reshape_2d(Matrix &delta) {
        /** @brief 將四維矩陣reshape回二維
          * @param delta 四維梯度圖像 (batch, filters, output_height, output_width)
          * @return 轉換完畢後的二維梯度圖像. (filters, batch * output_height *output_width).
          * */

        Matrix result(filters, delta.batch * delta.row * delta.col);

        for (int r = 0; r < result.row; r++) {
            for (int c = 0; c < result.col; c++) {
                int img_batch = c / (delta.col * delta.row); // 對應到四維圖像的哪張圖
                int img_r = (c / delta.row) % delta.row; // 對應到四維圖像的row座標
                int img_c = c % delta.col; // 對應到四維圖像的col座標
                result.matrix[r][c] = delta.matrix_4d[img_batch][r][img_r][img_c];
            }
        }

        return result;
    };
};

// ==============Flatten層==============
class FlattenLayer : public Layer {
public:
    void set_weight_bias() override {
        double temp = 1;
        for (int i = 0; i < input_shape.size(); i++) {
            temp *= input_shape[i];
        }
        this->output_shape = _1D_MATRIX{temp};
        layer_name = "FlattenLayer";
    };

    Matrix FP(Matrix x, bool training) override {
        Matrix result(x.batch, x.channel * x.row * x.col);

        for (int b = 0; b < x.batch; b++) {
            for (int ch = 0; ch < x.channel; ch++) {
                for (int r = 0; r < x.row; r++) {
                    for (int c = 0; c < x.col; c++) {
                        result.matrix[b][ch * x.row * x.col + r * x.row + c % x.col] = x.matrix_4d[b][ch][r][c];
                    }
                }
            }
        }

        return result;
    };

    Matrix BP(Matrix x, Matrix label, bool training) override {
        Matrix result(x.row, input_shape[0], input_shape[1], input_shape[2]);

        for (int b = 0; b < result.batch; b++) {
            for (int ch = 0; ch < result.channel; ch++) {
                for (int r = 0; r < result.row; r++) {
                    for (int c = 0; c < result.col; c++) {
                        result.matrix_4d[b][ch][r][c] = x.matrix[b][ch * result.row * result.col + r * result.row +
                                                                    c % result.col];
                    }
                }
            }
        }

        return result;
    };
};

// ==============MaxPooling層==============
class MaxpoolingLayer : public Layer {
public:
    Matrix max_matrix; // 用來記錄最大值的位置

    MaxpoolingLayer(int k_size) {
        this->k_size = k_size;
        this->stride = k_size;
        this->layer_name = "MaxpoolingLayer";
    }

    void set_weight_bias() override {
        int output_height = ((int) input_shape[1] % k_size == 0) ? input_shape[1] / k_size : input_shape[1] / k_size +
                                                                                             1;
        int output_width = ((int) input_shape[2] % k_size == 0) ? input_shape[2] / k_size : input_shape[2] / k_size + 1;
        this->output_shape = _1D_MATRIX{input_shape[0], (double) output_height, (double) output_width};
        this->filters = input_shape[0];

    }

    Matrix FP(Matrix x, bool training) override {
        this->x = im2col(x);

        this->max_matrix = Matrix(this->x.row, this->x.col);
        Matrix result = find_maximize(this->x);

        return reshape_4d(result);
    }

    Matrix BP(Matrix x, Matrix label, bool training) override {
        Matrix delta_2d = reshape_2d(x);

        Matrix result(this->x.row, this->x.col);

        for (int c = 0; c < max_matrix.col; c++) {
            for (int r = 0; r < max_matrix.row; r++) {
                if (max_matrix.matrix[r][c] == 1) {
                    result.matrix[r][c] = delta_2d.matrix[r / (k_size * k_size)][c];
                }
            }
        }
        result = col2im(result);

        return result;
    }

    Matrix im2col(Matrix &img) {
        /** @brief 將輸入的四維圖片轉為二維矩陣形式.
          * @param img 四維圖像 (B, C, H, W)
          * @return 轉換完畢後的二維矩陣. */

        Matrix result(img.channel * k_size * k_size, img.batch * output_shape[1] * output_shape[2]);
        for (int c = 0; c < result.col; c++) {
            for (int r = 0; r < result.row; r++) {
                int img_batch = c / (output_shape[1] * output_shape[2]); // 目前到哪張圖
                int img_channel = r / (k_size * k_size); // 目前到哪個通道
                int img_r = (r / k_size) % k_size + (c / (int) output_shape[2]) % (int) output_shape[2] +
                            (c / stride) % stride; // 對應到原圖的row座標
                int img_c = r % k_size + c % (int) output_shape[1] + c % stride;  // 對應到原圖的col座標

                if (img_batch >= img.batch || img_channel >= img.channel || img_r >= img.row || img_c >= img.col) {
                    continue;
                }
                result.matrix[r][c] = img.matrix_4d[img_batch][img_channel][img_r][img_c];
            }
        }
        return result;
    };

    Matrix col2im(Matrix &delta) {
        /** @brief 將輸入的二維圖片轉為四維矩陣形式.
          * @param img 二維梯度圖像 (channel * k_size * k_size, batch_size * output_height * output_width)
          * @return 轉換完畢後的四維矩陣. (batch, channel, output_height, output_width)*/
        int batch_size = delta.col / (output_shape[1] * output_shape[2]);
        Matrix result(batch_size, input_shape[0], input_shape[1], input_shape[2]);

        for (int c = 0; c < delta.col; c++) {
            for (int r = 0; r < delta.row; r++) {
                int img_batch = c / (output_shape[1] * output_shape[2]); // 目前到哪張圖
                int img_channel = r / (k_size * k_size); // 目前到哪個通道
                int img_r = (r / k_size) % k_size + (c / (int) output_shape[2]) % (int) output_shape[2] +
                            (c / stride) % stride; // 對應到原圖的row座標
                int img_c = r % k_size + c % (int) output_shape[1] + c % stride;  // 對應到原圖的col座標

//                if (img_batch >= result.batch || img_channel >= result.channel || img_r >= result.row || img_c >= result.col){
//                    continue;
//                }

                result.matrix_4d[img_batch][img_channel][img_r][img_c] = delta.matrix[r][c];
            }
        }

        return result;
    };

    Matrix reshape_4d(Matrix &img) {
        /** @brief 將點積完的二維矩陣reshape回四維
          * @param img 二維圖像 (filters, batch * output_height * output_width)
          * @return 轉換完畢後的四維圖片. (batch_size, channel, output_height, output_width).
          * */

        int batch_size = img.col / (output_shape[1] * output_shape[2]);
        Matrix result(batch_size, filters, output_shape[1], output_shape[2]);

        for (int r = 0; r < img.row; r++) {
            for (int c = 0; c < img.col; c++) {
                int img_batch = c / (output_shape[1] * output_shape[2]); // 對應到四維圖像的哪張圖
                int img_r = (c / (int) output_shape[1]) % (int) output_shape[1]; // 對應到四維圖像的row座標
                int img_c = c % (int) output_shape[2]; // 對應到四維圖像的col座標

                result.matrix_4d[img_batch][r][img_r][img_c] = img.matrix[r][c];
            }
        }

        return result;
    };

    Matrix reshape_2d(Matrix &delta) {
        /** @brief 將四維矩陣reshape回二維
          * @param delta 四維梯度圖像 (batch, filters, output_height, output_width)
          * @return 轉換完畢後的二維梯度圖像. (filters, batch * output_height *output_width).
          * */

        Matrix result(filters, delta.batch * delta.row * delta.col);

        for (int r = 0; r < result.row; r++) {
            for (int c = 0; c < result.col; c++) {
                int img_batch = c / (delta.col * delta.row); // 對應到四維圖像的哪張圖
                int img_r = (c / delta.row) % delta.row; // 對應到四維圖像的row座標
                int img_c = c % delta.col; // 對應到四維圖像的col座標
                result.matrix[r][c] = delta.matrix_4d[img_batch][r][img_r][img_c];
            }
        }

        return result;
    };

    Matrix find_maximize(Matrix &x) {
        /**
        * @brief 從轉換好的二維影像矩陣中，找出最大值與最大值的位置
        * @param img 二維梯度圖像 (channel * k_size * k_size, batch_size * output_height * output_width)
        * @return Maxpooling完後的四維矩陣. (batch, channel, output_height, output_width)*/
        Matrix result(x.row / (k_size * k_size), x.col);

        double max_num = -100;
        int max_r, max_c;

        for (int c = 0; c < x.col; c++) {
            for (int r = 0; r < x.row; r++) {
                if (x.matrix[r][c] > max_num) {
                    max_num = x.matrix[r][c];
                    max_r = r;
                    max_c = c;
                }

                if ((r + 1) % (k_size * k_size) == 0) {
                    result.matrix[r / (k_size * k_size)][c] = max_num;
                    max_matrix.matrix[max_r][max_c] = 1;
                    max_num = -100;
                }
            }
        }

        return result;
    }

};


// ========================================================================
// -- 優化器 ---------------------------------------------------------------------
// ==============================================================================
class Optimizer {
public:
    double learning_rate;

    virtual void gradient_decent(vector<Layer *> layer_list, int batch_size) = 0;
};

// ==============SGD==============
class SGD : public Optimizer {
public:
    SGD(double learning_rate) {
        this->learning_rate = learning_rate;
    }

    void gradient_decent(vector<Layer *> layer_list, int batch_size) override {
        /* 更新公式如下：
         * Wt = Wt - learning_rate * d_w
         */
        for (int i = 0; i < layer_list.size(); i++) {
            // 跳過dropout、flatten、MaxpoolingLayer層
            if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer" ||
                layer_list[i]->layer_name == "MaxpoolingLayer") {
                continue;
            }
            layer_list[i]->w = layer_list[i]->w - learning_rate * layer_list[i]->d_w / (double) batch_size;
            layer_list[i]->b = layer_list[i]->b - learning_rate * layer_list[i]->d_b / (double) batch_size;
        }
    }
};

// ==============Momentum==============
class Momentum : public Optimizer {
public:
    double beta = 0.9;  // Beta 為常數，通常設定為0.9
    bool initial_flag = true;
    vector<Matrix> last_v_w;  // 用來存放上一次weight的慣性
    vector<Matrix> last_v_b;  // 用來存放上一次bias的慣性

    Momentum(double learning_rate, double beta) {
        this->learning_rate = learning_rate;
        this->beta = beta;
    }

    void gradient_decent(vector<Layer *> layer_list, int batch_size) override {
        // 第一次進來前先初始化last_v_w和last_v_b
        if (initial_flag) {
            initial_flag = false;

            for (int i = 0; i < layer_list.size(); i++) {
                // 若是遇到dropout、Flatten、MaxpoolingLayer層，則加一個空陣列，方便後面計算
                if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer" ||
                    layer_list[i]->layer_name == "MaxpoolingLayer") {
                    last_v_w.emplace_back(Matrix(0, 0));
                    last_v_b.emplace_back(Matrix(0, 0));
                    continue;
                }

                int weight_row_size = layer_list[i]->w.row;
                int weight_col_size = layer_list[i]->w.col;
                int bias_row_size = layer_list[i]->b.row;
                int bias_col_size = layer_list[i]->b.col;

                last_v_w.emplace_back(Matrix(weight_row_size, weight_col_size));
                last_v_b.emplace_back(Matrix(bias_row_size, bias_col_size));
            }
        }

        // 更新梯度
        for (int i = 0; i < layer_list.size(); i++) {
            /* 更新公式如下：
             * Vt = Beta * Vt-1 - learning_rate * d_w
             * W = W + Vt
             */

            // 跳過dropout、flatten、MaxpoolingLayer層
            if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer" ||
                layer_list[i]->layer_name == "MaxpoolingLayer") {
                continue;
            }

            Matrix V_w_t = last_v_w[i] * beta - learning_rate * layer_list[i]->d_w / (double) batch_size;
            last_v_w[i] = V_w_t;
            layer_list[i]->w = layer_list[i]->w + V_w_t;

            Matrix V_b_t = last_v_b[i] * beta - learning_rate * layer_list[i]->d_b / (double) batch_size;
            last_v_b[i] = V_b_t;
            layer_list[i]->b = layer_list[i]->b + V_b_t;
        }
    }

};

// ==============================================================================
// -- 序列模型 -------------------------------------------------------------------
// ====================================================================================
class Sequential {
public:
    int epoch;  // 訓練次數
    int batch_size;  // 批量大小
    int layer_length; // 網路層數量
    LossFunc *loss; // 損失函式
    Optimizer *opt;  // 優化器
    vector<Layer *> layer_list;  // 存放網路層

    Sequential(int epoch, int batch_size, LossFunc *loss, Optimizer *opt) {
        this->epoch = epoch;
        this->batch_size = batch_size;
        this->loss = loss;
        this->opt = opt;
    };

    // 增加層數
    void add(Layer *layer) {
        layer_list.push_back(layer);
    }

    // 設置所有層數的權重
    void compile() {
        layer_length = layer_list.size();
        for (int i = 0; i < layer_length; i++) {
            layer_list[i]->set_weight_bias();
            if (i + 1 < layer_length) {
                layer_list[i + 1]->input_shape = layer_list[i]->output_shape;
            }
        }
    };

    // 訓練
    void fit(Matrix &train_x, Matrix &train_y) {
        for (int e = 0; e < epoch; e++) {
            Matrix batch_x;
            Matrix batch_y;
            Matrix output;

            if (train_x.is_4d) {
                for (int b = 0; b < train_x.batch; b += batch_size) {
                    batch_x = get_batch_data(train_x, b,
                                             min((int) b + batch_size, (int) train_x.batch));
                    batch_y = get_batch_data(train_y, b,
                                             min((int) b + batch_size, (int) train_y.row));
                    // 前向傳播
                    output = FP(batch_x);

                    // 反向傳播
                    BP(output, batch_y);

                    // 梯度更新
                    update_weight();

                    // 顯示訓練進度
                    double loss_ = (*loss).undiff(output, batch_y) / batch_size;
                    printf("\rnow epoch:%4d, loss:%3.4f, progress:%4.4lf%%", e, loss_, b * 100 / (train_x.batch - 1.));
                }
            } else {
                for (int b = 0; b < train_x.row; b += batch_size) {
                    // 每次訓練讀取batch size 的資料去訓練
                    batch_x = get_batch_data(train_x, b,
                                             min((int) b + batch_size, (int) train_x.row));
                    batch_y = get_batch_data(train_y, b,
                                             min((int) b + batch_size, (int) train_y.row));
                    // 前向傳播
                    output = FP(batch_x);

                    // 反向傳播
                    BP(output, batch_y);

                    // 梯度更新
                    update_weight();

                    // 顯示訓練進度
                    double loss_ = (*loss).undiff(output, batch_y) / batch_size;
                    printf("\rnow epoch:%4d, loss:%3.4f, progress:%4.4lf%%", e, loss_, b * 100 / (train_x.batch - 1.));
                }
            }


        }
    }

    // 將資料分成 BatchSize
    static inline Matrix get_batch_data(Matrix &train_data, int start, int end) {
        if (train_data.is_4d) {
            Matrix result(end - start, train_data.channel, train_data.row, train_data.col);
            for (int i = 0; i < (end - start); i++) {
                result.matrix_4d[i] = train_data.matrix_4d[start + i];
            }
            return result;
        } else {
            Matrix result(end - start, train_data.col);
            for (int i = 0; i < (end - start); i++) {
                result.matrix[i] = train_data.matrix[start + i];
            }
            return result;
        }
    }


    // 預測結果整理
    void predict(Matrix &x, Matrix &label, int batch_size) {
        // 取batch size筆資料進去預測
        Matrix batch_x = get_batch_data(x, 0, min((int) batch_size, (int) x.batch));

        // 前向傳播
        Matrix output = FP(batch_x);

        // 整理輸出結果與label對照
        cout << setw(8) << "Predict" << setw(2) << "|" << setw(8) << "Label" << endl;

        for (int r = 0; r < output.row; r++) {
            int cls = 0;
            double max = output.matrix[r][0];

            for (int c = 0; c < output.col; c++) {
                if (output.matrix[r][c] > max) {
                    max = output.matrix[r][c];
                    cls = c;
                }
            }
            cout << setw(8) << cls << setw(2) << "|" << setw(8) << label.matrix[r][0] << endl;
        }
    }

    // 前向傳播
    inline Matrix FP(Matrix &batch_x, bool training = true) {
        Matrix output = batch_x;

        for (int i = 0; i < layer_length; i++) {
            output = layer_list[i]->FP(output, training);
        }

        return output;
    }

    // 反向傳播
    inline void BP(Matrix &output, Matrix &batch_y, bool training = true) {
        Matrix delta = loss->diff(output, batch_y);
        for (int i = layer_length - 1; i > -1; i--) {
            delta = layer_list[i]->BP(delta, batch_y, training);
        }
    }

    // 更新權重
    inline void update_weight() const {
        opt->gradient_decent(layer_list, batch_size);
    }
};

int main() {
    srand(time(NULL));
    // 超參數
    int EPOCH = 1; // 學習次數
    int BATCH_SIZE = 32;  // 批量大小
    double LEARNING_RATE = 0.1;  // 學習率

    // 載入訓練資料
    printf("Load training data.....\n");
    Matrix *train_images = load_images("../train_images.idx3-ubyte", 60000);
    Matrix *train_labels = load_label("../train_labels.idx1-ubyte", 60000);
    Matrix *test_images = load_images("../test_images.idx3-ubyte", 10000);
    Matrix *test_labels = load_label("../test_labels.idx1-ubyte", 10000);
    printf("Loading successfully.\n");

    // 將label轉換為one hot code
    Matrix *train_one_hot_code = one_hot_code(*train_labels);

    // 創建序列模型 module(Epoch, Batch size, Loss Function, Optimizer)
    Sequential module(EPOCH, BATCH_SIZE, new Categorical_crosse_entropy, new Momentum(LEARNING_RATE, 0.8));

    module.add(new ConvolutionLayer(_1D_MATRIX{1, 28, 28}, 16, 3, 1, new relu));
    module.add(new MaxpoolingLayer(2));
//    module.add(new ConvolutionLayer(8, 3, 1, new relu));
    module.add(new FlattenLayer());
    module.add(new BaseLayer(10, new softmax));
    module.compile();

    // 訓練
    module.fit(*train_images, *train_one_hot_code);

    // 驗證
    module.predict(*test_images, *test_labels, 15);

    return 0;
}



