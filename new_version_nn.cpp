#include <cassert>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <vector>

using namespace std;
typedef vector<double> _1D_MATRIX;
typedef vector<vector<double>> _2D_MATRIX;
typedef vector<vector<vector<double>>> _3D_MATRIX;
typedef vector<vector<vector<vector<double>>>> _4D_MATRIX;


// ==============================================================================
// -- 工具 -----------------------------------------------------------------------
// ==============================================================================
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
            // 用來存訪label對應的類別
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
// -- 隱藏層 ---------------------------------------------------------------------
// ==============================================================================
class Layer {
public:
    int output_shape;  // 輸入維度
    int input_shape;  // 輸出維度
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
        init(input_shape, output_shape, activation, use_bias);
    }

    BaseLayer(int output_shape, ActivationFunc *activation, bool use_bias = true) {
        init(output_shape, activation, use_bias);
    }

    void init(int input_shape, int output_shape, ActivationFunc *activation = new sigmoid, bool use_bias = true) {
        this->input_shape = input_shape;
        this->output_shape = output_shape;
        this->use_bias = use_bias;
        this->activation = activation;
        layer_name = "BaseLayer";
    }

    void init(int output_shape, ActivationFunc *activation, bool use_bias = true) {
        init(input_shape, output_shape, activation, use_bias);
    }

    void set_weight_bias() override {
        w = Matrix(input_shape, output_shape);
        b = Matrix(1, output_shape);
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

// ==============================================================================
// -- 優化器 ---------------------------------------------------------------------
// ==============================================================================
class Optimizer {
public:
    double learning_rate;

    virtual void gradient_decent(vector<Layer *> layer_list) = 0;
};

// ==============SGD==============
class SGD : public Optimizer {
public:
    SGD(double learning_rate) {
        this->learning_rate = learning_rate;
    }

    void gradient_decent(vector<Layer *> layer_list) override {
        /* 更新公式如下：
         * Wt = Wt - learning_rate * d_w
         */
        for (int i = 0; i < layer_list.size(); i++) {
            // 跳過dropout層
            if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer") {
                continue;
            }
            layer_list[i]->w = layer_list[i]->w - learning_rate * layer_list[i]->d_w;
            layer_list[i]->b = layer_list[i]->b - learning_rate * layer_list[i]->d_b;
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

    void gradient_decent(vector<Layer *> layer_list) override {
        // 第一次進來前先初始化last_v_w和last_v_b
        if (initial_flag) {
            initial_flag = false;

            for (int i = 0; i < layer_list.size(); i++) {
                // 若是遇到dropout、Flatten層，則加一個空陣列，方便後面計算
                if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer") {
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
            // 跳過dropout層
            if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer") {
                continue;
            }

            Matrix V_w_t = last_v_w[i] * beta - learning_rate * layer_list[i]->d_w;
            last_v_w[i] = V_w_t;
            layer_list[i]->w = layer_list[i]->w + V_w_t;

            Matrix V_b_t = last_v_b[i] * beta - learning_rate * layer_list[i]->d_b;
            last_v_b[i] = V_b_t;
            layer_list[i]->b = layer_list[i]->b + V_b_t;
        }
    }

};

// ==============================================================================
// -- 序列模型 -------------------------------------------------------------------
// ==============================================================================
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
            for (int b = 0; b < train_x.row; b += batch_size) {
                // 每次訓練讀取batch size 的資料去訓練
                Matrix batch_x = get_batch_data(train_x, b,
                                                min((int) b + batch_size, (int) train_x.row));
                Matrix batch_y = get_batch_data(train_y, b,
                                                min((int) b + batch_size, (int) train_y.row));
                // 前向傳播
                Matrix output = FP(batch_x);

                // 反向傳播
                BP(output, batch_y);

                // 梯度更新
                update_weight();

                // 顯示訓練進度
//                cout << "\r" << "Epoch:" << e;


                // 顯示訓練資料
                if (e == epoch - 1) {
                    cout << "========================" << endl;
                    cout << "Pre:" << endl;
                    output.show_matrix();
                    cout << "Label:" << endl;
                    batch_y.show_matrix();
                    cout << "Loss:" << endl;
                    cout << (*loss).undiff(output, batch_y) << endl;
                    cout << "========================" << endl;
                }

            }
        }
    }

    // 將資料分成 Batchsize
    static inline Matrix get_batch_data(Matrix &train_data, int start, int end) {
        Matrix result(end - start, train_data.col);

        for (int i = 0; i < (end - start); i++) {
            result.matrix[i] = train_data.matrix[start + i];
        }
        return result;
    }

    // 驗證
    void evaluate(Matrix val_x, Matrix val_y) {
        Matrix output = FP(val_x, false);

        cout << "========================" << endl;
        cout << "Val Result:" << endl;
        cout << "Pre:" << endl;
        output.show_matrix();
        cout << "Label:" << endl;
        val_y.show_matrix();
        cout << "Loss:" << endl;
        cout << (*loss).undiff(output, val_y) << endl;
        cout << "========================" << endl;
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
        opt->gradient_decent(layer_list);
    }
};

int main() {
    srand(time(NULL));
    // 超參數
    int EPOCH = 1000; // 學習次數
    int BATCH_SIZE = 5;  // 批量大小
    double LEARNING_RATE = 0.2;  // 學習率

    // 訓練資料
    double train_x[5][25] = {{0, 1, 1, 0, 0,
                                     0, 0, 1, 0, 0,
                                     0, 0, 1, 0, 0,
                                     0, 0, 1, 0, 0,
                                     0, 1, 1, 1, 0},
                             {1, 1, 1, 1, 0,
                                     0, 0, 0, 0, 1,
                                     0, 1, 1, 1, 0,
                                     1, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1},
                             {1, 1, 1, 1, 0,
                                     0, 0, 0, 0, 1,
                                     0, 1, 1, 1, 0,
                                     0, 0, 1, 0, 1,
                                     1, 1, 1, 1, 0},
                             {0, 0, 0, 1, 0,
                                     0, 0, 1, 1, 0,
                                     0, 1, 0, 1, 0,
                                     1, 1, 1, 1, 1,
                                     0, 0, 0, 1, 0},
                             {1, 1, 1, 1, 1,
                                     1, 0, 0, 0, 0,
                                     1, 1, 1, 1, 0,
                                     0, 0, 0, 0, 1,
                                     1, 1, 1, 1, 0}};

    double train_y[5][5] = {{1, 0, 0, 0, 0},
                            {0, 1, 0, 0, 0},
                            {0, 0, 1, 0, 0},
                            {0, 0, 0, 1, 0},
                            {0, 0, 0, 0, 1}};

//    double train_y[4][1] = {{1},
//                            {0},
//                            {0},
//                            {1}};


    // 將訓練資料轉為Matrix類別
    Matrix train_x_matrix = Matrix(train_x);
    Matrix train_y_matrix = Matrix(train_y);

    // 創建序列模型 module(Epoch, Batch size, Loss Function, Optimizer)
    Sequential module(EPOCH, BATCH_SIZE, new Categorical_crosse_entropy, new Momentum(LEARNING_RATE, 0.7));

    module.add(new BaseLayer(25, 16, new sigmoid));
//    module.add(new Dropout(0.2));
    module.add(new BaseLayer(5, new softmax));
    module.compile();

    // 訓練
    module.fit(train_x_matrix, train_y_matrix);

    // 驗證
//    module.evaluate(val_x, val_y);

    return 0;
}



