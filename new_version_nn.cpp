#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <algorithm>
#include <time.h>

using namespace std;


// ==============================================================================
// -- 工具 -----------------------------------------------------------------------
// ==============================================================================

class Matrix {
public:
    double **matrix;
    int row;
    int col;

    template<size_t row, size_t col>
    Matrix(double (&arr)[row][col]){
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
            matrix[r] = (double *) malloc(sizeof(double) *  Matrix::col);
            for (int c = 0; c <  Matrix::col; c++) {
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

    Matrix operator+(Matrix other) {
        // 檢查維度
        _check_shape(&other);

        // 以最大維度為基準
        int max_row = max(Matrix::row, other.row);
        int max_col = max(Matrix::col, other.col);

        // 初始化結果
        Matrix result(max_row, max_col);

        if (Matrix::row == 1) {
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
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] + other.matrix[r][0];
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
        Matrix result(max_row, max_col);

        if (Matrix::row == 1) {
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
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] - other.matrix[r][0];
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
        Matrix result(max_row, max_col);

        if (Matrix::row == 1) {
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
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] * other.matrix[r][0];
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
        Matrix result(max_row, max_col);

        if (Matrix::row == 1) {
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
        } else {
            for (int r = 0; r < max_row; r++) {
                for (int c = 0; c < max_col; c++) {
                    result.matrix[r][c] = Matrix::matrix[r][c] / other.matrix[r][0];
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

    Matrix sum(int axis = 0) {
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

    double maximize(){
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

Matrix operator+(int i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = m.matrix[r][c] + i;
        }
    }
    return result;
}

Matrix operator-(int i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = i - m.matrix[r][c];
        }
    }
    return result;
}

Matrix operator*(int i, Matrix m) {
    // 初始化結果
    Matrix result(m.row, m.col);

    for (int r = 0; r < m.row; r++) {
        for (int c = 0; c < m.col; c++) {
            result.matrix[r][c] = m.matrix[r][c] * i;
        }
    }
    return result;
}

Matrix operator/(int i, Matrix m) {
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
// -- 激活函式抽象類 --------------------------------------------------------------
// ==============================================================================
class ActivationFunc {
public:
    virtual Matrix undiff(Matrix m) = 0;

    virtual Matrix diff(Matrix m, Matrix label) = 0;
};

// ==============sigmoid==============
class sigmoid : public ActivationFunc {
public:
    Matrix undiff(Matrix m) override {
        Matrix result(m.row, m.col, 0);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.col; c++) {
                result.matrix[r][c] = 1. / (1. + exp(-m.matrix[r][c]));
            }
        }
        return result;

    }

    Matrix diff(Matrix m, Matrix label) override {
        Matrix y = undiff(m);
        Matrix result(m.row, m.col, 0);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.row; c++) {
                result.matrix[r][c] = y.matrix[r][c] * (1. - y.matrix[r][c]);
            }
        }
        return result;
    }

};

// ==============relu==============
class relu : public ActivationFunc {
public:
    Matrix undiff(Matrix m) override {
        Matrix result(m.row, m.col, 0);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.col; c++) {
                result.matrix[r][c] = (m.matrix[r][c] > 0) ? m.matrix[r][c] : 0;
            }
        }
        return result;

    }

    Matrix diff(Matrix m, Matrix label) override {
        Matrix result(m.row, m.col, 0);

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
    Matrix undiff(Matrix m) override {
        return m;
    }

    Matrix diff(Matrix m, Matrix label) override {
        return Matrix(m.row, m.col, 1);
    }
};

// ==============softmax==============
class softmax : public ActivationFunc {
public:
    Matrix undiff(Matrix m) override {
        // 把所有元素減去最大值
        double max = m.maximize();
        Matrix new_m = m - max;

        // 對每一列求exp總和
        Matrix exp_sum(new_m.row, 1, 0);
        for (int r = 0; r < new_m.row; r++) {
            for (int c = 0; c < new_m.col; c++) {
                exp_sum.matrix[r][0] += std::exp(new_m.matrix[r][c]);
            }
        }
        // 將所有元素都以exp為底
        Matrix exp_m = Matrix::exp(new_m);

        // 將每一列都除上剛剛的exp_sum
        Matrix result(exp_m.row, exp_m.col, 0);

        for (int r = 0; r < m.row; r++) {
            for (int c = 0; c < m.col; c++) {
                result.matrix[r][c] = exp_m.matrix[r][c] / exp_sum.matrix[r][0];
            }
        }
        return result;
    }

    Matrix diff(Matrix m, Matrix label) override {
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
        Matrix result(row_size, col_size, 0);

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
        Matrix o1 = sub(pre, label);
        double o2 = total_sum(o1);
        o2 = (o2 * o2) / 2;

        return o2;
    }

    Matrix diff(Matrix pre, Matrix label) override {
        Matrix o1 = sub(pre, label);
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

        Matrix left_loss = multiply(multiply(label, ln(pre)), -1.);
        Matrix right_loss = multiply(sub(1., label), ln(sub(1, pre)));
        Matrix loss = sub(left_loss, right_loss);
        loss = sum(loss, 0);

        return loss[0][0];
    }

    Matrix diff(Matrix pre, Matrix label) override {
        /*
         * 公式如下：
         * (Yi - Di) / [Yi * (1 - Yi)]
         */

        Matrix left_loss = sub(pre, label);
        Matrix right_loss = multiply(pre, sub(1., pre));
        Matrix o1 = generate_mat((int) pre.size(), 1, 0, false);

        for (int i = 0; i < pre.size(); i++) {
            o1[i][0] = left_loss[i][0] / right_loss[i][0];
        }
        return o1;
    }
};


// ==============Categorical cross entropy==============
class Categorical_crosse_entropy : public LossFunc {
    double undiff(Matrix pre, Matrix label) override {
        /*
         * 公式如下 (Add 1e-7 Avoid ln(0)):
         * - sum(Di * ln(Yi + 0.0000001))
         */
        double loss = total_sum(multiply(multiply(label, ln(add(pre, 1e-7))), -1));
        return loss;
    }

    Matrix diff(Matrix pre, Matrix label) override {
        int row_size = label.size();
        int col_size = label[0].size();
        Matrix result;

        for (int r = 0; r < row_size; r++) {
            for (int c = 0; c < col_size; c++) {
                // 找出label對應的類別
                // -ln(yi) 微分結果剛好等於 - 1 / yi，把每一列都填入此數值
                if (label[r][c] == 1) {
                    result.push_back(vector<double>(col_size, -1. / pre[r][c]));
                    break;
                }
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
    ActivationFunc *activation; // 激活函式
    LossFunc *loss; // 損失函式
    Matrix w;  // Weight
    Matrix b;  // Bias
    Matrix x;  // 輸入
    Matrix u;  // 未使用激活函式前的輸出
    Matrix y;  // 使用激活函式後的輸出
    Matrix d_w; // Weight的梯度
    Matrix d_b; // Bias的梯度

    virtual void set_weight_bias() = 0;  // 初始化權重與偏置值
    virtual Matrix FP(Matrix, bool training) = 0; // 前向傳播
    virtual Matrix
    BP(Matrix, Matrix label, bool training) = 0; // 反向傳播
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
    }

    void init(int output_shape, ActivationFunc *activation, bool use_bias = true) {
        init(input_shape, output_shape, activation, use_bias);
    }


    void set_weight_bias() override {
        w = generate_mat(input_shape, output_shape);
        b = generate_mat(1, output_shape);
    }

    Matrix FP(Matrix x, bool training) override {
        this->x = x;

        u = dot(x, w);

        if (use_bias) {
            u = add(u, b);
        }
        y = (*activation).undiff(u);

        return y;
    }

    Matrix BP(Matrix delta, Matrix label, bool training) override {
        delta = multiply(delta, (*activation).diff(u, label));
        d_w = dot(transpose(x), delta);
        d_b = sum(delta, 0);
        Matrix d_x = dot(delta, transpose(w));

        return d_x;
    }
};

// ==============Dropout層==============
class Dropout : public Layer {
public:
    double prob; // 丟棄概率

    Dropout(double prob) {
        this->prob = prob;
    }

    void set_weight_bias() override {
        this->output_shape = input_shape;
    }

    Matrix FP(Matrix x, bool training) override {
        this->x = x;

        // 如果不是在訓練過程，則直接返回輸入值
        if (training == false) {
            x = multiply(x, 1. - prob);
            return x;
        } else {
            // 初始化權重
            if (w.size() == 0) {
                // 取得輸入x的shape
                int input_row_size = x.size();
                int input_col_size = x[0].size();
                w = generate_mat(input_row_size, input_col_size);
            }

            // 設置權重，若隨機數小於設置概率則將權重設為0，否則為1
            for (int r = 0; r < w.size(); r++) {
                for (int c = 0; c < w[0].size(); c++) {
                    double rand_num = rand() / (RAND_MAX + 1.0);
                    w[r][c] = (rand_num < prob) ? 0 : 1;
                }
            }

            // 將輸入與w相乘
            y = multiply(x, w);

            return y;
        }
    }

    Matrix BP(Matrix delta, Matrix label, bool training) override {
        if (training == false){
            return delta;
        }else{
            return multiply(delta, w);
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
            if (layer_list[i]->b.size() == 0){
                continue;
            }
            layer_list[i]->w = sub(layer_list[i]->w,
                                   multiply((Matrix) layer_list[i]->d_w, (double) learning_rate));
            layer_list[i]->b = sub(layer_list[i]->b,
                                   multiply((Matrix) layer_list[i]->d_b, (double) learning_rate));

        }
    }
};

// ==============Momentum==============
class Momentum : public Optimizer {
public:
    double beta = 0.9;  // Beta 為常數，通常設定為0.9
    vector<Matrix> last_dw;  // 用來存放上一次weight 的梯度
    vector<Matrix> last_db;  // 用來存放上一次bias 的梯度

    Momentum(double learning_rate, double beta) {
        this->learning_rate = learning_rate;
        this->beta = beta;
    }

    void gradient_decent(vector<Layer *> layer_list) override {
        // 第一次進來前先將上一次的梯度初始化為0
        if (last_dw.size() == 0 && last_db.size() == 0) {
            for (int i = 0; i < layer_list.size(); i++) {
                // 跳過dropout層
                if (layer_list[i]->b.size() == 0){
                    continue;
                }
                int weight_row_size = layer_list[i]->w.size();
                int weight_col_size = layer_list[i]->w[0].size();
                int bias_row_size = layer_list[i]->b.size();
                int bias_col_size = layer_list[i]->b[0].size();

                last_dw.push_back(generate_mat(weight_row_size, weight_col_size, 0, false));
                last_db.push_back(generate_mat(bias_row_size, bias_col_size, 0, false));
            }
        }

        // 更新梯度
        for (int i = 0; i < layer_list.size(); i++) {
            /* 更新公式如下：
             * Vt = Beta * Vt-1 - learning_rate * d_w
             * W = W + Vt
             */
            // 跳過dropout層
            if (layer_list[i]->b.size() == 0){
                continue;
            }

            Matrix V_w_t = multiply(last_dw[i], beta);
            V_w_t = sub(V_w_t, multiply(layer_list[i]->d_w, learning_rate));
            last_dw[i] = V_w_t;
            layer_list[i]->w = add(layer_list[i]->w, V_w_t);

            Matrix V_b_t = multiply(last_db[i], beta);
            V_b_t = sub(V_b_t, multiply(layer_list[i]->d_b, learning_rate));
            last_db[i] = V_b_t;
            layer_list[i]->b = add(layer_list[i]->b, V_b_t);
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
        int layer_length = layer_list.size();

        for (int i = 0; i < layer_length; i++) {
            layer_list[i]->set_weight_bias();

            if (i + 1 < layer_length) {
                layer_list[i + 1]->input_shape = layer_list[i]->output_shape;
            }
        }
    };

    // 訓練
    void fit(Matrix train_x, Matrix train_y) {
        for (int e = 0; e < epoch; e++) {
            for (int b = 0; b < train_x.size(); b += batch_size) {
                // 每次訓練讀取batch size 的資料去訓練
                Matrix batch_x = get_batch_data(train_x, b,
                                                                min((int) b + batch_size, (int) train_x.size()));
                Matrix batch_y = get_batch_data(train_y, b,
                                                                min((int) b + batch_size, (int) train_x.size()));
                Matrix output = FP(batch_x);
                BP(output, batch_y);
                update_weight();

                // 顯示訓練資料
                if (e == epoch - 1) {
                    cout << "========================" << endl;
                    cout << "Pre:" << endl;
                    show_mat(output);
                    cout << "Label:" << endl;
                    show_mat(batch_y);
                    cout << "Loss:" << endl;
                    cout << (*loss).undiff(output, batch_y) << endl;
                    cout << "========================" << endl;
                }

            }
        }
    }

    // 將資料分成 Batchsize
    inline Matrix get_batch_data(Matrix train_x, int start, int end) {
        Matrix result = generate_mat(end - start, train_x[0].size());

        for (int i = 0; i < (end - start); i++) {
            result[i] = train_x[start + i];
        }
        return result;
    }

    // 驗證
    void evaluate(Matrix val_x, Matrix val_y) {
        Matrix output = FP(val_x, false);

        cout << "========================" << endl;
        cout << "Val Result:" << endl;
        cout << "Pre:" << endl;
        show_mat(output);
        cout << "Label:" << endl;
        show_mat(val_y);
        cout << "Loss:" << endl;
        cout << (*loss).undiff(output, val_y) << endl;
        cout << "========================" << endl;
    }

    // 前向傳播
    Matrix FP(Matrix batch_x, bool training = true) {
        Matrix output = batch_x;

        for (int i = 0; i < layer_list.size(); i++) {
            output = layer_list[i]->FP(output, training);
        }
        return output;
    }

    // 反向傳播
    inline void BP(Matrix output, Matrix batch_y, bool training = true) {
        Matrix delta = loss->diff(output, batch_y);

        for (int i = layer_list.size() - 1; i > -1; i--) {
            delta = layer_list[i]->BP(delta, batch_y, training);
        }
    }

    // 更新權重
    inline void update_weight() {
        opt->gradient_decent(layer_list);
    }
};

int main() {
    srand(1);
    // 超參數
    int EPOCH = 5000; // 學習次數
    int BATCH_SIZE = 5;  // 批量大小
    double LEARNING_RATE = 0.001;  // 學習率
    double BETA = 0.9; // 用於Momentum 優化方法使用
    double DROPOUT_PROB = 0.5; // dropout概率

    // 訓練資料
    Matrix train_x = {{0, 1, 1, 0, 0,
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

    Matrix train_y = {{1, 0, 0, 0, 0},
                                      {0, 1, 0, 0, 0},
                                      {0, 0, 1, 0, 0},
                                      {0, 0, 0, 1, 0},
                                      {0, 0, 0, 0, 1}};



    // 創建序列模型 module(Epoch, Batch size, Loss Function, Optimizer)
    Sequential module(EPOCH, BATCH_SIZE, new Categorical_crosse_entropy, new SGD(LEARNING_RATE));

    module.add(new BaseLayer(25, 32, new relu));
    module.add(new Dropout(DROPOUT_PROB));
    module.add(new BaseLayer(128, new relu));
    module.add(new BaseLayer(5, new softmax));
    module.compile();

    // 訓練
    module.fit(train_x, train_y);

    // 驗證
    module.evaluate(val_x, val_y);

    return 0;
}


