#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <time.h>

using namespace std;


// ==============================================================================
// -- 工具 -----------------------------------------------------------------------
// ==============================================================================

// 生成權重
vector<vector<double>> generate_mat(int r, int c, int initial_value = 0, bool use_random = true) {
    vector<vector<double>> matrix(r, vector<double>(c, initial_value));

    if (use_random) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                matrix[i][j] = 2 * rand() / (RAND_MAX + 1.0) - 1;
            }
        }
    }

    return matrix;
}

// 點積運算
vector<vector<double>> dot(vector<vector<double>> a, vector<vector<double>> b) {
    int a_row_size = a.size();
    int b_row_size = b.size();
    int b_col_size = b[0].size();

    vector<vector<double>> result = generate_mat(a_row_size, b_col_size, 0, false);

    for (int i = 0; i < a_row_size; i++) {
        for (int j = 0; j < b_col_size; j++) {
            for (int k = 0; k < b_row_size; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

// 矩陣乘法
vector<vector<double>> multiply(vector<vector<double>> a, vector<vector<double>> b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = a[i][j] * b[i][j];
    }

    return result;
}

vector<vector<double>> multiply(vector<vector<double>> a, double b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = a[i][j] * b;
    }
    return result;
}

// 轉置運算
vector<vector<double>> transpose(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();

    vector<vector<double>> result = generate_mat(col, row, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[j][i] = m[i][j];
    }

    return result;
}

// 矩陣相加
vector<vector<double>> add(vector<vector<double>> a, vector<vector<double>> b) {
    int a_row = a.size();
    int a_col = a[0].size();

    int b_row = b.size();
    int b_col = b[0].size();

    vector<vector<double>> result = generate_mat(a_row, a_col, 0, false);

    if (a_row == b_row && a_col == b_col) {
        for (int i = 0; i < a_row; i++) {
            for (int j = 0; j < a_col; j++)
                result[i][j] = a[i][j] + b[i][j];
        }
        return result;
    } else {
        for (int i = 0; i < a_row; i++) {
            for (int j = 0; j < a_col; j++) {
                result[i][j] = a[i][j] + b[0][j];
            }
        }
        return result;
    }
}

vector<vector<double>> add(vector<vector<double>> a, double b) {
    int a_row = a.size();
    int a_col = a[0].size();

    vector<vector<double>> result = generate_mat(a_row, a_col, 0, false);

    for (int i = 0; i < a_row; i++) {
        for (int j = 0; j < a_col; j++)
            result[i][j] = a[i][j] + b;
    }

    return result;
}

// 矩陣相減
vector<vector<double>> sub(vector<vector<double>> a, vector<vector<double>> b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = a[i][j] - b[i][j];
    }

    return result;
}

vector<vector<double>> sub(vector<vector<double>> a, double b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = a[i][j] - b;
    }

    return result;
}

vector<vector<double>> sub(double b, vector<vector<double>> a) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = b - a[i][j];
    }

    return result;
}

// 矩陣平均
double mean(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();
    double total = 0;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            total += m[i][j];
    }

    return total / (row * col);
}

// 矩陣平方
vector<vector<double>> square(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();
    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = m[i][j] * m[i][j];
    }

    return result;
}

// 求矩陣元素總和
vector<vector<double>> sum(vector<vector<double>> a, int axis = 0) {
    int row = a.size();
    int col = a[0].size();

    // 向列求和
    if (axis == 0) {
        vector<vector<double>> result = generate_mat(1, col, 0, false);

        for (int i = 0; i < col; i++) {
            for (int j = 0; j < row; j++) {
                result[0][i] += a[j][i];
            }
        }
        return result;
    }
        // 向行求和
    else {
        vector<vector<double>> result = generate_mat(row, 1, 0, false);

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result[i][0] += a[i][j];
            }
        }
        return result;
    }
}

// 求ln
vector<vector<double>> ln(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();
    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = log(m[i][j]);
    }

    return result;
}

// 求exp
vector<vector<double>> exp(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();
    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = std::exp(m[i][j]);
    }

    return result;
}

// 找最大值
double maximize(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();
    double max = 0;

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (result[i][j] > max) {
                max = result[i][j];
            }
        }
    }

    return max;


}


double total_sum(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();
    double total = 0;

    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            total += m[j][i];
        }
    }
    return total;
}

// 顯示矩陣內容
void show_mat(vector<vector<double>> m) {
    int row = m.size();
    int col = m[0].size();

    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            printf("%6f ", m[r][c]);
        }
        cout << endl;
    }
    cout << endl;
}

// 顯示矩陣維度
void shape(vector<vector<double>> m) {
    cout << "(" << m.size() << "," << m[0].size() << ")" << endl;
}


// ==============================================================================
// -- 激活函式抽象類 --------------------------------------------------------------
// ==============================================================================
class ActivationFunc {
public:
    virtual vector<vector<double>> undiff(vector<vector<double>> m) = 0;
    virtual vector<vector<double>> diff(vector<vector<double>> m) = 0;
};

// ==============sigmoid==============
class sigmoid : public ActivationFunc {
public:
    vector<vector<double>> undiff(vector<vector<double>> m) override {
        int row = m.size();
        int col = m[0].size();

        vector<vector<double>> result = generate_mat(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result[r][c] = 1. / (1. + exp(-m[r][c]));
            }
        }
        return result;

    }
    vector<vector<double>> diff(vector<vector<double>> m) override {
        int row = m.size();
        int col = m[0].size();
        vector<vector<double>> y = undiff(m);
        vector<vector<double>> result = generate_mat(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result[r][c] = y[r][c] * (1. - y[r][c]);
            }
        }
        return result;
    }

};

// ==============relu==============
class relu : public ActivationFunc {
public:
    vector<vector<double>> undiff(vector<vector<double>> m) override {
        int row = m.size();
        int col = m[0].size();

        vector<vector<double>> result = generate_mat(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result[r][c] = (m[r][c] > 0) ? m[r][c] : 0;
            }
        }
        return result;

    }
    vector<vector<double>> diff(vector<vector<double>> m) override{
        int row = m.size();
        int col = m[0].size();

        vector<vector<double>> result = generate_mat(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result[r][c] = (m[r][c] > 0) ? 1 : 0;
            }
        }
        return result;
    }
};

// ==============linear==============
class linear : public ActivationFunc {
public:
    vector<vector<double>> undiff(vector<vector<double>> m) override {
        return m;
    }

    vector<vector<double>> diff(vector<vector<double>> m) override{
        return generate_mat(m.size(), m[0].size(), 1, false);
    }
};

// ==============softmax==============
class softmax : public ActivationFunc {
public:
    vector<vector<double>> undiff(vector<vector<double>> m) override {
        // 把所有元素減去最大值
        double max = maximize(m);
        vector<vector<double>> new_m = sub(m, max);

        // 對每一列求exp總和
        vector<vector<double>> exp_sum = generate_mat(new_m.size(), 1, 0, false);
        for (int r = 0; r < new_m.size(); r++) {
            for (int c = 0; c < new_m[0].size(); c++) {
                exp_sum[r][0] += std::exp(new_m[r][c]);
            }
        }

        // 將所有元素都以exp為底
        vector<vector<double>> exp_m = exp(new_m);

        // 將每一列都除上剛剛的exp_sum
        vector<vector<double>> result = generate_mat(new_m.size(), new_m[0].size(), 0, false);

        for (int r = 0; r < m.size(); r++) {
            for (int c = 0; c < m[0].size(); c++) {
                result[r][c] = exp_m[r][c] / exp_sum[r][0];
            }
        }
        return result;
    }

    vector<vector<double>> diff(vector<vector<double>> m) override {
        // 回傳全為1的矩陣
        vector<vector<double>> result = generate_mat(m.size(), m[0].size(), 1, false);

        return result;
    }

};

// ==============================================================================
// -- 損失函式 --------------------------------------------------------------------
// ==============================================================================
class LossFunc {
public:
    virtual double undiff(vector<vector<double>> pre, vector<vector<double>> label) = 0;

    virtual vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) = 0;

};

// ==============均方誤差損失函式==============
class MSE : public LossFunc {
public:
    double undiff(vector<vector<double>> pre, vector<vector<double>> label) {
        vector<vector<double>> o1 = sub(pre, label);
        double o2 = total_sum(o1);
        o2 = (o2 * o2) / 2;

        return o2;
    }

    vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) {
        vector<vector<double>> o1 = sub(pre, label);
        return o1;
    }
};

// ==============Binary cross entropy==============
class Binary_cross_entropy : public LossFunc {
public:
    double undiff(vector<vector<double>> pre, vector<vector<double>> label) {
        /*
         * 公式如下：
         * -Di * ln(Yi) - (1 - Di) * ln(1 - Yi)
         */

        vector<vector<double>> left_loss = multiply(multiply(label, ln(pre)), -1.);
        vector<vector<double>> right_loss = multiply(sub(1., label), ln(sub(1, pre)));
        vector<vector<double>> loss = sub(left_loss, right_loss);
        loss = sum(loss, 0);

        return loss[0][0];
    }

    vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) {
        /*
         * 公式如下：
         * (Yi - Di) / [Yi * (1 - Yi)]
         */

        vector<vector<double>> left_loss = sub(pre, label);
        vector<vector<double>> right_loss = multiply(pre, sub(1., pre));
        vector<vector<double>> o1 = generate_mat((int) pre.size(), 1, 0, false);

        for (int i = 0; i < pre.size(); i++) {
            o1[i][0] = left_loss[i][0] / right_loss[i][0];
        }
        return o1;
    }
};


// ==============Categorical cross entropy==============
class Categorical_crosse_entropy : public LossFunc {
    double undiff(vector<vector<double>> pre, vector<vector<double>> label) {
        /*
         * 公式如下 (Add 1e-7 Avoid ln(0)):
         * - sum(Di * ln(Yi + 0.0000001))
         */
        double loss = total_sum(multiply(multiply(label, ln(add(pre, 1e-7))), -1));

        return loss;
    }

    vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) {
        /*
         * 使用softmax和多分類交叉熵的話，其反向傳播公式如下:
         *
         * ∂L   ∂L   ∂y
         * － = － x  －
         * ∂u   ∂y   ∂u
         *
         * if i == j:
         *       1
         *     - － x yi x (1 - yi) = yi - 1
         *       yi
         * else:
         *        1
         *     - － x yi x yj = yj
         *       yi
         */

        int row_size = label.size();
        int col_size = label[0].size();

        for (int r = 0; r < row_size; r++) {
            for (int c = 0; c < col_size; c++) {
                if (label[r][c] == 1) {
                    pre[r][c] = pre[r][c] - 1;
                }
            }
        }

        return pre;
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
    vector<vector<double>> w;  // Weight
    vector<vector<double>> b;  // Bias
    vector<vector<double>> x;  // 輸入
    vector<vector<double>> u;  // 未使用激活函式前的輸出
    vector<vector<double>> y;  // 使用激活函式後的輸出
    vector<vector<double>> d_w; // Weight的梯度
    vector<vector<double>> d_b; // Bias的梯度

    virtual void set_weight_bias() = 0;  // 初始化權重與偏置值
    virtual vector<vector<double>> FP(vector<vector<double>>) = 0; // 前向傳播
    virtual vector<vector<double>> BP(vector<vector<double>>) = 0; // 反向傳播
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


    void set_weight_bias() {
        w = generate_mat(input_shape, output_shape);
        b = generate_mat(1, output_shape);
    }

    vector<vector<double>> FP(vector<vector<double>> x) {
        this->x = x;

        u = dot(x, w);

        if (use_bias) {
            u = add(u, b);
        }
        y = (*activation).undiff(u);

        return y;
    }

    vector<vector<double>> BP(vector<vector<double>> delta) {
        delta = multiply(delta, (*activation).diff(u));
        d_w = dot(transpose(x), delta);
        d_b = sum(delta, 0);
        vector<vector<double>> d_x = dot(delta, transpose(w));

        return d_x;
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

    void gradient_decent(vector<Layer *> layer_list) {
        /* 更新公式如下：
         * Wt = Wt - learning_rate * d_w
         */

        for (int i = 0; i < layer_list.size(); i++) {
            layer_list[i]->w = sub(layer_list[i]->w,
                                   multiply((vector<vector<double>>) layer_list[i]->d_w, (double) learning_rate));
            layer_list[i]->b = sub(layer_list[i]->b,
                                   multiply((vector<vector<double>>) layer_list[i]->d_b, (double) learning_rate));
        }
    }
};

// ==============Momentum==============
class Momentum : public Optimizer {
public:
    double beta = 0.9;  // Beta 為常數，通常設定為0.9
    vector<vector<vector<double>>> last_dw;  // 用來存放上一次weight 的梯度
    vector<vector<vector<double>>> last_db;  // 用來存放上一次bias 的梯度

    Momentum(double learning_rate, double beta) {
        this->learning_rate = learning_rate;
        this->beta = beta;
    }

    void gradient_decent(vector<Layer *> layer_list) {
        // 第一次進來前先將上一次的梯度初始化為0
        if (last_dw.size() == 0 && last_db.size() == 0) {
            for (int i = 0; i < layer_list.size(); i++) {
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
            vector<vector<double>> V_w_t = multiply(last_dw[i], beta);
            V_w_t = sub(V_w_t, multiply(layer_list[i]->d_w, learning_rate));
            layer_list[i]->w = add(layer_list[i]->w, V_w_t);

            vector<vector<double>> V_b_t = multiply(last_db[i], beta);
            V_b_t = sub(V_b_t, multiply(layer_list[i]->d_b, learning_rate));
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
    void fit(vector<vector<double>> train_x, vector<vector<double>> train_y) {
        for (int e = 0; e < epoch; e++) {
            for (int b = 0; b < train_x.size(); b += batch_size) {
                // 每次訓練讀取batch size 的資料去訓練
                vector<vector<double>> batch_x = get_batch_data(train_x, b,
                                                                min((int) b + batch_size, (int) train_x.size()));
                vector<vector<double>> batch_y = get_batch_data(train_y, b,
                                                                min((int) b + batch_size, (int) train_x.size()));
                vector<vector<double>> output = FP(batch_x);
                BP(output, batch_y);
                update_weight();

                // 顯示訓練資料
                if (e == epoch - 1){
                    cout << "========================" << endl;
                    cout << "Pre:" << endl;
                    show_mat(output);
                    cout << "Label:" << endl;
                    show_mat(batch_y);
                    cout << "========================" << endl;
                }

            }
        }
    }

    // 將資料分成 Batchsize
    inline vector<vector<double>> get_batch_data(vector<vector<double>> train_x, int start, int end) {
        vector<vector<double>> result = generate_mat(end - start, train_x[0].size());

        for (int i = 0; i < (end - start); i++) {
            result[i] = train_x[start + i];
        }
        return result;
    }

    // 前向傳播
    vector<vector<double>> FP(vector<vector<double>> batch_x) {
        vector<vector<double>> output = batch_x;

        for (int i = 0; i < layer_list.size(); i++) {
            output = layer_list[i]->FP(output);
        }
        return output;
    }

    // 反向傳播
    inline void BP(vector<vector<double>> output, vector<vector<double>> batch_y) {
        vector<vector<double>> delta = loss->diff(output, batch_y);

        for (int i = layer_list.size() - 1; i > -1; i--) {
            delta = layer_list[i]->BP(delta);
        }
    }

    // 更新權重
    inline void update_weight() {
        opt->gradient_decent(layer_list);
    }
};

int main() {
    srand(time(NULL));

    vector<vector<double>> x = {{0, 1, 1, 0, 0,
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

    vector<vector<double>> y = {{1, 0, 0, 0, 0},
                                {0, 1, 0, 0, 0},
                                {0, 0, 1, 0, 0},
                                {0, 0, 0, 1, 0},
                                {0, 0, 0, 0, 1}};

    Sequential module(400, 5, new Categorical_crosse_entropy, new Momentum(0.2, 0.9));

    module.add(new BaseLayer(25, 64, new sigmoid));
    module.add(new BaseLayer(5, new softmax));

    module.compile();
    module.fit(x, y);


    return 0;
}



