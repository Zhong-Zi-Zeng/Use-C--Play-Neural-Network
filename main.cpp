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

// 生成權重
vector<vector<vector<vector<double>>>> generate_mat(int b, int r, int c, int n, double initial_value = 0) {
    vector<vector<vector<vector<double>>>> matrix(b, vector<vector<vector<double>>>(r, vector<vector<double>>(c,
                                                                                                              vector<double>(
                                                                                                                      n,
                                                                                                                      initial_value))));

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
    } else if (a_col == b_col && b_row == 1) {
        for (int i = 0; i < a_row; i++) {
            for (int j = 0; j < a_col; j++) {
                result[i][j] = a[i][j] + b[0][j];
            }
        }
        return result;
    } else {
        for (int i = 0; i < a_row; i++) {
            for (int j = 0; j < a_col; j++) {
                result[i][j] = a[i][j] + b[i][0];
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

vector<vector<double>> sum_axis1(vector<vector<double>> a) {
    int row = a.size();
    int col = a[0].size();

    // 向行求和
    vector<vector<double>> result = generate_mat(row, 1, 0, false);

    for (int i = 0; i < row; i++) {
        double total = 0;
        for (int j = 0; j < col; j++) {
            total += a[i][j];
        }
        result[i][0] = total;
    }
    return result;

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

    virtual vector<vector<double>> diff(vector<vector<double>> m, vector<vector<double>> label) = 0;
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

    vector<vector<double>> diff(vector<vector<double>> m, vector<vector<double>> label) override {
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

    vector<vector<double>> diff(vector<vector<double>> m, vector<vector<double>> label) override {
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

    vector<vector<double>> diff(vector<vector<double>> m, vector<vector<double>> label) override {
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

    vector<vector<double>> diff(vector<vector<double>> m, vector<vector<double>> label) override {
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
        vector<vector<double>> y = undiff(m);
        int row_size = label.size();
        int col_size = label[0].size();
        vector<vector<double>> result = generate_mat(row_size, col_size, 0, false);

        //        show_mat(y);
        //        show_mat(label);

        for (int r = 0; r < row_size; r++) {
            // 先找出label所對應的類別
            int maximize_index = 0;
            for (int l = 0; l < col_size; l++) {
                if (label[r][l] == 1) {
                    maximize_index = l;
                    break;
                }
            }

            // 判斷目前softmax輸出的索引是否跟label一樣
            for (int c = 0; c < col_size; c++) {
                if (c == maximize_index) {
                    result[r][c] = y[r][c] * (1. - y[r][c]);
                } else {
                    result[r][c] = -y[r][c] * y[r][maximize_index];
                }
            }
        }
        //        show_mat(result);

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
    double undiff(vector<vector<double>> pre, vector<vector<double>> label) override {
        vector<vector<double>> o1 = sub(pre, label);
        double o2 = total_sum(o1);
        o2 = (o2 * o2) / 2;

        return o2;
    }

    vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) override {
        vector<vector<double>> o1 = sub(pre, label);
        return o1;
    }
};

// ==============Binary cross entropy==============
class Binary_cross_entropy : public LossFunc {
public:
    double undiff(vector<vector<double>> pre, vector<vector<double>> label) override {
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

    vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) override {
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
    double undiff(vector<vector<double>> pre, vector<vector<double>> label) override {
        /*
         * 公式如下 (Add 1e-7 Avoid ln(0)):
         * - sum(Di * ln(Yi + 0.0000001))
         */
        double loss = total_sum(multiply(multiply(label, ln(add(pre, 1e-7))), -1));
        return loss;
    }

    vector<vector<double>> diff(vector<vector<double>> pre, vector<vector<double>> label) override {
        int row_size = label.size();
        int col_size = label[0].size();
        vector<vector<double>> result;

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
// -- 神經網路層 ---------------------------------------------------------------------
// ==============================================================================
class Layer {
public:
    int output_shape;  // 輸入維度
    int input_shape;  // 輸出維度
    bool use_bias; // 是否使用偏置值
    ActivationFunc *activation; // 激活函式
    LossFunc *loss; // 損失函式
    string layer_name; // 網路層名稱
    int layer_id; // 網路層編號
    int batch_size; // 照片數量
    int channel; // 通道數量
    int img_height; // 圖片高度
    int img_width; // 圖片寬度
    int filters; // 卷積核數量
    int k_size; // 卷積核大小
    int output_height; // 輸出圖片高度
    int output_width; // 輸出圖片寬度
    vector<vector<double>> w;  // Weight
    vector<vector<double>> b;  // Bias
    vector<vector<double>> x;  // 輸入
    vector<vector<double>> u;  // 未使用激活函式前的輸出
    vector<vector<double>> y;  // 使用激活函式後的輸出
    vector<vector<double>> d_w; // Weight的梯度
    vector<vector<double>> d_b; // Bias的梯度

    virtual void set_weight_bias() = 0;  // 初始化權重與偏置值
    virtual vector<vector<double>> FP(vector<vector<double>> x, bool training) = 0;
    virtual vector<vector<vector<vector<double>>>> FP(vector<vector<vector<vector<double>>>> x, bool training) = 0;
//    virtual vector<vector<double>> FP(vector<vector<vector<vector<double>>>> x, bool training) = 0;
    virtual vector<vector<double>> BP(vector<vector<double>> delta, vector<vector<double>> label, bool training) = 0;
    virtual vector<vector<vector<vector<double>>>>
    BP(vector<vector<vector<vector<double>>>> delta, vector<vector<double>> label, bool training) = 0;
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
        w = generate_mat(input_shape, output_shape);
        b = generate_mat(1, output_shape);
    }

    vector<vector<double>> FP(vector<vector<double>> x, bool training) override {
        this->x = x;

        u = dot(x, w);

        if (use_bias) {
            u = add(u, b);
        }
        y = (*activation).undiff(u);

        return y;
    }

    vector<vector<double>> BP(vector<vector<double>> delta, vector<vector<double>> label, bool training) override {
        delta = multiply(delta, (*activation).diff(u, label));
        d_w = dot(transpose(x), delta);
        d_b = sum(delta, 0);
        vector<vector<double>> d_x = dot(delta, transpose(w));

        return d_x;
    }
};

// ==============卷積層==============
class ConvolutionLayer : public Layer {
public:
    vector<vector<double>> img_2d; // 用來存放轉換完畢後的2維影像

    ConvolutionLayer(int batch_size, int channel, int img_height, int img_width, int filters, int k_size,
                     ActivationFunc *activation, bool use_bias = true) {
        init(batch_size, channel, img_height, img_width, filters, k_size, activation, use_bias);
    }

    ConvolutionLayer(int filters, int k_size, ActivationFunc *activation, bool use_bias = true) {
        init(filters, k_size, activation, use_bias);
    }

    void init(int batch_size, int channel, int img_height, int img_width, int filters, int k_size,
              ActivationFunc *activation = new sigmoid, bool use_bias = true) {
        this->batch_size = batch_size;
        this->channel = channel;
        this->img_height = img_height;
        this->img_width = img_width;
        this->filters = filters;
        this->k_size = k_size;
        this->output_height = (img_height - k_size) + 1;
        this->output_width = (img_width - k_size) + 1;
        this->use_bias = use_bias;
        this->activation = activation;
        layer_name = "ConvolutionLayer";
    }

    void init(int filters, int k_size, ActivationFunc *activation, bool use_bias = true) {
        init(batch_size, channel, img_height, img_width, filters, k_size, activation, use_bias);
    }

    void set_weight_bias() override {
        this->output_height = (img_height - k_size) + 1;
        this->output_width = (img_width - k_size) + 1;
        w = generate_mat(filters, channel * k_size * k_size);
        b = generate_mat(filters, 1);
    }

    vector<vector<double>> FP(vector<vector<double>> x, bool training) override {

    }

    vector<vector<double>> BP(vector<vector<double>> delta, vector<vector<double>> label, bool training) override {

    };

    vector<vector<vector<vector<double>>>> FP(vector<vector<vector<vector<double>>>> x, bool training) override {
        img_2d = im2col(x); // (Channel * f_h * f_w, batch_size * output_height * output_width)
        u = dot(w, img_2d); // (filters, batch_size * output_height * output_width)

        if (use_bias) {
            u = add(u, b);
        }

        y = (*activation).undiff(u);

        return img_reshape4d(y);
    }

    vector<vector<vector<vector<double>>>>
    BP(vector<vector<vector<vector<double>>>> delta, vector<vector<double>> label, bool training) override {
        vector<vector<double>> delta_2d = img_reshape2d(delta); // (filters, batch_size * output_height * output_width)
        delta_2d = multiply(delta_2d, (*activation).diff(u, label));
        d_w = dot(delta_2d, transpose(img_2d)); // (filters,  batch_size * output_height * output_width)
        d_b = sum_axis1(delta_2d);
        vector<vector<double>> d_x = dot(transpose(w),
                                         delta_2d); // (channel * k_size * k_size, batch_size * output_height * output_width)
        vector<vector<vector<vector<double>>>> d_x_4d = img_reshape4d(
                d_x); // (batch_size, channel, output_height, output_width).

        return d_x_4d;

    };

    vector<vector<double>> im2col(vector<vector<vector<vector<double>>>> img) {
        /** @brief 將輸入圖片轉為二維矩陣形式.
          * @param img 四維圖像
          * @param k_size  卷積核大小
          * @return 轉換完畢後的二維矩陣. */

        vector<vector<double>> result = generate_mat(channel * k_size * k_size,
                                                     batch_size * output_height * output_width, 0);

        for (int c = 0; c < result[0].size(); c++) {
            for (int r = 0; r < result.size(); r++) {
                int img_batch = c / (output_height * output_width); // 目前到哪張圖
                int img_channel = r / (k_size * k_size); // 目前到哪個通道
                int img_r = (r / k_size) % k_size + (c / output_width) % output_width; // 對應到原圖的row座標
                int img_c = r % k_size + c % output_height;  // 對應到原圖的col座標

                result[r][c] = img[img_batch][img_channel][img_r][img_c];
            }
        }

        return result;
    }

    vector<vector<vector<vector<double>>>> col2im(vector<vector<double>> img) {
        /** @brief 將輸入二維矩陣轉為4維圖片.
          * @param img 二維圖像 (channel * k_size * k_size, batch_size * output_height * output_width)
          * @return 轉換完畢後的二維矩陣. (batch_size, channel, img_height, img_width)
          * */

        vector<vector<vector<vector<double>>>> result = generate_mat(batch_size, channel, img_height, img_width, 0);

        for (int c = 0; c < img[0].size(); c++) {
            for (int r = 0; r < img.size(); r++) {
                int img_batch = c / (output_height * output_width); // 目前到哪張圖
                int img_channel = r / (k_size * k_size); // 目前到哪個通道
                int img_r = (r / k_size) % k_size + (c / output_width) % output_width; // 對應到原圖的row座標
                int img_c = r % k_size + c % output_height;  // 對應到原圖的col座標

                result[img_batch][img_channel][img_r][img_c] = img[r][c];
            }
        }

        return result;
    }

    vector<vector<vector<vector<double>>>> img_reshape4d(vector<vector<double>> img_2d) {
        /** @brief 將二維圖片轉換回原四維圖片
          * @param img 二維圖像
          * @return 轉換完畢後的四維圖片. (batch_size, channel, output_height, output_width).
          * */

        int row_size = img_2d.size();
        int col_size = img_2d[0].size();

        vector<vector<vector<vector<double>>>> result = generate_mat(batch_size, filters, output_height, output_width);

        for (int r = 0; r < row_size; r++) {
            for (int c = 0; c < col_size; c++) {
                int img_batch = c / (output_width * output_height); // 對應到四維圖像的哪張圖
                int img_r = (c / output_height) % output_height; // 對應到四維圖像的row座標
                int img_c = c % output_width; // 對應到四維圖像的col座標

                result[img_batch][r][img_r][img_c] = img_2d[r][c];
            }
        }

        return result;
    }

    vector<vector<double>> img_reshape2d(vector<vector<vector<vector<double>>>> img) {
        /** @brief 將四維梯度圖轉換回二維矩陣
        * @param img 四維圖像 (batch_size, filters, output_height, output_width).
        * @return 轉換完畢後的二維圖片. (filters, batch_size * output_height * output_width)
         * */

        vector<vector<double>> result = generate_mat(filters, batch_size * output_height * output_width);

        for (int r = 0; r < result.size(); r++) {
            for (int c = 0; c < result[0].size(); c++) {
                int img_batch = c / (output_width * output_height); // 對應到四維圖像的哪張圖
                int img_r = (c / output_height) % output_height; // 對應到四維圖像的row座標
                int img_c = c % output_width; // 對應到四維圖像的col座標
                result[r][c] = img[img_batch][r][img_r][img_c];
            }
        }

        return result;
    }
};

// ==============Flatten層==============
//class FlattenLayer : public Layer {
//public:
//    FlattenLayer() {
//        layer_name = "FlattenLayer";
//    }
//
//    void set_weight_bias() override {
//        this->output_shape = input_shape;
//    }
//
//    vector<vector<double>> FP(vector<vector<vector<vector<double>>>> x, bool training) {
//
//
//    }
//
//};


// ==============Dropout層==============
class DropoutLayer : public Layer {
public:
    double prob; // 丟棄概率

    DropoutLayer(double prob) {
        this->prob = prob;
        layer_name = "DropoutLayer";
    }

    void set_weight_bias() override {
        this->output_shape = input_shape;
    }

    vector<vector<double>> FP(vector<vector<double>> x, bool training) override {
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

    vector<vector<double>> BP(vector<vector<double>> delta, vector<vector<double>> label, bool training) override {
        if (training == false) {
            return delta;
        } else {
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
            if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer") {
                continue;
            }
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

    void gradient_decent(vector<Layer *> layer_list) override {
        // 第一次進來前先將上一次的梯度初始化為0
        if (last_dw.size() == 0 && last_db.size() == 0) {
            for (int i = 0; i < layer_list.size(); i++) {
                // 若是遇到dropout、Flatten層，則加一個空陣列，方便後面計算
                if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer") {
                    last_dw.push_back(generate_mat(0, 0, 0, false));
                    last_db.push_back(generate_mat(0, 0, 0, false));
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
            // 跳過dropout、Flatten層
            if (layer_list[i]->layer_name == "DropoutLayer" || layer_list[i]->layer_name == "FlattenLayer") {
                continue;
            }
            vector<vector<double>> V_w_t = multiply(last_dw[i], beta);
            V_w_t = sub(V_w_t, multiply(layer_list[i]->d_w, learning_rate));
            last_dw[i] = V_w_t;
            layer_list[i]->w = add(layer_list[i]->w, V_w_t);

            vector<vector<double>> V_b_t = multiply(last_db[i], beta);
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
            layer_list[i]->layer_id = i;
//            cout << "batch:" <<layer_list[i]->batch_size << endl;
//            cout << "channel:" <<layer_list[i]->channel << endl;
//            cout << "img_height:" <<layer_list[i]->img_height << endl;
//            cout << "img_width:" <<layer_list[i]->img_width << endl;
//            cout << "filters:" <<layer_list[i]->filters << endl;
//            cout << "k_size:" <<layer_list[i]->k_size << endl;
//            cout << "output_height:" <<layer_list[i]->output_height << endl;
//            cout << "output_width:" <<layer_list[i]->output_width << endl;
//            cout << "layer_id:" <<layer_list[i]->layer_id << endl << endl;

            if (i + 1 < layer_length) {
                if (layer_list[i + 1]->layer_name == "ConvolutionLayer") {
                    layer_list[i + 1]->batch_size = layer_list[i]->batch_size;
                    layer_list[i + 1]->channel = layer_list[i]->filters;
                    layer_list[i + 1]->img_height = layer_list[i]->output_height;
                    layer_list[i + 1]->img_width = layer_list[i]->output_width;
                } else if (layer_list[i + 1]->layer_name == "FlattenLayer") {
                    layer_list[i + 1]->batch_size = layer_list[i]->batch_size;
                    layer_list[i + 1]->input_shape = layer_list[i]->filters * layer_list[i]->output_height *
                                                     layer_list[i]->output_width;

                } else {
                    layer_list[i + 1]->input_shape = layer_list[i]->output_shape;
                }
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

    void fit(vector<vector<vector<vector<double>>>> train_x, vector<vector<double>> train_y) {
        for (int e = 0; e < epoch; e++) {
            for (int b = 0; b < train_x.size(); b += batch_size) {
                // 每次訓練讀取batch size 的資料去訓練
                vector<vector<vector<vector<double>>>> batch_x = get_batch_data(train_x, b, min((int) b + batch_size,
                                                                                                (int) train_x.size()));
                vector<vector<double>> batch_y = get_batch_data(train_y, b,
                                                                min((int) b + batch_size, (int) train_x.size()));

                vector<vector<vector<vector<double>>>> output = FP(batch_x);

                for (int b = 0; b < batch_size; b++) {
                    for (int c = 0; c < output[0].size(); c++) {
                        show_mat(output[b][c]);
                    }
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

    inline vector<vector<vector<vector<double>>>>
    get_batch_data(vector<vector<vector<vector<double>>>> train_x, int start, int end) {
        vector<vector<vector<vector<double>>>> result = generate_mat(end - start, train_x[0].size(),
                                                                     train_x[0][0].size(), train_x[0][0][0].size(), 0);

        for (int i = 0; i < (end - start); i++) {
            result[i] = train_x[start + i];
        }
        return result;
    }


    // 驗證
    void evaluate(vector<vector<double>> val_x, vector<vector<double>> val_y) {
        vector<vector<double>> output = FP(val_x, false);

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
    vector<vector<double>> FP(vector<vector<double>> batch_x, bool training = true) {
        vector<vector<double>> output = batch_x;

        for (int i = 0; i < layer_list.size(); i++) {
            output = layer_list[i]->FP(output, training);

        }
        return output;
    }

    vector<vector<vector<vector<double>>>> FP(vector<vector<vector<vector<double>>>> batch_x, bool training = true) {
        vector<vector<vector<vector<double>>>> output = batch_x;

        for (int i = 0; i < layer_list.size(); i++) {
            output = layer_list[i]->FP(output, training);

//            cout << "channel:" <<output[0].size() << endl;
//            cout << "img_height:" <<output[0][0].size() << endl;
//            cout << "img_height:" <<output[0][0][0].size() << endl << endl;

        }
        return output;

    }

    // 反向傳播
    inline void BP(vector<vector<double>> output, vector<vector<double>> batch_y, bool training = true) {
        vector<vector<double>> delta = loss->diff(output, batch_y);

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
    int EPOCH = 1; // 學習次數
    int BATCH_SIZE = 1;  // 批量大小
    double LEARNING_RATE = 0.001;  // 學習率
    double DROPOUT_PROB = 0.5; // dropout概率


    vector<vector<vector<vector<double>>>> train_x = {{{{1,  2,  3,  4},
                                                               {5,  6,  7,  8},
                                                               {9,  10, 11, 12},
                                                               {13, 14, 15, 16}}},

                                                      {{{17, 18, 19, 20},
                                                               {21, 22, 23, 24},
                                                               {25, 26, 27, 28},
                                                               {29, 30, 31, 32}}},

                                                      {{{33, 34, 35, 36},
                                                               {37, 38, 39, 40},
                                                               {41, 42, 43, 44},
                                                               {45, 46, 47, 48}}},

                                                      {{{{49, 50, 51, 52},
                                                             {53, 54, 55, 56},
                                                                 {57, 58, 59, 60},
                                                                     {61, 62, 63, 64}}}}};

    vector<vector<double>> train_y = {{0, 1, 0},
                                      {1, 0, 0},
                                      {1, 0, 0},
                                      {0, 0, 1}};


    // 創建序列模型 module(Epoch, Batch size, Loss Function, Optimizer)
    Sequential module(EPOCH, BATCH_SIZE, new Categorical_crosse_entropy, new Momentum(LEARNING_RATE, 0.8));

    // ConvolutionLayer(batch_size, channel, img_height, img_width, filters, k_size, ActivationFunc)
    module.add(new ConvolutionLayer(BATCH_SIZE, 1, 4, 4, 2, 2, new relu, false));
//    module.add(new ConvolutionLayer(16, 2, new relu));
    module.compile();

    // 訓練
    module.fit(train_x, train_y);

    // 驗證
//    module.evaluate(val_x, val_y);

    return 0;
}



