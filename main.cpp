#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <time.h>

using namespace std;

// ==============工具==============
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
    } else{
        for (int i = 0; i < a_row; i++) {
            for (int j = 0; j < a_col; j++) {
                result[i][j] = a[i][j] + b[0][j];
            }
        }
        return result;
    }
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

// linspace
vector<vector<double>> linspace(double start, double end, int num) {
    double interval = (end - start) / (num - 1);
    vector<vector<double>> result = generate_mat(num, 1);

    for (int i = 0; i < num; i++) {
        result[i][0] = start + i * interval;
    };

    return result;
}

vector<vector<double>> generate_y(vector<vector<double>> x) {
    int row = x.size();

    vector<vector<double>> result = generate_mat(row, 1);

    for (int i = 0; i < row; i++) {
        result[i][0] = sin(x[i][0]);
    }

    return result;
}


// ==============激活函式抽象類==============
class ActivationFunc {
public:
    virtual double undiff(double pre) = 0;

    virtual double diff(double pre) = 0;

    // 計算激活函式輸出值
    vector<vector<double>> cal_activation(vector<vector<double>> m, bool diff = false) {
        int row = m.size();
        int col = m[0].size();

        vector<vector<double>> result = generate_mat(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result[r][c] = diff ? this->diff(m[r][c]) : this->undiff(m[r][c]);
            }
        }
        return result;
    }

};

// ==============sigmoid==============
class sigmoid : public ActivationFunc {
public:
    double undiff(double x) {
        return 1. / (1. + exp(-x));
    }

    double diff(double x) {
        double output = undiff(x);
        return output * (1. - output);
    }
};

// ==============relu==============
class relu : public ActivationFunc {
public:
    double undiff(double x) {
        return (x > 0) ? x : 0;
    }

    double diff(double x) {
        return (x > 0) ? 1 : 0;
    }
};

// ==============linear==============
class linear : public ActivationFunc {
public:
    double undiff(double x) {
        return x;
    }

    double diff(double x) {
        return 1;
    }
};

// ==============損失函式抽象類==============
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

// ==============抽象層==============
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
        y = (*activation).cal_activation(u, false);

        return y;
    }

    vector<vector<double>> BP(vector<vector<double>> delta) {
        delta = multiply(delta, (*activation).cal_activation(u, true));
        d_w = dot(transpose(x), delta);
        d_b = sum(delta, 0);
        vector<vector<double>> d_x = dot(delta, transpose(w));

        return d_x;
    }
};


// ==============序列模型==============
class Sequential {
public:
    int epoch;  // 訓練次數
    int batch_size;  // 批量大小
    double learning_rate; // 學習率
    LossFunc *loss; // 損失函式
    vector<Layer *> layer_list;  // 存放網路層

    Sequential(int epoch, int batch_size, double learning_rate, LossFunc *loss) {
        this->epoch = epoch;
        this->batch_size = batch_size;
        this->learning_rate = learning_rate;
        this->loss = loss;
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

                vector<vector<double>> batch_x = get_batch_data(train_x, b,
                                                                min((int) b + batch_size, (int) train_x.size()));
                vector<vector<double>> batch_y = get_batch_data(train_y, b,
                                                                min((int) b + batch_size, (int) train_x.size()));
                vector<vector<double>> output = FP(batch_x);
                BP(output, batch_y);
                update_weight();

                cout << "Epoch:" << e << endl;
//                cout << "batch_x:" << endl;
//                show_mat(batch_x);
//                cout << "batch_y:" << endl;
//                show_mat(batch_y);
                cout << "Pre:" << endl;
                show_mat(output);
                cout << "Label:" << endl;
                show_mat(batch_y);


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

    // 更新梯度
    inline void update_weight() {
        for (int i = 0; i < layer_list.size(); i++) {
            layer_list[i]->w = sub(layer_list[i]->w, multiply(
                    multiply((vector<vector<double>>) layer_list[i]->d_w, (double) learning_rate), (double ) 1. / batch_size));
            layer_list[i]->b = sub(layer_list[i]->b, multiply(
                    multiply((vector<vector<double>>) layer_list[i]->d_b, (double) learning_rate), (double ) 1. / batch_size));
        }
    }
};

int main() {
    srand(time(NULL));

    vector<vector<double>> x = {{0, 0, 1},
                                {0, 1, 1},
                                {1, 0, 1},
                                {1, 1, 1}};
    vector<vector<double>> y = {{0},
                                {0},
                                {1},
                                {1}};


    Sequential module(1000, 2, 0.2, new MSE);

    module.add(new BaseLayer(3, 16, new sigmoid));
    module.add(new BaseLayer(32, new sigmoid));
    module.add(new BaseLayer(64, new sigmoid));
    module.add(new BaseLayer(1, new sigmoid));

    module.compile();
    module.fit(x, y);

    return 0;
}



