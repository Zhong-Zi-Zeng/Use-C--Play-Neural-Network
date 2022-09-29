#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>
#include <fstream>


using namespace std;


// ==============工具==============
// 生成權重
vector<vector<double>> generate_mat(int r, int c, int initial_value = 0, bool use_random = true) {
    vector<vector<double>> matrix(r, vector<double>(c, initial_value));

    if (use_random) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                matrix[i][j] = (double) rand() / (RAND_MAX + 1.0);
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

// 矩陣相乘
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
vector<vector<double>> multiply(vector<vector<double>> a, double b){
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
    int row = a.size();
    int col = a[0].size();

    vector<vector<double>> result = generate_mat(row, col, 0, false);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            result[i][j] = a[i][j] + b[i][j];
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

// linspace
vector<double> linspace(double start, double end, int num){
    double interval = (end - start) / (num - 1);
    vector<double> result(num, 0);

    for (int i=0; i < num; i++){
        result[i] = start + i * interval;
    };

    return result;
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

// ==============激活函式抽象類==============
class ActivationFunc{
public:
    virtual double undiff (double pre) = 0;
    virtual double diff (double pre) = 0;

    // 計算激活函式輸出值
    vector<vector<double>> cal_activation(vector<vector<double>> m, bool diff = false) {
        int row = m.size();
        int col = m[0].size();

        vector<vector<double>> result = generate_mat(row, col);

        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                result[r][c] =  diff ? this->diff(m[r][c]) : this->undiff(m[r][c]);
            }
        }
        return result;
    }

};

// ==============sigmoid==============
class sigmoid : public ActivationFunc{
public:
    double undiff(double x) {
        return 1 / (1 + exp(-x));
    }

    double diff(double x) {
        double output = undiff(x);
        return output * (1 - output);
    }
};

// ==============relu==============
class relu : public ActivationFunc{
public:
    double undiff(double x) {
        return (x > 0)? x : 0;
    }

    double diff(double x) {
        return (x > 0)? 1 : 0;
    }
};

// ==============linear==============
class linear : public ActivationFunc{
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
    virtual double undiff (vector<vector<double>> pre, vector<vector<double>> label) = 0;
    virtual vector<vector<double>> diff (vector<vector<double>> pre, vector<vector<double>> label) = 0;

};

// ==============均方誤差損失函式==============
class MSE :public LossFunc{
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
        this->input_shape = input_shape;
        this->output_shape = output_shape;
        this->use_bias = use_bias;
        this->activation = activation;
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
 c
        return d_x;
    }
};


// ==============輸出層==============
class OutputLayer : public Layer {
public:
    OutputLayer(int input_shape, int output_shape, ActivationFunc *activation, LossFunc *loss, bool use_bias = true) {
        this->input_shape = input_shape;
        this->output_shape = output_shape;
        this->use_bias = use_bias;
        this->activation = activation;
        this->loss = loss;
    }

    void set_weight_bias() {
        w = generate_mat(input_shape, output_shape);
        b = generate_mat(1, output_shape);
    }

    double get_loss(vector<vector<double>> lable){
        return (*loss).undiff(y, lable);
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

    vector<vector<double>> BP(vector<vector<double>> lable) {
        vector<vector<double>> delta = (*loss).diff(y, lable);

        delta = multiply(delta, (*activation).cal_activation(u, true));

        d_w = dot(transpose(x), delta);
        d_b = sum(delta, 0);
        vector<vector<double>> d_x = dot(delta, transpose(w));

        return d_x;
    }

};


// ==============生成訓練數據==============
vector<vector<double>> generate_train_x(){
    vector<vector<double>> x(100, vector<double>(1, 0));
    int row = x.size();
    int col = x[0].size();
    double num = 0;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            x[i][j] = num;
            num += 1;
    }

    return x;
}

//vector<vector<double>> generate_train_y(vector<double> x){
//    vector<vector<double>> y(100, vector<double>(1, 0));
//    int row = y.size();
//
//    for (int i = 0; i < row; i++) {
//        for (int j = 0; j < col; j++)
//            y[i][j] = sin(x[i][j]);
//    }
//
//    return y;
//
//}

// ==============Batch==============
//vector<vector<double>> get_batch

int main() {
    srand(time(NULL));

    vector<double> r = linspace(0, 1, 100);

    for (int i=0; i < 100; i ++){
        cout << r[i] << endl;
    }

//    ofstream myFile;
//    myFile.open("test.txt");
//
////  實例化激活函式
//    sigmoid sigmoid;
//    linear linear;
//
////  實例化損失函式
//    MSE mse;
//
////  產生訓練數據
//    vector<vector<double>> train_x = generate_train_x(); // (100, 1)
//    vector<vector<double>> train_y = generate_train_y(train_x);// (100, 1)
//
////  建立層
//    BaseLayer D1(1, 128, &sigmoid, true);
//    OutputLayer D2(128, 1, &linear, &mse, true);
//
////  設置weight
//    D1.set_weight_bias();
//    D2.set_weight_bias();
//
//    for (int e=0; e < 100; e++){
//        for (int i=0; i < 100; i++){
//            vector<vector<double>> x(1, vector<double>(1, train_x[i][0]));
//            vector<vector<double>> y(1, vector<double>(1, train_y[i][0]));
//
//            //  FP
//            vector<vector<double>> o1 = D1.FP(x);
//            vector<vector<double>> o2 = D2.FP(o1);
//
//            cout << "loss:" << D2.get_loss(y) << endl;
////            cout << "label:" << y[0][0] << "  ";
//
//            if (e == 99){
//                myFile << o2[0][0] << "\n";
//            }
//
////            show_mat(o2);
//
//            // BP
//            vector<vector<double>> d_x1 = D2.BP(y);
//            vector<vector<double>> d_x2 = D1.BP(d_x1);
//
//            // gradient descent
//            D1.w = sub(D1.w, multiply(D1.d_w, (double) 0.01));
//            D1.b = sub(D1.b, multiply(D1.d_b, (double) 0.01));
//            D2.w = sub(D2.w, multiply(D2.d_w, (double) 0.01));
//            D2.b = sub(D2.b, multiply(D2.d_b, (double) 0.01));
//        }
//
//
//    }
//
//    myFile.close();

    return 0;
}



