#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

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

void img_reshape2d(vector<vector<vector<vector<double>>>> img) {
    // img (batch, f_n, o_h, o_w)

    int batch = img.size();
    int f_n = img[0].size(); // 卷積核數量
    int output_height = img[0][0].size(); // 輸出圖像高度
    int output_width = img[0][0][0].size(); // 輸出圖像寬度

    vector<vector<double>> result = generate_mat(f_n, batch * output_height * output_width);

    for (int r = 0; r < result.size(); r++) {
        for (int c = 0; c < result[0].size(); c++) {
            int img_batch = c / (output_width * output_height); // 對應到四維圖像的哪張圖
            int img_r = (c / output_height) % output_height; // 對應到四維圖像的row座標
            int img_c = c % output_width; // 對應到四維圖像的col座標
            result[r][c] = img[img_batch][r][img_r][img_c];
        }
    }

    show_mat(result);

//    return result;

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


int main() {
    vector<vector<vector<vector<double>>>> img = {{{{0, 1, 2},
                                                    {3, 4, 5},
                                                    {6, 7, 8}},

                                                   {{9, 10, 11},
                                                    {12, 13, 14},
                                                    {15, 16, 17}}},

                                                  {{{18, 19, 20},
                                                    {21, 22, 23},
                                                    {24, 25, 26}},
                                                   {{27, 28, 29},
                                                    {30, 31, 32},
                                                    {33, 34, 35}}}};

//    img_reshape2d(img);
    vector<vector<double>> w = generate_mat(5, 6, 1, false);
    vector<vector<double>> s = sum_axis1(w);
    show_mat(s);



    return 0;
}