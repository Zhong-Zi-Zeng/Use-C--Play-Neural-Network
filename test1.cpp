#include <cassert>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <vector>
#include<Windows.h>


using namespace std;



int main(){
    string bar = "=";
    int loss = 2;

    for(int i = 0; i < 1000; i++){
        printf("\rloss:%d", loss);

        int block_num = i * 20 / 1000;
        for (int j = 1; j <= block_num; j++){
            cout << "=";
            Sleep(10);
        }
        loss += 1;
    }


    return 0;
}