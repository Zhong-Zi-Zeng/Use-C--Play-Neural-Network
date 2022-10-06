#include <iostream>
#include <vector>
using namespace std;

class a{
public:
    int x;

    a(int x){
        this->x = x;
    }

    void change() const{
        this->x = 5;
    }

};


int main(){

    a n2(5);

    cout << n1 * n2 << endl;

    return 0;
}