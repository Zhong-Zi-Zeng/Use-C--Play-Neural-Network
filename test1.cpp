#include <iostream>
using namespace std;

class Parent{
    public:
        virtual void FP() = 0;
        virtual void BP() = 0;

        void hi(){
            cout << "Hi" << endl;
        }

};

class MSE: public Parent{
    protected: int passward = 1231234;

    public:
        void FP(){
            cout << "fp" << endl;
        }

        void BP(){
            cout << "bp" << endl;
        }
};

void show(Parent &cls){
    cls.FP();
    cls.hi();
}

int main(){
    int a[] = {1 , 2 ,3};

    cout << a[-1] << endl;

    return 0;
}