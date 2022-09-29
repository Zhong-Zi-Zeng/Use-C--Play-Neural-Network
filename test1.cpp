
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
    MSE loss;

    show(loss);
    return 0;
}