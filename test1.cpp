#include <iostream>
#include "pbPlots.hpp"
#include "supportLib.hpp"

using namespace std;

int main(){
    RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
    char error[10] = "error!";

    vector<double> x{-2, 1, 2, 3, 4, 5};
    vector<double> y{-2, 1, 2, 3, 4, 5};
    vector<double> x2{-1, -1, 4, 3, 4, 5};
    vector<double> y2{-2, 1, 5, 3, 4, 7};

    DrawScatterPlot(imageRef, 600., 400., &x, &y);
    DrawScatterPlot(imageRef, 600., 400., &x2, &y2);
    vector<double> *pngData = ConvertToPNG(imageRef->image);
    WriteToFile(pngData, "plot.png");
    DeleteImage(imageRef->image);

    return 0;
}