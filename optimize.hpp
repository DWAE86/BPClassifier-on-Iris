
#ifndef optimize_hpp
#define optimize_hpp

#include <bits/stdc++.h>

using namespace std;

class Optimize
{
private:
    double decay_rate;
    void dropout(double keep_prop = 1);
    void L2(double weight_decay);
public:
    Optimize();

    bool isdropout,isL2;
    double weight_decay,keep_prop;

    void learning_rate_decay(double decay_rate);
    void learning_rate_decay(double &eta,int epoch_num);

    void regularization(string c,int parameter);
    //void optimizer_adam();
};




#endif

