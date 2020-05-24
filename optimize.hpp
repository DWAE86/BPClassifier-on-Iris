
#ifndef optimize_hpp
#define optimize_hpp

#include <bits/stdc++.h>

using namespace std;

class Optimize
{
private:
    double decay_rate;
    void L2(double weight_decay);
public:
    Optimize();

    bool isdropout,isL2,isadam;
    double weight_decay,eta0_w,eta0_b;

    void learning_rate_decay(double decay_rate);
    void learning_rate_decay(double &eta_b,double &eta_w,int epoch_num);
    void learning_rate_record(const double &eta_b,const double &eta_w);

    void regularization(string c,int parameter);
    void optimizer_adam();
};




#endif

