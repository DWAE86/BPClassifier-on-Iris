
#include"optimize.hpp"

Optimize::Optimize():isdropout(false),isL2(false),isadam(false),decay_rate(0),weight_decay(0),keep_prop(1),eta0_b(0),eta0_w(0) {}

void Optimize::dropout(double keep_prop)
{
    this->keep_prop = keep_prop;
    return ;
}

void Optimize::L2(double weight_decay)
{
    this->weight_decay = weight_decay;
    return ;
}

void Optimize::learning_rate_decay(double decay_rate)
{
    this->decay_rate = decay_rate;
    return ;
}

void Optimize::learning_rate_record(const double &eta_b,const double &eta_w)
{
    eta0_b = eta_b;
    eta0_w = eta_w;
    return ;
}

void Optimize::learning_rate_decay(double &eta_b,double &eta_w,int epoch_num)
{
    eta_b = eta0_b/(1+decay_rate*epoch_num);
    eta_w = eta0_w/(1+decay_rate*epoch_num);
//    eta *= 0.95 ;
    return ;
}

void Optimize::regularization(string c, int parameter)
{
    if(c == "dropout")
    {
        isdropout = true;
        dropout(parameter);
    }
    if(c == "L2")
    {
        isL2 = true;
        L2(parameter);
    }
    return ;
}

void Optimize::optimizer_adam()
{
    isadam = true;
    return ;
}
