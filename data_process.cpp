//
//  data_process.cpp
//  data_process
//
//  Created by GoatWu on 2020/5/12.
//  Copyright © 2020 吴朱冠宇. All rights reserved.
//

#include "data_process.hpp"

template <class T>
void DataFrame::shuffle(vector<T> &a) {
    srand(unsigned(time(0)));
    for (int i = 0; i < a.size(); i++) {
        int id = rand() % (a.size() - i) + i;
        swap(a[i], a[id]);
    }
}

void DataFrame::read_file(const char *name) {
    freopen(name, "r", stdin);
    string s;
    tot = 0;
    for (; getline(cin, s); tot++) {
        if (s.length() < 5) {
            break;
        }
        str.push_back(s);
    }
    shuffle(str);
}


bool DataFrame::isdouble(string s) {
    for (char c : s) {
        if (!isdigit(c) && c != '.') {
            return false;
        }
    }
    return true;
}

void DataFrame::toOnehot(vector<Data> &d, Matrix &Y) {
    int cnt = 0;
    map<string, int> ma;
    for (Data x: d) {
        if (!ma.count(x.label)) {
            ma[x.label] = cnt++;
        }
    }
    Y.r = (int)d.size();
    Y.c = cnt;
    for (int i = 0; i < Y.r; i++) {
        vector<double> v(cnt);
        Data x = d[i];
        v[ma[x.label]] = 1;
        Y.v.push_back(v);
    }
}

void DataFrame::Normalize(vector<Data> &d, Matrix &X) {
    for (Data x : d) {
        X.v.push_back(x.v);
    }
    X.r = (int)d.size();
    X.c = (int)d[0].v.size();
    for (int i = 0; i < X.c; i++) {
        double sum = 0;
        for (int j = 0; j < X.r; j++) {
            sum += X.v[j][i];
        }
        double mean = sum / X.r;
        double std = 0;
        for (int j = 0; j < X.r; j++) {
            std += (X.v[j][i] - mean) * (X.v[j][i] - mean);
        }
        std /= X.r - 1;
        std = sqrt(std);
        for (int j = 0; j < X.r; j++) {
            (X.v[j][i] -= mean) /= std;
        }
    }
}

void DataFrame::deal_data(vector<string> &vs, Matrix &X, Matrix &Y) {
    vector<string> v;
    string s;
    vector<Data> data;
    Data d;
    for (int i = 0; i < vs.size(); i++) {
        v.resize(0);
        d.v.resize(0);
        istringstream is(vs[i]);
        while (getline(is, s, ',')) {
            v.push_back(s);
        }
        for (string x : v) {
            if (isdouble(x)) {
                d.v.push_back(stod(x));
            }
            else {
                d.label = x;
            }
        }
        data.push_back(d);
    }
    toOnehot(data, Y);
    Normalize(data, X);
}

void DataFrame::init() {
    deal_data(str, X_train, Y_train);
    int beg = 4 * tot / 5;
    for (int i = beg; i < tot; i++) {
        vector<double> x = X_train.v[i], y = Y_train.v[i];
        X_test.v.push_back(x);
        Y_test.v.push_back(y);
    }
    X_train.v.resize(beg); Y_train.v.resize(beg);
    X_train.r = Y_train.r = beg;
    X_test.r = Y_test.r = tot - beg;
    X_test.c = X_train.c;
    Y_test.c = Y_train.c;
}
