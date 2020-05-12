//
//  main.cpp
//  data_process
//
//  Created by GoatWu on 2020/5/12.
//  Copyright © 2020 吴朱冠宇. All rights reserved.
//

#include <bits/stdc++.h>
#include "data_process.hpp"
using namespace std;

DataFrame data;

int main(int argc, const char * argv[]) {
    data.read_file("iris.data");
    data.init();
    for (int i = 0; i < data.X_train.r; i++) {
        for (int j = 0; j < data.X_train.c; j++) {
            printf("%7.2lf", data.X_train.v[i][j]);
        }
        putchar(10);
    }
    return 0;
}
