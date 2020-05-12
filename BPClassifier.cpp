#include <bits/stdc++.h>
using namespace std;

struct Matrix
{
    int r, c;
    vector<vector<double> > v;
    Matrix(int r, int c) : r(r), c(c)
    {
        v.resize(r, vector<double>(c, 0));
    }
};

class BPClassifier
{
private:
	const int LAYER = 3;
	const int NUM = 10; //neurons
	
    int iters;
	double eta_w;
	double eta_b;
	
	int in_num;
	int hd_num;
	int ou_num;
	
	double ***w;
	double **b;
    double **s;
    double **delta;
	
    void get_num(const Matrix&, const Matrix&);
    void generate_array(double***, int, int);
    void generate_array(double****, int, int, int);
    void initialize_network();
	void forward_propagation();
    void calculate_delta(const vector<double>&);
    void improve_network();
    void backward_propagation(const vector<double>&);
    void record_network();
    
    int forecast(const vector<double>&);
    
    double random_01();
	double sigmoid(double);
    double calculate_accuracy(const Matrix&, const Matrix&);
    
public:
    BPClassifier(int iters = 20, double eta_w = 1e-1, double eta_b = 1e-1);
	~BPClassifier();
    void fit(const Matrix&, const Matrix&);
	vector<int> predict(const Matrix&);
};

BPClassifier::BPClassifier(int iters, double eta_w, double eta_b)
    : iters(iters), eta_w(eta_w), eta_b(eta_b)
{
    generate_array(&w, LAYER-1, NUM, NUM);
    generate_array(&b, LAYER, NUM);
    generate_array(&s, LAYER, NUM);
    generate_array(&delta, LAYER, NUM);
}

BPClassifier::~BPClassifier()
{
    delete w, b, s, delta;
}

void BPClassifier::get_num(const Matrix &X, const Matrix &Y)
{
    in_num = X.c;
    ou_num = Y.c;
    hd_num = min(NUM, (int)sqrt(in_num+ou_num)+6);
    return;
}

void BPClassifier::generate_array(double ***a, int n0, int n1)
{
    *a = new double *[n0];
    for(int i = 0; i < n0; i++) 
    {
        (*a)[i] = new double [n1];
    }
    return;
}

void BPClassifier::generate_array(double ****a, int n0, int n1, int n2)
{
    *a = new double **[n0];
    for(int i = 0; i < n0; i++)
    {
        (*a)[i] = new double *[n1];
        for(int j = 0; j < n1; j++) 
        {
            (*a)[i][j] = new double [n2];
        }
    }
    return;
}

void BPClassifier::initialize_network()
{
    srand((unsigned int) time(0));
    for(int i = 0; i < LAYER; i++) 
    for(int j = 0; j < NUM; j++) b[i][j] = random_01();

    for(int i = 0; i < LAYER-1; i++)
    for(int j = 0; j < NUM; j++)
    for(int k = 0; k < NUM; k++) w[i][j][k] = random_01();
    return;
}

void BPClassifier::forward_propagation()
{
    for(int j = 0; j < hd_num; j++)
    {
        double tmp = 0;
        for(int i = 0; i < in_num; i++) tmp += w[0][i][j] * s[0][i];
        s[1][j] = sigmoid(tmp + b[1][j]);
    }

    for(int j = 0; j < ou_num; j++)
    {
        double tmp = 0;
        for(int i = 0; i < hd_num; i++) tmp += w[1][i][j] * s[1][i];
        s[2][j] = sigmoid(tmp + b[2][j]);
    }
    return;
}

void BPClassifier::calculate_delta(const vector<double> &y)
{
    for(int i = 0; i < ou_num; i++)
    {
//    	printf("!!!%d %.2f\n", i, y[i]);
        delta[2][i] = (s[2][i] - y[i]) * s[2][i] * (1 - s[2][i]);
    }
    for(int i = 0; i < hd_num; i++)
    {
        double tmp = 0;
        for(int j = 0; j < ou_num; j++) tmp += w[1][i][j] * delta[2][j];
        delta[1][i] = tmp;
    }
    return;
}

void BPClassifier::improve_network()
{
    for(int j = 0; j < ou_num; j++)
    {
        for(int i = 0; i < hd_num; i++) 
        {
            w[1][i][j] -= eta_w * delta[2][j] * s[1][i];
//            printf(">>>%.5f %.5f %.5f %.5f %.5f\n", eta_w, delta[2][j], s[1][i], w[1][i][j], eta_w * delta[2][j] * s[1][i]);
        }
        b[2][j] -= eta_b * delta[2][j];
    }
    
    for(int j = 0; j < hd_num; j++)
    {
        for(int i = 0; i < in_num; i++)
        {
            w[0][i][j] -= eta_w * delta[1][j] * s[0][i];
        }
        b[1][j] -= eta_b * delta[1][j];
    }
    return;
}

void BPClassifier::backward_propagation(const vector<double> &y)
{
//	printf("???%.5f %.5f %.5f\n", s[2][0], s[2][1], s[2][2]);
    calculate_delta(y);
//    cout << "---\n";
//    for(int i = 0; i < ou_num; i++) printf("%.2f ", delta[2][i]); putchar(10);
//    for(int i = 0; i < hd_num; i++) printf("%.2f ", delta[1][i]); putchar(10);
    improve_network();
    return;
}

void BPClassifier::record_network()
{
//    freopen("network.txt", "w", stdout);

//    cout << "input layer neurons number: " << in_num << endl;
//    cout << "hidden layer neurons number: " << hd_num << endl;
//    cout << "output layer neurons number: " << ou_num << endl;

    cout << "\nweight between input layer and hidden layer:\n";
    for(int i = 0; i < in_num; i++, putchar(10))
    for(int j = 0; j < hd_num; j++)
	{
    	cout << fixed << setprecision(4) << w[0][i][j] << " ";
	} 

    cout << "\nweight between input layer and hidden layer:\n";
    for(int i = 0; i < hd_num; i++, putchar(10))
    for(int j = 0; j < ou_num; j++)
	{
		cout << fixed << setprecision(4) << w[1][i][j] << " ";
	} 

//    fclose(stdout);
    return;
}

int BPClassifier::forecast(const vector<double> &x)
{
    for(int i = 0; i < in_num; i++) s[0][i] = x[i];
    forward_propagation();
    double tmp = 0;
    int res = -1;
    for(int i = 0; i < ou_num; i++) 
    {
        if(s[2][i] > tmp) tmp = s[2][res = i];
    }
    return res;
}

double BPClassifier::random_01()
{
    return (double) (rand()%1001) / 1000.0;
}

double BPClassifier::sigmoid(double x)
{
    return 1 / ( 1 + exp(-x) );
}

double BPClassifier::calculate_accuracy(const Matrix &X, const Matrix &Y)
{
    int correct = 0;
    for(int i = 0; i < X.r; i++) 
    {
        int p = forecast(X.v[i]);
        correct += Y.v[i][p] == 1;
    }
    return 1.0 * correct / X.r;
}

void BPClassifier::fit(const Matrix &X, const Matrix &Y)
{
    cout << "Start to train the BP Neutual Network Classifier!\n";
    get_num(X, Y);
    initialize_network();

    int num = X.r;
    double best = 0; 
    for(int iter = 0; iter < iters; iter++)
    {
        for(int p = 0; p < num; p++)
        {
            for(int i = 0; i < in_num; i++) s[0][i] = X.v[p][i];
            forward_propagation();
            backward_propagation(Y.v[p]);
        }
        double now = calculate_accuracy(X, Y);
        best = max(best, now);
        cout << "Finish training iteraion " << iter+1;
        cout << ". Accuracy on train set: " << now << endl;
        if(best-now >= 1e-2) break;
//    	record_network();
    }

    record_network();
    cout << "End training!\n";
    return;
}

vector<int> BPClassifier::predict(const Matrix &X)
{
    vector<int> res;
    for(int i = 0; i < X.r; i++) res.push_back(forecast(X.v[i]));
    return res;
}

int main()
{
//	freopen("network.txt", "w", stdout);
    cout << "Hello Test!!!\n";
    int n = 1000;
    Matrix X_train(n, 2), Y_train(n, 3), X_test(1, 2);
    for(int i = 0; i < n; i++)
    {
        X_train.v[i][0] = rand()%1001/1000.0;
        X_train.v[i][1] = rand()%1001/1000.0;
        if(X_train.v[i][1] > 0.5) Y_train.v[i][0] = 1;
        else if(X_train.v[i][0] < 0.5) Y_train.v[i][1] = 1;
        else Y_train.v[i][2] = 1;
//        printf(">>%f %f %f %f %f\n", X_train.v[i][0], X_train.v[i][1], Y_train.v[i][0], Y_train.v[i][1], Y_train.v[i][2]);
    }
    BPClassifier bp_clf;
    bp_clf.fit(X_train, Y_train);
    X_test.v[0][0] = 0.25, X_test.v[0][1] = 0.25;
    cout << bp_clf.predict(X_test)[0] << endl;
    return 0;
}
