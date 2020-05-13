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
	const int LAYER = 3;  // 3 layers
	const int NUM = 10;   // the limit of neurons in each layer
	
	int iters;            // max training iteration
	
	int in_num;           // neurons number in input layer
	int hd_num;           // neurons number in hidden layer
	int ou_num;           // neurons number in output layer
	
	double ***w;          // weight between neurons
	double **b;           // bias for neurons
	double **s;           // output for neurons
	double **delta;       // delta for neurons
	
	void get_num(const Matrix&, const Matrix&);              // get each layer's neurons number
	void generate_array(double***, int, int);                // generate 2-d array
	void generate_array(double****, int, int, int);          // generate 3-d array
	void initialize_network();                               // give w/b random starting point
	void forward_propagation();                              // calculate s
	void calculate_delta(const vector<double>&);             // calculate delta
	void improve_network(int);                               // update w/b
	void backward_propagation(const vector<double>&, int);   // bp
	void record_network();                                   // record num/w/b to txt
	void read_network();                                     // read num/w/b from txt
	
	int forecast(const vector<double>&);                     // forecast the kind of input
	
	double random_01();                                      // return random float in [0,1]
	double sigmoid(double);                                  // digmoid function
	double eta(int t);                                       // calculate learning rate
	double calculate_accuracy(const Matrix&, const Matrix&); // calculate accuracy
	
public:
	BPClassifier(int iters = 100);
	~BPClassifier();
	void fit(const Matrix&, const Matrix&);
	vector<int> predict(const Matrix&);
};

BPClassifier::BPClassifier(int iters)
	: iters(iters)
{
	generate_array(&w, LAYER-1, NUM, NUM);
	generate_array(&b, LAYER, NUM);
}

BPClassifier::~BPClassifier()
{
	delete w, b;
}

void BPClassifier::get_num(const Matrix &X, const Matrix &Y)
{
	in_num = X.c;                                    // number of features
	ou_num = Y.c;                                    // number of label
	hd_num = min(NUM, (int)sqrt(in_num+ou_num)+6);   // a = 6
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
	generate_array(&s, LAYER, NUM);
	generate_array(&delta, LAYER, NUM);
	
	srand((unsigned int) time(0));
	for(int i = 1; i < LAYER; i++)                         // input layer b = 0
	for(int j = 0; j < NUM; j++) b[i][j] = random_01();

	for(int i = 0; i < LAYER-1; i++)
	for(int j = 0; j < NUM; j++)
	for(int k = 0; k < NUM; k++) w[i][j][k] = random_01();
	return;
}

void BPClassifier::forward_propagation()
{
	for(int j = 0; j < hd_num; j++)       // calculate s for hidden layer
	{
		double tmp = 0;
		for(int i = 0; i < in_num; i++) tmp += w[0][i][j] * s[0][i];
		s[1][j] = sigmoid(tmp + b[1][j]);
	}

	for(int j = 0; j < ou_num; j++)       // calculate s for output layer
	{
		double tmp = 0;
		for(int i = 0; i < hd_num; i++) tmp += w[1][i][j] * s[1][i];
		s[2][j] = sigmoid(tmp + b[2][j]);
	}
	return;
}

void BPClassifier::calculate_delta(const vector<double> &y)
{
	for(int i = 0; i < ou_num; i++)       // calculate delta for output layer
	{
		delta[2][i] = (s[2][i] - y[i]) * s[2][i] * (1 - s[2][i]);
	}
	for(int i = 0; i < hd_num; i++)       // calculate delta for hidden layer
	{
		double tmp = 0;
		for(int j = 0; j < ou_num; j++) tmp += w[1][i][j] * delta[2][j];
		delta[1][i] = tmp;
	}
	return;
}

void BPClassifier::improve_network(int t)
{
	for(int j = 0; j < ou_num; j++)
	{
		for(int i = 0; i < hd_num; i++)   // update w between hd/ou layer
		{
			w[1][i][j] -= eta(t) * delta[2][j] * s[1][i];
		}
		b[2][j] -= eta(t) * delta[2][j];   // update b for output layer
	}
	
	for(int j = 0; j < hd_num; j++)
	{
		for(int i = 0; i < in_num; i++)   // update w between in/hd layer
		{
			w[0][i][j] -= eta(t) * delta[1][j] * s[0][i];
		}
		b[1][j] -= eta(t) * delta[1][j];   // update b for hidden layer
	}
	return;
}

void BPClassifier::backward_propagation(const vector<double> &y, int t)
{
	calculate_delta(y);                   // y is grounding result
	improve_network(t);
	return;
}

void BPClassifier::record_network()
{
	ofstream myout;
	myout.clear();
	myout.open("network.txt", ios::out);

	myout << "input-layer-neurons-number: " << in_num << endl;
	myout << "hidden-layer-neurons-number: " << hd_num << endl;
	myout << "output-layer-neurons-number: " << ou_num << endl;

	myout << "weight-between-input-layer-and-hidden-layer:\n";
	for(int i = 0; i < in_num; i++)
	for(int j = 0; j < hd_num; j++)
	{
		myout << fixed << setprecision(20) << w[0][i][j] << (j == hd_num-1 ? "\n":"\t");
	} 

	myout << "weight-between-hidden-layer-and-output-layer:\n";
	for(int i = 0; i < hd_num; i++)
	for(int j = 0; j < ou_num; j++)
	{
		myout << fixed << setprecision(20) << w[1][i][j] << (j == ou_num-1 ? "\n":"\t");
	} 
	myout << "bias-for-hidden-layer:\n";
	for(int j = 0; j < hd_num; j++)
	{
		myout << fixed << setprecision(20) << b[1][j] << "\t";
	} 

	myout << "\nbias-for-output-layer:\n";
	for(int j = 0; j < ou_num; j++)
	{
		myout << fixed << setprecision(20) << b[2][j] << "\t";
	}
	
	myout.close();
	return;
}

void BPClassifier::read_network()
{
	ifstream myin;
	myin.clear();
	myin.open("network.txt", ios::in);
	
	string xyz;
	myin >> xyz >> xyz >> xyz >> xyz >> xyz >> xyz;
	
	myin >> xyz;
	for(int i = 0; i < in_num; i++)
	for(int j = 0; j < hd_num; j++)
	{
		myin >> w[0][i][j];
	} 

	myin >> xyz;
	for(int i = 0; i < hd_num; i++)
	for(int j = 0; j < ou_num; j++)
	{
		myin >> w[1][i][j];
	} 
	
	myin >> xyz;
	for(int j = 0; j < hd_num; j++)
	{
		myin >> b[1][j];
	} 

	myin >> xyz;
	for(int j = 0; j < ou_num; j++)
	{
		myin >> b[2][j];
	}
	
	myin.close();
	return;
}

int BPClassifier::forecast(const vector<double> &x)
{
	for(int i = 0; i < in_num; i++) s[0][i] = x[i];
	forward_propagation();
	double tmp = 0;                       // largest score
	int res = -1;                         // the kind of largest score
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

double BPClassifier::eta(int t)           // eta by Simulated Annealing
{
	return 5000.0 / (t + 50000);
}

double BPClassifier::calculate_accuracy(const Matrix &X, const Matrix &Y)
{
	int correct = 0;                      // number of the correct forecast
	for(int i = 0; i < X.r; i++) 
	{
		int p = forecast(X.v[i]);
		correct += Y.v[i][p] > 0.99;
	}
	return 1.0 * correct / X.r;           // return accuracy
}

void BPClassifier::fit(const Matrix &X, const Matrix &Y)
{
	cout << string(70, '*')+"\n";
	cout << "Start to train the BP Neutual Network Classifier!\n";
	get_num(X, Y);
	initialize_network();

	int num = X.r;
	double best = 0; 
	vector<int> id; 
	for(int i = 0; i < num; i++) id.push_back(i);

	for(int iter = 0; iter < iters; iter++)
	{
		random_shuffle(id.begin(), id.end());    // instance in random order
		for(int p : id)                          // trained by every instance
		{
			for(int i = 0; i < in_num; i++) s[0][i] = X.v[p][i];
			forward_propagation();
			backward_propagation(Y.v[p], p+iter*num);
		}
		double now = calculate_accuracy(X, Y);
		if(best < now)
		{
			record_network();                    // store the best network
			best = now;
		}
		cout << "Finish training iteraion " << iter+1;
		cout << ". Accuracy on train set: " << now << endl;
		if(best-now >= 2e-2) break;              // early stopping
	}
	
	read_network();                              // read the best network
	cout << string(70, '*')+"\n";
	cout << "End training!\n";
	cout << "Final Accuracy on train set: " << calculate_accuracy(X, Y);
	cout << "\n"+string(70, '*')+"\n";
	
	delete s, delta; 
	return;
}

vector<int> BPClassifier::predict(const Matrix &X)
{
	generate_array(&s, LAYER, NUM);
	vector<int> res;
	for(int i = 0; i < X.r; i++) res.push_back(forecast(X.v[i]));
	delete s;
	return res;
}

/*
model testing: 3 kind on 2-d grid
kind 0: 0<=x<=1  , 0.5<y<=1
kind 1: 0<=x<0.5 , 0<=y<=0.5
kind 2: 0.5<=x<=1, 0<=y<=0.5
*/

/*
int main()
{
	srand((unsigned int) time(0));
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
	}
	BPClassifier bp_clf;
	bp_clf.fit(X_train, Y_train);
	X_test.v[0][0] = 0.45, X_test.v[0][1] = 0.75;
	cout << bp_clf.predict(X_test)[0] << endl;
	X_test.v[0][0] = 0.25, X_test.v[0][1] = 0.25;
	cout << bp_clf.predict(X_test)[0] << endl;
	X_test.v[0][0] = 0.75, X_test.v[0][1] = 0.25;
	cout << bp_clf.predict(X_test)[0] << endl;
	return 0;
}
*/
