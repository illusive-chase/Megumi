#pragma once
#include "..\NN.h"
#include <cstdio>
#include <ctime>
using namespace Megumi;
using namespace std;

constexpr unsigned SIZE = 28;
constexpr unsigned SIZE_SQUARE = SIZE * SIZE;
constexpr unsigned BATCH = 60000;
constexpr unsigned CLASS_NUM = 10;

float* g_train_data_src = nullptr;
float* g_train_label_src = nullptr;
float(*g_train_data)[BATCH][1][SIZE_SQUARE] = nullptr;
float(*g_train_label)[BATCH][1][CLASS_NUM] = nullptr;

void load() {
	FILE* f = nullptr;
	fopen_s(&f, "Src\\mnist_train.csv", "r");
#if defined(_DEBUG)
	_ASSERT(f);
#endif
	g_train_data_src = new float[BATCH * SIZE_SQUARE]{};
	g_train_data = (decltype(g_train_data))g_train_data_src;
	g_train_label_src = new float[BATCH * CLASS_NUM]{};
	g_train_label = (decltype(g_train_label))g_train_label_src;
	for (unsigned i = 0; i < BATCH; ++i) {
		int tmp;
		fscanf_s(f, "%d,", &tmp), (*g_train_label)[i][0][tmp] = 1.0;
		for (unsigned j = 0; j < SIZE_SQUARE - 1; ++j) fscanf_s(f, "%d,", &tmp), (*g_train_data)[i][0][j] = tmp / 255.0;
		fscanf_s(f, "%d", &tmp), (*g_train_data)[i][0][SIZE_SQUARE - 1] = tmp / 255.0;
	}
	fclose(f);
}

void release() {
	delete[] g_train_data_src, g_train_data_src = nullptr;
	delete[] g_train_label_src, g_train_label_src = nullptr;
}

void print(const float (*src)[1][SIZE_SQUARE]) {
	putchar('\n');
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			float val = (*src)[0][i * SIZE + j];
			if (val < 0.3) putchar(' ');
			else if (val < 0.7) putchar('.');
			else putchar('*');
		}
		putchar('\n');
	}
	putchar('\n');
}

int classify(const float(*src)[1][CLASS_NUM]) {
	for (int i = 0; i < CLASS_NUM; ++i) {
		if ((*src)[0][i] > 0.99) return i;
	}
	return -1;
}

void view() {
	int r;
	while (cin >> r && r >= 0 && r < BATCH) {
		system("cls");
		cout << "Label " << classify(g_train_label[r]) << endl;
		print(g_train_data[r]);
	}
}


void full_connected_model() {
	load();
	NN nn;

	auto x = nn.placeholder<1, SIZE_SQUARE>();
	auto y = nn.placeholder<1, CLASS_NUM>();
	auto wa = nn.constant(nn.zero_tensor<SIZE_SQUARE, CLASS_NUM>());
	auto ba = nn.constant(nn.zero_tensor<1, CLASS_NUM>());
	auto y_ = softmax(x * wa + ba);
	auto loss = cross_entropy(y, y_);
	auto accuracy = argmax(y) == argmax(y_);

	for (int i = 1; i <= 10; ++i) {
		cout << i << endl;
		nn.train(loss, 1, 100, SGDOptimizer(100, 0.01, 0.9), FeedDict(0, 55000) = { x = g_train_data,y = g_train_label });
		nn.test(accuracy, FeedDict(55000, 60000) = { x = g_train_data,y = g_train_label });
	}

	
	nn.trace(y_, "result");
	nn.trace(y, "label");
	nn.trace(loss, "loss");

	int r;

	while (cin >> r && r >= 0 && r < BATCH) {
		float(*test_data)[1][1][SIZE_SQUARE];
		float(*test_label)[1][1][CLASS_NUM];
		test_data = (decltype(test_data))&(*g_train_data)[r];
		test_label = (decltype(test_label))&(*g_train_label)[r];
		nn.test(loss, FeedDict() = { x = test_data,y = test_label });

	}

	release();
}

int main() {
	load();
	NN nn;
	auto x = nn.placeholder<1, SIZE_SQUARE>();
	auto y = nn.placeholder<1, CLASS_NUM>();
	auto w1 = nn.constant(nn.normal_tensor<4, 25>(0, 0.1));
	auto b1 = nn.constant(nn.zero_tensor<4, SIZE_SQUARE>());
	auto h1 = max_pool(rectified_linear_unit(conv_same(x, shape<SIZE, SIZE>, w1, shape<5, 5>) + b1), shape<SIZE, SIZE>, shape<2, 2>);
	auto w2 = nn.constant(nn.normal_tensor<4 * 8, 25>(0, 0.1));
	auto b2 = nn.constant(nn.zero_tensor<8, 196>());
	auto h2 = max_pool(rectified_linear_unit(conv_same(h1, shape<14, 14>, w2, shape<5, 5>) + b2), shape<14, 14>, shape<2, 2>);
	auto h3 = reshape(h2);
	auto w3 = nn.constant(nn.normal_tensor<7 * 7 * 8, 256>(0, 0.1));
	auto b3 = nn.constant(nn.zero_tensor<1, 256>());
	auto h4 = rectified_linear_unit(h3 * w3 + b3);
	auto w4 = nn.constant(nn.normal_tensor<256, 10>(0, 0.1));
	auto b4 = nn.constant(nn.zero_tensor<1, 10>());
	auto y_ = softmax(h4 * w4 + b4);
	auto loss = cross_entropy(y, y_);
	auto accuracy = argmax(y) == argmax(y_);

	clock_t cl = clock();
	for (int i = 1; i <= 1; ++i) {
		cout << i << endl;
		nn.train(loss, 5, 20, SGDOptimizer(100, 5e-3), FeedDict(0, 1000) = { x = g_train_data,y = g_train_label });
		nn.test(accuracy, FeedDict(0, 1000) = { x = g_train_data,y = g_train_label });
		nn.test(accuracy, FeedDict(59500, 60000) = { x = g_train_data,y = g_train_label });
	}
	printf("%d\n", int(clock() - cl));
	system("pause");
	release();
}




/*auto w1 = nn.constant(nn.normal_tensor<4, 25>(0, 0.1));
auto b1 = nn.constant(nn.zero_tensor<4, SIZE_SQUARE>());
auto h1 = max_pool(rectified_linear_unit(conv_same(x, shape<SIZE, SIZE>, w1, shape<5, 5>) + b1), shape<SIZE, SIZE>, shape<2, 2>);
auto w2 = nn.constant(nn.normal_tensor<4 * 8, 25>(0, 0.1));
auto b2 = nn.constant(nn.zero_tensor<8, 196>());
auto h2 = max_pool(rectified_linear_unit(conv_same(h1, shape<14, 14>, w2, shape<5, 5>) + b2), shape<14, 14>, shape<2, 2>);
auto h3 = reshape(h2);
auto w3 = nn.constant(nn.normal_tensor<7 * 7 * 8, 256>(0, 0.1));
auto b3 = nn.constant(nn.zero_tensor<1, 256>());
auto h4 = rectified_linear_unit(h3 * w3 + b3);
auto w4 = nn.constant(nn.normal_tensor<256, 10>(0, 0.1));
auto b4 = nn.constant(nn.zero_tensor<1, 10>());
auto y_ = softmax(h4 * w4 + b4);*/

