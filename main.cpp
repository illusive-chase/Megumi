/*
MIT License

Copyright (c) 2019 illusive-chase

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/



#define import_all
#include <top_element/SImport.h>

#include "src/node.h"
using namespace megumi;

void System::Setup() {
	stage.addConsole();
	node<5, 2> x = { 1,1,1,2,1,3,1,4,1,5 };
	node<5, 1> y = { 96,204,297,403,501 };
	node<2, 1> theta = { 0.0f,100.0f };
	auto bias = x * theta - y;
	auto J = transpose(bias) * bias;
	auto D = partial_node(J, theta);
	output o = D.value();
	o.run();
	bias.print("bias");
	J.print("J");
	D.print("D");
	system("pause");
}