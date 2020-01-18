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
#pragma once
#include <vector>
#include <list>
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

namespace megumi {

	using scalar = float;

	class matrix {
	public:
		using pmatrix = std::shared_ptr<matrix>;
		using functor = void(*)(unsigned, unsigned, scalar*, scalar*, std::list<pmatrix>&);


		std::list<pmatrix> next, prev;
		unsigned flag;
		functor cal_val, cal_dval;
		unsigned M, N;
		scalar* val;
		scalar* dval;


		matrix(unsigned M, unsigned N) :flag(0), cal_val(nullptr), cal_dval(nullptr),
			M(M), N(N), val(new scalar[M * N]), dval(new scalar[M * N]) {}

		template<unsigned CM, unsigned CN>
		matrix(scalar(&arr)[CM][CN]) : flag(0), cal_val(nullptr), cal_dval(nullptr),
			M(CM), N(CN), val(&arr), dval(new scalar[CM * CN]) {}

		template<typename F>
		matrix(unsigned M, unsigned N, F) : flag(0), cal_val(F::calculate), cal_dval(F::derivative),
			M(M), N(N), val(new scalar[M * N]), dval(new scalar[M * N]) {}

		matrix(const matrix&) = delete;

		void reset() {
			if (cal_val) memset(val, 0, M * N * sizeof(scalar));
			if (cal_dval) memset(dval, 0, M * N * sizeof(scalar));
			flag = 0;
			for (pmatrix& p : next) p->reset();
		}

		void calculate() {
			if (!flag) {
				for (pmatrix& p : next) p->calculate();
				if (cal_val) cal_val(M, N, val, dval, next);
				flag++;
			}
		}

		void derivative() {
			if (flag == 1) {
				for (pmatrix& p : prev) p->derivative();
				if (cal_dval) cal_dval(M, N, val, dval, prev);
				flag++;
			}
		}

		static inline void link_to(pmatrix a, pmatrix b) {
			a->next.push_back(b);
			b->prev.push_back(a);
		}

	};

	namespace operation {

		struct plus {
			static void calculate(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				for (matrix::pmatrix& ptr : li) for (unsigned i = 0; i < M * N; ++i) val[i] += ptr->val[i];
			}
			static void derivative(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				for (matrix::pmatrix& ptr : li) for (unsigned i = 0; i < M * N; ++i) ptr->dval[i] += dval[i];
			}
		};

		struct minus {
			static void calculate(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				scalar* pa = li.front()->val, * pb = li.back()->val;
				for (unsigned i = 0; i < M * N; ++i) val[i] += pa[i];
				for (unsigned i = 0; i < M * N; ++i) val[i] -= pb[i];
			}
			static void derivative(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				scalar* pa = li.front()->dval, * pb = li.back()->dval;
				for (unsigned i = 0; i < M * N; ++i) pa[i] += dval[i];
				for (unsigned i = 0; i < M * N; ++i) pb[i] -= dval[i];
			}
		};

		struct multiply {
			static void calculate(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				scalar* pa = li.front()->val, * pb = li.back()->val;
				for (unsigned k = 0, len = li.front()->N; k < len; ++k)
					for (unsigned i = 0; i < M; ++i)
						for (unsigned j = 0; j < N; ++j)
							val[i * N + j] += pa[i * len + k] * pb[k * N + j];
			}
			static void derivative(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				matrix::pmatrix pa = li.front(), pb = li.back();
				for (unsigned k = 0, len = li.front()->N; k < len; ++k)
					for (unsigned i = 0; i < M; ++i)
						for (unsigned j = 0; j < N; ++j) {
							pa->dval[i * len + k] += pb->val[k * N + j] * dval[i * N + j];
							pb->dval[k * N + j] += pa->val[i * len + k] * dval[i * N + j];
						}
			}
		};

		struct reshape {
			static void calculate(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				scalar* pa = li.front()->val;
				memcpy_s(val, M * N * sizeof(scalar), pa, M * N * sizeof(scalar));
			}
			static void derivative(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				scalar* pa = li.front()->dval;
				memcpy_s(pa, M * N * sizeof(scalar), dval, M * N * sizeof(scalar));
			}
		};

		template<unsigned seed = 5489U>
		struct random {
			static void calculate(unsigned M, unsigned N, scalar* val, scalar* dval, std::list<matrix::pmatrix>& li) {
				static std::default_random_engine engine(seed);
				std::uniform_real_distribution<scalar> distribution;
				for (unsigned i = 0; i < M * N; ++i) val[i] = distribution(engine);
			}
			static constexpr matrix::functor derivative = nullptr;
		};

	}
}