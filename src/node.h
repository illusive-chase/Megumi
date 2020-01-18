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

#include "operator.h"

namespace megumi {

	template<unsigned M, unsigned N>
	class node {
	public:
		using pmatrix = matrix::pmatrix;
		using functor = matrix::functor;
		template<unsigned A, unsigned B> friend class node;
		template<typename F, unsigned M, unsigned N> friend node<M, N> operator_node();

	private:
		pmatrix val;
		node(pmatrix ptr) :val(ptr) {}

	public:
		node(scalar(&arr)[M][N]) :val(new matrix(arr)) {}
		node(std::initializer_list<scalar> init_li) :val(new matrix(M, N)) {
			for (unsigned i = 0, len = std::min(init_li.size(), M * N); i < len; ++i) val->val[i] = init_li.begin()[i];
		}
		explicit node(scalar constant) :val(new matrix(M, N)) {
			for (unsigned i = 0, len = std::min(M, N); i < len; ++i) val->val[i] = constant;
		}
		node() :val(new matrix(M, N)) {}
		

		
		node<M, N> operator +(const node<M, N>& rhs) const {
			node<M, N> ret(std::make_shared<matrix>(M, N, operation::plus()));
			matrix::link_to(ret.val, val); matrix::link_to(ret.val, rhs.val);
			return ret;
		}

		node<M, N> operator -(const node<M, N>& rhs) const {
			node<M, N> ret(std::make_shared<matrix>(M, N, operation::minus()));
			matrix::link_to(ret.val, val); matrix::link_to(ret.val, rhs.val);
			return ret;
		}

		template<unsigned K>
		node<M, K> operator *(const node<N, K>& rhs) const {
			node<M, K> ret(std::make_shared<matrix>(M, K, operation::multiply()));
			matrix::link_to(ret.val, val); matrix::link_to(ret.val, rhs.val);
			return ret;
		}

		template<unsigned A, unsigned B>
		node<A, B> reshape() const {
			static_assert(A * B == M * N);
			node<A, B> ret(std::make_shared<matrix>(A, B, operation::reshape()));
			matrix::link_to(ret.val, val);
			return ret;
		}

		static node<M, N> random_node() {
			return node<M, N>(std::make_shared<matrix>(M, N, operation::random<>()));
		}

		inline void init() { val->reset(); }
		inline void fp() { val->calculate(); }

		void print(const char* label, std::ostream& os = std::cout) {
			os << label << '<' << M << ',' << N << ">: " << std::endl;
			for (unsigned i = 0; i < M; ++i) {
				os << ' ' << val->val[i * N];
				for (unsigned j = 1; j < N; ++j) os << ", " << val->val[i * N + j];
				os << std::endl;
			}
			os << std::endl;
		}

	};

	template<unsigned M,unsigned N>
	node<N, M> transpose(const node<M, N>& rhs) {
		return rhs.reshape<N, M>();
	}

}