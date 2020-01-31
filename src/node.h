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


	class output {
	public:
		using pmatrix = matrix::pmatrix;
		using functor = matrix::functor;

	private:
		std::list<pmatrix> seq;

	public:
		output(std::list<pmatrix>&& li) :seq(li) {}
		void run() { seq.back()->reset(); for (pmatrix& p : seq) p->calculate(); }
	};

	template<unsigned M, unsigned N>
	class node {
	public:
		using pmatrix = matrix::pmatrix;
		using functor = matrix::functor;
		template<unsigned A, unsigned B> friend class node;
		template<typename F, unsigned A, unsigned B> friend node<A, B> operator_node(F, const node<A, B>&);
		template<unsigned A, unsigned B, unsigned C, unsigned D>
		friend node<A, B> partial_node(const node<C, D>&, const node<A, B>&);

	private:
		pmatrix val;
		node(pmatrix ptr) :val(ptr) {}

	public:
		node(scalar(&arr)[M][N]) :val(std::make_shared<matrix>(arr)) {}
		node(std::initializer_list<scalar> init_li) :val(std::make_shared<matrix>(M, N)) {
			for (unsigned i = 0, len = std::min(init_li.size(), M * N); i < len; ++i) val->val[i] = init_li.begin()[i];
		}
		explicit node(scalar constant) :val(std::make_shared<matrix>(M, N)) {
			for (unsigned i = 0, len = std::min(M, N); i < len; ++i) val->val[i] = constant;
		}
		node() :val(std::make_shared<matrix>(M, N)) {}
		

		
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

		output value() {
			val->reset();
			std::list<pmatrix> li;
			std::queue<pmatrix> zero_ind;
			std::stack<pmatrix> stk;
			stk.push(val);
			int cnt = 0;
			while (!stk.empty()) {
				pmatrix temp = stk.top();
				stk.pop();
				if (!temp->sflag) {
					cnt++;
					if (temp->next.empty()) {
						zero_ind.push(temp);
						temp->sflag = ~0;
					} else temp->sflag = (unsigned)temp->next.size();
					for (pmatrix& p : temp->next) stk.push(p);
				}
			}
			while (!zero_ind.empty()) {
				pmatrix temp = zero_ind.front();
				zero_ind.pop();
				li.push_back(temp);
				cnt--;
				for (pmatrix& p : temp->prev) {
					if (--p->sflag == 0) zero_ind.push(p);
				}
			}
			assert(cnt == 0);
			return output(std::move(li));
		}

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

	template<typename F, unsigned A, unsigned B> 
	node<A, B> operator_node(F, const node<A, B>& x) {
		node<A, B> ret(std::make_shared<matrix>(A, B, F()));
		matrix::link_to(ret.val, val);
		return ret;
	}

	template<unsigned M, unsigned N, unsigned A, unsigned B>
	node<M, N> partial_node(const node<A, B>& y, const node<M, N>& x) {
		y.val->reset();
		node<M, N> ret(std::make_shared<matrix>(M, N, operation::partial()));
		std::list<matrix::pmatrix> subgraph;
		std::stack<matrix::pmatrix> stk;
		std::queue<matrix::pmatrix> zero_ind;
		stk.push(y.val);
		zero_ind.push(x.val);
		const matrix* find = x.val.get();
		int cnt = 1;

		while (!stk.empty()) {
			matrix::pmatrix temp = stk.top();
			stk.pop();
			if (bool(temp)) {
				subgraph.push_back(temp);
				stk.push(nullptr);
				for (matrix::pmatrix& ptr : temp->next) {
					if (ptr.get() == find) {
						for (auto it = subgraph.begin(); it != subgraph.end(); ++it) {
							auto nxt = it;
							nxt++;
							if ((*it)->sflag) {
								if (nxt != subgraph.end() && (*nxt)->sflag == 0) (*it)->sflag++;
							} else {
								(*it)->sflag++;
								cnt++;
							}
						}
					} else stk.push(ptr);
				}
			} else subgraph.pop_back();
		}
		while (!zero_ind.empty()) {
			matrix::pmatrix temp = zero_ind.front();
			zero_ind.pop();
			temp->active();
			ret.val->next.push_front(temp);
			temp->prev.push_back(ret.val);
			cnt--;
			for (matrix::pmatrix& p : temp->prev) {
				if (--p->sflag == 0) zero_ind.push(p);
			}
		}
		assert(ret.val->next.size() && cnt == 0);
		return ret;
	}

}