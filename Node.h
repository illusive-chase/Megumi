#pragma once
#include <cstring>
#include <functional>
#include <iostream>
#include <string>

namespace MegumiWrapped {

	class Node {
	public:
		virtual void Output() = 0;
		virtual void BackProp() = 0;
		virtual void Reset(bool) = 0;
		virtual void Print(const std::string&) = 0;
		virtual void Update(float) = 0;
		virtual void SetGrad(float) = 0;
		virtual void UpdateGrad() = 0;
	};

	template<unsigned M, unsigned N, typename Value>
	class TensorNode :public Node {
	public:
		using value_type = Value;
		static constexpr unsigned DM = M;
		static constexpr unsigned DN = N;

		Value value[M][N];
		Value grad_value[M][N];
		Value temp_grad_value[M][N];
		bool calculated;

#if defined(_DEBUG)
		bool vis;
#endif

		TensorNode(int initial_value = 0) :calculated(false) {
			memset(value, initial_value, sizeof value);
			memset(grad_value, 0, sizeof grad_value);
			memset(temp_grad_value, 0, sizeof temp_grad_value);
#if defined(_DEBUG)
			vis = false;
#endif
		}

		virtual void Output() {}
		virtual void BackProp() {}
		virtual void Update(float) {}
		virtual void UpdateGrad() {}
		virtual void Print(const std::string&) {}

		inline void SetGrad(float alpha) {
			if (alpha == 0.0) {
				memset(grad_value, 0, sizeof grad_value);
			} else {
				for (int i = 0; i < M; ++i) {
					for (int j = 0; j < N; ++j) {
						grad_value[i][j] *= alpha;
					}
				}
			}
		}

		inline void Reset(bool back) {
			calculated = false;
#if defined(_DEBUG)
			vis = false;
#endif
			if (back) return;
			memset(temp_grad_value, 0, sizeof temp_grad_value);
		}

		

	};

	template<unsigned M, unsigned N, typename Value>
	class ConstNode :public TensorNode<M, N, Value> {
	public:
		using value_type = Value;
		using TensorNode<M, N, Value>::value;
		using TensorNode<M, N, Value>::grad_value;
		using TensorNode<M, N, Value>::temp_grad_value;
		using TensorNode<M, N, Value>::calculated;

		ConstNode(const decltype(&value) src) :TensorNode<M, N, Value>() {
			memcpy(value, *src, sizeof value);
		}

		inline void Output() {}
		inline void BackProp() {}

		inline void UpdateGrad() {
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					grad_value[i][j] += temp_grad_value[i][j];
				}
			}
		}

		inline void Update(float study_rate) {
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					value[i][j] -= grad_value[i][j] * study_rate;
				}
			}
		}

		void Print(const std::string& name) {
			std::cout << "Tensor " << M << "*" << N << " " << name << " :" << std::endl;
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) std::cout << value[i][j] << "(" << grad_value[i][j] << ") ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	};

	template<unsigned M, unsigned N, typename Value>
	class VariableNode :public TensorNode<M, N, Value> {
	public:
		using value_type = Value;
		using TensorNode<M, N, Value>::value;
		using TensorNode<M, N, Value>::grad_value;
		using TensorNode<M, N, Value>::temp_grad_value;
		using TensorNode<M, N, Value>::calculated;

		decltype(&value) src;

		VariableNode(decltype(&value) src) :TensorNode<M, N, Value>(), src(src) {}

		inline void Output() { if (!calculated) memcpy(value[0], src, sizeof value), calculated = true; }
		inline void BackProp() {}
		inline void Update(float study_rate) {}
		inline void UpdateGrad() {}

		void Print(const std::string& name) {
			std::cout << "Tensor " << M << "*" << N << " " << name << " :" << std::endl;
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) std::cout << value[i][j] << " ";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	};

	template<typename TensorNodeType>
	class OperationNode :public TensorNodeType {
	public:
		using value_type = typename TensorNodeType::value_type;
		using TensorNodeType::value;
		using TensorNodeType::grad_value;
		using TensorNodeType::temp_grad_value;
		using TensorNodeType::calculated;

#if defined(_DEBUG)
		using TensorNodeType::vis;
#endif
		std::function<void(TensorNodeType*)> func;
		std::function<void(TensorNodeType*)> grad;

		template<typename TensorNodeType1>
		OperationNode(TensorNodeType1* para1, std::function<void(TensorNodeType*, TensorNodeType1*)> func,
			std::function<void(TensorNodeType*, TensorNodeType1*)> grad) :TensorNodeType() {
			this->func = std::bind(func, std::placeholders::_1, para1);
			this->grad = std::bind(grad, std::placeholders::_1, para1);
		}

		template<typename TensorNodeType1, typename TensorNodeType2>
		OperationNode(TensorNodeType1* para1, TensorNodeType2* para2,
			std::function<void(TensorNodeType*, TensorNodeType1*, TensorNodeType2*)> func,
			std::function<void(TensorNodeType*, TensorNodeType1*, TensorNodeType2*)> grad)
			: TensorNodeType() {
			this->func = std::bind(func, std::placeholders::_1, para1, para2);
			this->grad = std::bind(grad, std::placeholders::_1, para1, para2);
		}

		inline void Output() {
			if (!calculated) {
#if defined(_DEBUG)
				_ASSERT(!vis);
				vis = true;
#endif
				memset(value, 0, sizeof value);
				func(this);
				calculated = true;
			}
		}

		inline void BackProp() {
			grad(this);
		}


		inline void Update(float study_rate) {}
		inline void UpdateGrad() {}

		void Print(const std::string& name) {
			std::cout << "Tensor " << TensorNodeType::DM << "*" << TensorNodeType::DN << " " << name << " :" << std::endl;
			for (int i = 0; i < TensorNodeType::DM; ++i) {
				for (int j = 0; j < TensorNodeType::DN; ++j) std::cout << value[i][j] << "(" << grad_value[i][j] << ")	";
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}

	};

}