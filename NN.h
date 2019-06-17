#pragma once
#include "Node.h"
#include "Operation.h"
#include "Optimizer.h"
#include <list>
#include <map>
#include <random>

namespace MegumiWrapped {
	using std::list;

	template<unsigned OM, unsigned ON, typename OV>
	class NNWrapped {
	public:
		TensorNode<OM, ON, OV>* output;
		OV sum_output[OM][ON];
		list<Node*> nodes;

		NNWrapped() :output(nullptr) { memset(sum_output, 0, sizeof sum_output); }
		~NNWrapped() {
			for (auto& it : nodes) delete it;
		}

		template<unsigned M, unsigned N, typename Value>
		TensorNode<M, N, Value>* addVariable(Value(*src)[M][N]) {
			nodes.push_back(new VariableNode<M, N, Value>(src));
			return static_cast<TensorNode<M, N, Value>*>(nodes.back());
		}

		template<unsigned M, unsigned N, typename Value>
		TensorNode<M, N, Value>* addConstant(Value(*src)[M][N]) {
			nodes.push_back(new ConstNode<M, N, Value>(src));
			return static_cast<TensorNode<M, N, Value>*>(nodes.back());
		}

		template<typename TensorNodeType1, template<typename> typename UnaryOperationType>
		typename UnaryOperationType<TensorNodeType1>::return_type* addOperation(TensorNodeType1* para1) {
			nodes.push_back(
				new OperationNode<typename UnaryOperationType<TensorNodeType1>::return_type>(para1, 
				std::function<void(typename UnaryOperationType<TensorNodeType1>::return_type*, TensorNodeType1*)>(UnaryOperationType<TensorNodeType1>::value),
				std::function<void(typename UnaryOperationType<TensorNodeType1>::return_type*, TensorNodeType1*)>(UnaryOperationType<TensorNodeType1>::gradient))
			);
			return static_cast<typename UnaryOperationType<TensorNodeType1>::return_type*>(nodes.back());
		}

		template<typename TensorNodeType1, typename TensorNodeType2, template<typename, typename> typename BinaryOperationType>
		typename BinaryOperationType<TensorNodeType1, TensorNodeType2>::return_type* addOperation(TensorNodeType1* para1, TensorNodeType2* para2)
		{
			nodes.push_back(
				new OperationNode<typename BinaryOperationType<TensorNodeType1, TensorNodeType2>::return_type> (para1, para2, 
					std::function<void(typename BinaryOperationType<TensorNodeType1, TensorNodeType2>::return_type*, TensorNodeType1*, TensorNodeType2*)>
					(BinaryOperationType<TensorNodeType1, TensorNodeType2>::value),
					std::function<void(typename BinaryOperationType<TensorNodeType1, TensorNodeType2>::return_type*, TensorNodeType1*, TensorNodeType2*)>
					(BinaryOperationType<TensorNodeType1, TensorNodeType2>::gradient))
			);
			return static_cast<typename BinaryOperationType<TensorNodeType1, TensorNodeType2>::return_type*>(nodes.back());
		}

		inline void setOutput(TensorNode<OM, ON, OV>* output) {
			this->output = output;
		}

		inline decltype(output->value)* eval() {
			for (auto& it : nodes) it->Reset(false);
			output->Output();
			return &output->value;
		}

		inline void backProp() {
			for (auto& it : nodes) it->Reset(true);
			for (unsigned i = 0; i < OM; ++i){
				for (unsigned j = 0; j < ON; ++j) {
					sum_output[i][j] += output->value[i][j];
					output->temp_grad_value[i][j] = 1;
				}
			}
			output->BackProp();
			for (auto& it : nodes) it->UpdateGrad();
		}

		template<typename Optimizer>
		void train(unsigned begin, unsigned end, Optimizer opt, std::list<std::function<void(unsigned)>>& funcs) {
			for (auto& it : nodes) it->SetGrad(opt.momentum_factor());
			memset(sum_output, 0, sizeof sum_output);
			unsigned back_times = 0;
			for (opt.init(begin, end); !opt.finish(); opt.run()) {
				for (auto& it : funcs) it(opt.id());
				eval();
				backProp();
				back_times++;
				if (opt.update()) {
					for (auto& it : nodes) it->Update(opt.rate() / back_times);
					back_times = 0;
				}
			}
		}
	};

}



namespace Megumi {

	template<unsigned M, unsigned N>
	class ShapeTrait {};

	template<unsigned M, unsigned N>
	ShapeTrait<M, N> shape() { return ShapeTrait<M, N>(); }


	class NN;

	template<typename TensorNodeType>
	class Symbol {
	private:
		NN& net;
		TensorNodeType* const node;
		int placeholder;

		Symbol(NN& net, TensorNodeType* node, int placeholder = 0) :net(net), node(node), placeholder(placeholder) {}

	public:
		friend class NN;
		template<typename> friend class Symbol;

		template<unsigned M, unsigned N, unsigned S>
		std::pair<unsigned, std::function<void(unsigned)>> operator =(float(*src)[S][M][N]) {
			return std::make_pair(
				S,
				[=](unsigned i) { static_cast<MegumiWrapped::VariableNode<M, N, float>*>(node)->src = &(*src)[i]; }
			);
		}

		template<typename TensorNodeType_, template<typename, typename> typename BinaryOperationType>
		inline Symbol<typename BinaryOperationType<TensorNodeType, TensorNodeType_>::return_type> 
			Execute(Symbol<TensorNodeType_>& other) {
#if defined(_DEBUG)
			_ASSERT(&net == &other.net);
#endif
			return Symbol<typename BinaryOperationType<TensorNodeType, TensorNodeType_>::return_type>(net, 
				net.nn.addOperation<TensorNodeType, TensorNodeType_, BinaryOperationType>(node, other.node));
		}

		template<template<typename> typename UnaryOperationType>
		inline Symbol<typename UnaryOperationType<TensorNodeType>::return_type> Execute() {
			return Symbol<typename UnaryOperationType<TensorNodeType>::return_type>(net,
				net.nn.addOperation<TensorNodeType, UnaryOperationType>(node));
		}


		template<typename TensorNodeType_>
		inline Symbol<typename MegumiWrapped::Add<TensorNodeType, TensorNodeType_>::return_type>
			operator +(Symbol<TensorNodeType_> other) {
			return Execute<TensorNodeType_, MegumiWrapped::Add>(other);
		}

		template<typename TensorNodeType_>
		inline Symbol<typename MegumiWrapped::MatMul<TensorNodeType, TensorNodeType_>::return_type>
			operator *(Symbol<TensorNodeType_> other) {
			return Execute<TensorNodeType_, MegumiWrapped::MatMul>(other);
		}

		template<typename TensorNodeType_>
		inline Symbol<typename MegumiWrapped::Equal<TensorNodeType, TensorNodeType_>::return_type>
			operator ==(Symbol<TensorNodeType_> other) {
			return Execute<TensorNodeType_, MegumiWrapped::Equal>(other);
		}
	};


	class FeedDict {
	public:
		std::list<std::function<void(unsigned)>> funcs;
		unsigned begin, end;

		FeedDict(unsigned begin, unsigned end) :begin(begin), end(end) {}
		FeedDict() :begin(0), end(1) {}

		FeedDict& operator =(std::initializer_list<std::pair<unsigned, std::function<void(unsigned)>>> paras) {
			funcs.clear();
			if (end == 1) end = paras.begin()->first;
			for (auto& i : paras) {
#if defined(_DEBUG)
				_ASSERT(end <= i.first);
#endif
				funcs.push_back(i.second);
			}
			return *this;
		}

	};


	class NN {
	private:
		MegumiWrapped::NNWrapped<1, 1, float> nn;
		std::map<std::string, MegumiWrapped::Node*> trace_nodes;
		float* temp_initializer;
		static FeedDict empty_feed_dict;
		std::default_random_engine random_engine;
	public:
		template<typename> friend class Symbol;

		NN() :temp_initializer(nullptr) {}

		template<unsigned M, unsigned N>
		Symbol<MegumiWrapped::TensorNode<M, N, float>> variable(float(*src)[M][N]) {
			return Symbol<MegumiWrapped::TensorNode<M, N, float>>(*this, nn.addVariable(src));
		}

		template<unsigned M, unsigned N>
		Symbol<MegumiWrapped::TensorNode<M, N, float>> placeholder() {
			return Symbol<MegumiWrapped::TensorNode<M, N, float>>(*this, nn.addVariable<M, N, float>(nullptr), -1);
		}

		template<unsigned M, unsigned N>
		Symbol<MegumiWrapped::TensorNode<M, N, float>> constant(float(*src)[M][N]) {
			return Symbol<MegumiWrapped::TensorNode<M, N, float>>(*this, nn.addConstant(src));
		}

		template<unsigned M, unsigned N>
		Symbol<MegumiWrapped::TensorNode<M, N, float>> variable(float(*src)[M][N], const std::string& name) {
			auto ptr = nn.addVariable(src);
			trace_nodes[name] = static_cast<MegumiWrapped::Node*>(ptr);
			return Symbol<MegumiWrapped::TensorNode<M, N, float>>(*this, ptr);
		}

		template<unsigned M, unsigned N>
		Symbol<MegumiWrapped::TensorNode<M, N, float>> placeholder(const std::string& name) {
			auto ptr = nn.addVariable<M, N, float>(nullptr);
			trace_nodes[name] = static_cast<MegumiWrapped::Node*>(ptr);
			return Symbol<MegumiWrapped::TensorNode<M, N, float>>(*this, ptr, -1);
		}

		template<unsigned M, unsigned N>
		Symbol<MegumiWrapped::TensorNode<M, N, float>> constant(float(*src)[M][N], const std::string& name) {
			auto ptr = nn.addConstant(src);
			trace_nodes[name] = static_cast<MegumiWrapped::Node*>(ptr);
			return Symbol<MegumiWrapped::TensorNode<M, N, float>>(*this, ptr);
		}

		template<typename SymbolType>
		inline void trace(SymbolType sym, const std::string& name) {
			trace_nodes[name] = sym.node;
		}

		inline void showTrace() {
			for (auto& pr : trace_nodes) pr.second->Print(pr.first);
		}

		template<typename SymbolType, typename Optimizer>
		void train(SymbolType output, unsigned epoch_num, unsigned epoch_times, Optimizer opt, FeedDict& fd = empty_feed_dict) {
#if defined(_DEBUG)
			_ASSERT(&output.net == this);
#endif
			opt.set_random_engine(&random_engine);
			nn.setOutput(output.node);
			for (unsigned i = 0; i < epoch_num; ++i) {
				std::cout << "Epoch " << i * epoch_times + 1 << " begins..." << std::endl;
				for (unsigned j = 0; j < epoch_times; ++j) {
					nn.train(fd.begin, fd.end, opt, fd.funcs);
				}
				std::cout << "Epoch " << (i + 1) * epoch_times << " ends.\n" << std::endl;
				std::cout << "Loss: " << nn.sum_output[0][0] << std::endl;
				showTrace();
				std::cout << "****************************" << std::endl;
			}
		}

		template<typename SymbolType>
		void test(SymbolType output, FeedDict& fd = empty_feed_dict) {
#if defined(_DEBUG)
			_ASSERT(&output.net == this);
#endif
			nn.setOutput(output.node);
			std::cout << "Test begins..." << std::endl;
			nn.train(fd.begin, fd.end, NoneOptimizer(), fd.funcs);
			std::cout << "Test ends..." << std::endl;
			std::cout << "Loss: " << nn.sum_output[0][0] << std::endl;
			showTrace();
			std::cout << "****************************" << std::endl;
		}

		inline void set_seed(unsigned seed) {
			random_engine.seed(seed);
		}

		template<unsigned M, unsigned N>
		auto zero_tensor()->float(*)[M][N]{
			if (temp_initializer) delete[] temp_initializer;
		    temp_initializer = new float[M*N];
		    memset(temp_initializer, 0, sizeof(float)*M*N);
		    return (float(*)[M][N])temp_initializer;
		}

		template<unsigned M, unsigned N>
		auto bias_tensor(float value)->float(*)[M][N]{
			if (temp_initializer) delete[] temp_initializer;
			temp_initializer = new float[M*N];
			for (int i = 0; i < M; ++i) {
				for (int j = 0; j < N; ++j) {
					temp_initializer[i*N + j] = value;
				}
			}
			return (float(*)[M][N])temp_initializer;
		}

		template<unsigned M, unsigned N>
		auto bernoulli_tensor(float pa, float a = 1.0, float b = 0.0)->float(*)[M][N]{
			if (temp_initializer) delete[] temp_initializer;
		    temp_initializer = new float[M*N];
			std::bernoulli_distribution distribution(pa);
			for (unsigned i = 0; i < M; ++i) {
				for (unsigned j = 0; j < N; ++j) {
					temp_initializer[i*N + j] = distribution(random_engine) ? a : b;
				}
			}
		    return (float(*)[M][N])temp_initializer;
		}

		template<unsigned M, unsigned N>
		auto normal_tensor(float mean, float sigma)->float(*)[M][N]{
			if (temp_initializer) delete[] temp_initializer;
			temp_initializer = new float[M*N];
			std::normal_distribution<float> distribution(mean, sigma);
			for (unsigned i = 0; i < M; ++i) {
				for (unsigned j = 0; j < N; ++j) {
					temp_initializer[i*N + j] = distribution(random_engine);
				}
			}
			return (float(*)[M][N])temp_initializer;
		}
		

	};

	FeedDict NN::empty_feed_dict;

#define UNARY_REGISTER(NAME, AS) template<typename TensorNodeType>\
	inline auto AS(Symbol<TensorNodeType> para)->decltype(para.Execute<NAME>()) { return para.Execute<NAME>(); }
#define BINARY_REGISTER(NAME, AS) template<typename TensorNodeType, typename TensorNodeType_>\
	inline auto AS(Symbol<TensorNodeType> para, Symbol<TensorNodeType_> para_)->decltype(para.Execute<TensorNodeType_, NAME>(para_))\
    { return para.Execute<TensorNodeType_, NAME>(para_); }

	
	UNARY_REGISTER(MegumiWrapped::ArgMax, argmax)
	BINARY_REGISTER(MegumiWrapped::CrossEntropy, cross_entropy)
	UNARY_REGISTER(MegumiWrapped::Exp, exp)
	UNARY_REGISTER(MegumiWrapped::LeaklyReLU, leakly_rectified_linear_unit)
	UNARY_REGISTER(MegumiWrapped::Log, ln)
	BINARY_REGISTER(MegumiWrapped::MSE, mean_squared_error)
	UNARY_REGISTER(MegumiWrapped::Neg, neg)
	UNARY_REGISTER(MegumiWrapped::ReLU, rectified_linear_unit)
	UNARY_REGISTER(MegumiWrapped::Sigmoid, sigmoid)
	UNARY_REGISTER(MegumiWrapped::Softmax, softmax)
	UNARY_REGISTER(MegumiWrapped::Tanh, tanh)
	UNARY_REGISTER(MegumiWrapped::Trans, transpose)
	UNARY_REGISTER(MegumiWrapped::Reshape, reshape)
	
	template<typename TensorNodeType, unsigned M,unsigned N,unsigned MM,unsigned NN>
	inline auto max_pool(Symbol<TensorNodeType> para, ShapeTrait<M, N>(*para_shape)(), ShapeTrait<MM, NN>(*pool_shape)())->
		Symbol<typename MegumiWrapped::MaxPoolWrap<TensorNodeType, M, N, MM, NN>::MaxPool::return_type> {
		return para.Execute<MegumiWrapped::MaxPoolWrap<TensorNodeType, M, N, MM, NN>::MaxPoolImpl>();
	}

	template<typename TensorNodeType, typename TensorNodeType_, unsigned M, unsigned N, unsigned MM, unsigned NN>
	inline auto conv_valid(Symbol<TensorNodeType> para, ShapeTrait<M, N>(*para_shape)(), Symbol<TensorNodeType_> para_, ShapeTrait<MM, NN>(*para__shape)())->
		Symbol<typename MegumiWrapped::ConvWrap<TensorNodeType, TensorNodeType_, M, N, MM, NN>::ConvValid::return_type> {
		return para.Execute<TensorNodeType_, MegumiWrapped::ConvWrap<TensorNodeType, TensorNodeType_, M, N, MM, NN>::ConvValidImpl>(para_);
	}

	template<typename TensorNodeType, typename TensorNodeType_, unsigned M, unsigned N, unsigned MM, unsigned NN>
	inline auto conv_same(Symbol<TensorNodeType> para, ShapeTrait<M, N>(*para_shape)(), Symbol<TensorNodeType_> para_, ShapeTrait<MM, NN>(*para__shape)())->
		Symbol<typename MegumiWrapped::ConvWrap<TensorNodeType, TensorNodeType_, M, N, MM, NN>::ConvSame::return_type> {
		return para.Execute<TensorNodeType_, MegumiWrapped::ConvWrap<TensorNodeType, TensorNodeType_, M, N, MM, NN>::ConvSameImpl>(para_);
	}

#undef UNARY_REGISTER
#undef BINARY_REGISTER

}