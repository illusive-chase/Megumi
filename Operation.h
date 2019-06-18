#pragma once
#include "Node.h"
#include <algorithm>

namespace MegumiWrapped {

#define UNARY_OPERATION(CLASS_NAME, ...) template<typename TensorNodeType1>\
	class CLASS_NAME {\
	public:\
		using para = TensorNodeType1;\
		using return_type = __VA_ARGS__;\
		static void value(return_type* src, para* para1);\
		static void gradient(return_type* src, para* para1);\
	};
	
#define BINARY_OPERATION(CLASS_NAME, ...) template<typename TensorNodeType1, typename TensorNodeType2>\
	class CLASS_NAME {\
	public:\
		using para = TensorNodeType1;\
		using para_ = TensorNodeType2;\
		using return_type = __VA_ARGS__;\
		static void value(return_type* src, para* para1, para_* para2);\
		static void gradient(return_type* src, para* para1, para_* para2);\
	};

#define UNARY_VALUE(CLASS_NAME) template<typename TensorNodeType1>\
    void CLASS_NAME<TensorNodeType1>::value(return_type* src, para* para1)
#define UNARY_GRAD(CLASS_NAME) template<typename TensorNodeType1>\
    void CLASS_NAME<TensorNodeType1>::gradient(return_type* src, para* para1)
#define BINARY_VALUE(CLASS_NAME) template<typename TensorNodeType1, typename TensorNodeType2>\
    void CLASS_NAME<TensorNodeType1, TensorNodeType2>::value(return_type* src, para* para1, para_* para2)
#define BINARY_GRAD(CLASS_NAME) template<typename TensorNodeType1, typename TensorNodeType2>\
    void CLASS_NAME<TensorNodeType1, TensorNodeType2>::gradient(return_type* src, para* para1, para_* para2)
	


	BINARY_OPERATION(Equal, para)
	BINARY_VALUE(Equal) {
		para1->Output();
		para2->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para_::DM == M && para::DM == M && para::DN == N && para_::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) 
				src->value[i][j] = (typename return_type::value_type)((para2->value[i][j] == para1->value[i][j]) ? 1 : 0);
		}
	}

	BINARY_GRAD(Equal) {}

	UNARY_OPERATION(ArgMax, TensorNode<1, 1, typename para::value_type>)
	UNARY_VALUE(ArgMax) {
		para1->Output();
		constexpr unsigned N = para::DN;
		static_assert(para::DM == 1, "InvalidOperation.");
		int arg = 0;
		typename return_type::value_type arg_max = para1->value[0][0];
		for (int j = 1; j < N; ++j) {
			if (para1->value[0][j] > arg_max) arg_max = para1->value[0][arg = j];
		}
		src->value[0][0] = (typename return_type::value_type)arg;
	}

	UNARY_GRAD(ArgMax) {}

	//------------------------------------
	//------------------------------------
	//------------------------------------
	//------------------------------------
	//------------------------------------
	//------------------------------------


	UNARY_OPERATION(Neg, para)
	UNARY_VALUE(Neg) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) src->value[i][j] = -para1->value[i][j];
		}
	}

	UNARY_GRAD(Neg) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) para1->temp_grad_value[i][j] -= src->temp_grad_value[i][j];
		}
		para1->BackProp();
	}

	UNARY_OPERATION(Log, para)
	UNARY_VALUE(Log) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
#if defined(_DEBUG)
				_ASSERT(para1->value[i][j] > 0);
#endif
				src->value[i][j] = log(para1->value[i][j]);
			}
		}
	}

	UNARY_GRAD(Log) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				para1->temp_grad_value[i][j] += exp(-para1->value[i][j]) * src->temp_grad_value[i][j];
			}
		}
		para1->BackProp();
	}


	UNARY_OPERATION(Exp, para)
	UNARY_VALUE(Exp) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) src->value[i][j] = exp(para1->value[i][j]);
		}
	}

	UNARY_GRAD(Exp) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) para1->temp_grad_value[i][j] += para1->value[i][j] * src->temp_grad_value[i][j];
		}
		para1->BackProp();
	}

	UNARY_OPERATION(Softmax, para)
	UNARY_VALUE(Softmax) {
		para1->Output();
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == 1 && para::DN == N, "InvalidOperation.");
		typename return_type::value_type sum = 0;
		typename return_type::value_type max_v = para1->value[0][0];
		typename return_type::value_type exps[N];
		for (int j = 1; j < N; ++j) if (para1->value[0][j] > max_v) max_v = para1->value[0][j];
		for (int j = 0; j < N; ++j) sum += (exps[j] = exp(para1->value[0][j] - max_v));
		for (int j = 0; j < N; ++j) src->value[0][j] = exps[j] / sum;
	}

	UNARY_GRAD(Softmax) {
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == 1 && para::DN == N, "InvalidOperation.");
		for (int j = 0; j < N; ++j) para1->temp_grad_value[0][j] += src->value[0][j] * src->temp_grad_value[0][j] * (1 - src->value[0][j]);
		para1->BackProp();
	}

	UNARY_OPERATION(Sigmoid, para)
	UNARY_VALUE(Sigmoid) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) src->value[i][j] = 1 / (1 + exp(-para1->value[i][j]));
		}
	}

	UNARY_GRAD(Sigmoid) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) para1->temp_grad_value[i][j] += src->value[i][j] * src->temp_grad_value[i][j] * (1 - src->value[i][j]);
		}
		para1->BackProp();
	}

	UNARY_OPERATION(Tanh, para)
	UNARY_VALUE(Tanh) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				src->value[i][j] = 1 - 2 / (1 + exp(2 * para1->value[i][j]));
			}
		}
	}

	UNARY_GRAD(Tanh) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				para1->temp_grad_value[i][j] += (1 - para1->value[i][j] * para1->value[i][j]) * src->temp_grad_value[i][j];
			}
		}
		para1->BackProp();
	}

	UNARY_OPERATION(ReLU, para)
	UNARY_VALUE(ReLU) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				src->value[i][j] = para1->value[i][j] > 0 ? para1->value[i][j] : 0;
			}
		}
	}

	UNARY_GRAD(ReLU) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				if (src->value[i][j] > 0) para1->temp_grad_value[i][j] += src->temp_grad_value[i][j];
			}
		}
		para1->BackProp();
	}

	UNARY_OPERATION(LeaklyReLU, para)
	UNARY_VALUE(LeaklyReLU) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				src->value[i][j] = para1->value[i][j] > 0 ? para1->value[i][j] : 0.01 * para1->value[i][j];
			}
		}
	}

	UNARY_GRAD(LeaklyReLU) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == M && para::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				if (src->value[i][j] > 0)
					para1->temp_grad_value[i][j] += src->temp_grad_value[i][j] * (para1->value[i][j] > 0 ? 1 : 0.01);
			}
		}
		para1->BackProp();
	}

	UNARY_OPERATION(Trans, TensorNode<para::DN, para::DM, typename para::value_type>)
	UNARY_VALUE(Trans) {
		para1->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == N && para::DN == M, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) src->value[j][i] = para1->value[i][j];
		}
	}

	UNARY_GRAD(Trans) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para::DM == N && para::DN == M, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) para1->temp_grad_value[i][j] += src->temp_grad_value[j][i];
		}
		para1->BackProp();
	}

	BINARY_OPERATION(Add, para)
	BINARY_VALUE(Add) {
		para1->Output();
		para2->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para_::DM == M && para::DM == M && para::DN == N && para_::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) src->value[i][j] = para2->value[i][j] + para1->value[i][j];
		}
	}

	BINARY_GRAD(Add) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N = return_type::DN;
		static_assert(para_::DM == M && para::DM == M && para::DN == N && para_::DN == N, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				para1->temp_grad_value[i][j] += src->temp_grad_value[i][j];
				para2->temp_grad_value[i][j] += src->temp_grad_value[i][j];
			}
		}
		para1->BackProp();
		para2->BackProp();
	}

	BINARY_OPERATION(MatMul, TensorNode<para::DM, para_::DN, typename para::value_type>)
	BINARY_VALUE(MatMul) {
		para1->Output();
		para2->Output();
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N_ = return_type::DN;
		constexpr unsigned N = para::DN;
		static_assert(para_::DM == N && para::DM == M && para_::DN == N_, "InvalidOperation.");
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				typename para::value_type p1 = para1->value[i][j];
				for (int k = 0; k < N_; ++k) src->value[i][k] += p1 * para2->value[j][k];
			}
		}
	}

	BINARY_GRAD(MatMul) {
		constexpr unsigned M = return_type::DM;
		constexpr unsigned N_ = return_type::DN;
		constexpr unsigned N = para::DN;
		static_assert(para_::DM == N && para::DM == M && para_::DN == N_, "InvalidOperation.");
		memset(src->value, 0, sizeof(src->value));
		for (int i = 0; i < M; ++i) {
			for (int j = 0; j < N; ++j) {
				typename para::value_type p1 = para1->value[i][j];
				typename para::value_type ptg1 = para1->temp_grad_value[i][j];
				for (int k = 0; k < N_; ++k) {
					ptg1 += src->temp_grad_value[i][k] * para2->value[j][k];
					para2->temp_grad_value[j][k] += p1 * src->temp_grad_value[i][k];
				}
				para1->temp_grad_value[i][j] = ptg1;
			}
		}
		para1->BackProp();
		para2->BackProp();
	}

	BINARY_OPERATION(MSE, TensorNode<1, 1, typename para::value_type>)
	BINARY_VALUE(MSE) {
		para1->Output();
		para2->Output();
		static_assert(para_::DM == 1 && para::DN == para_::DN && para::DM == 1, "InvalidOperation.");
		src->value[0][0] = 0;
		for (int j = 0; j < para::DN; ++j) {
			typename return_type::value_type d = para2->value[0][j] - para1->value[0][j];
			src->value[0][0] += d * d;
		}
		src->value[0][0] /= para::DN;
	}

	BINARY_GRAD(MSE) {
		static_assert(para_::DM == 1 && para::DN == para_::DN && para::DM == 1, "InvalidOperation.");
		for (int j = 0; j < para::DN; ++j) {
			typename return_type::value_type d = 2 * (para2->value[0][j] - para1->value[0][j]) / para::DN;
			para1->temp_grad_value[0][j] += src->temp_grad_value[0][0] * (-d);
			para2->temp_grad_value[0][j] += src->temp_grad_value[0][0] * d;
		}
		para1->BackProp();
		para2->BackProp();
	}

	BINARY_OPERATION(CrossEntropy, TensorNode<1, 1, typename para::value_type>)
	BINARY_VALUE(CrossEntropy) {
		para1->Output();
		para2->Output();
		static_assert(para_::DM == 1 && para::DN == para_::DN && para::DM == 1, "InvalidOperation.");
		src->value[0][0] = 0;
		for (int j = 0; j < para::DN; ++j) {
			src->value[0][0] -= para1->value[0][j] * log(para2->value[0][j]) + (1 - para1->value[0][j]) * log(1 - para2->value[0][j]);
		}
	}

	BINARY_GRAD(CrossEntropy) {
		static_assert(para_::DM == 1 && para::DN == para_::DN && para::DM == 1, "InvalidOperation.");
		for (int j = 0; j < para::DN; ++j) {
			para1->temp_grad_value[0][j] -= src->temp_grad_value[0][0] * (log(para2->value[0][j]) -log(1 - para2->value[0][j]));
			para2->temp_grad_value[0][j] -= src->temp_grad_value[0][0] * (para1->value[0][j] / para2->value[0][j] - (1 - para1->value[0][j]) / (1 - para2->value[0][j]));
		}
		para1->BackProp();
		para2->BackProp();
	}

	UNARY_OPERATION(Reshape, TensorNode<1, para::DM * para::DN, typename para::value_type>)
	UNARY_VALUE(Reshape) {
		para1->Output();
		for (int i = 0; i < para::DM; ++i) {
			for (int j = 0; j < para::DN; ++j) {
				src->value[0][i * para::DN + j] = para1->value[i][j];
			}
		}
	}
	UNARY_GRAD(Reshape) {
		for (int i = 0; i < para::DM; ++i) {
			for (int j = 0; j < para::DN; ++j) {
				para1->temp_grad_value[i][j] += src->temp_grad_value[0][i * para::DN + j];
			}
		}
		para1->BackProp();
	}

	template<typename TensorNodeType1, unsigned M, unsigned N, unsigned MM, unsigned NN>
	class MaxPoolWrap {
	public:
		static_assert(M % MM == 0 && N % NN == 0, "InvalidOperation.");

		class MaxPool {
		public:
			using para = TensorNodeType1;
			static_assert(para::DN == M * N, "InvalidOperation.");
			using return_type = TensorNode<para::DM, para::DN / (MM * NN), typename para::value_type>;
			static void value(return_type* src, para* para1) {
				para1->Output();
				constexpr unsigned RM = M / MM;
				constexpr unsigned RN = N / NN;
				for (int channel = 0; channel < para::DM; ++channel) {
					for (int i = 0; i < RM; ++i) {
						for (int j = 0; j < RN; ++j) {
							typename para::value_type pool = para1->value[channel][i * MM * N + j * NN];
							for (int ii = 0; ii < MM; ++ii) {
								for (int jj = ii ? 0 : 1; jj < NN; ++jj) {
									pool = std::max(pool, para1->value[channel][(i * MM + ii) * N + (j * NN + jj)]);
								}
							}
							src->value[channel][i * RN + j] = pool;
						}
					}
				}
			}
			static void gradient(return_type* src, para* para1) {
				constexpr unsigned RM = M / MM;
				constexpr unsigned RN = N / NN;
				for (int channel = 0; channel < para::DM; ++channel) {
					for (int i = 0; i < RM; ++i) {
						for (int j = 0; j < RN; ++j) {
							int pool_pos = i * MM * N + j * NN;
							for (int ii = 0; ii < MM; ++ii) {
								for (int jj = ii ? 0 : 1; jj < NN; ++jj) {
									int pos = (i * MM + ii) * N + (j * NN + jj);
									if (para1->value[channel][pool_pos] < para1->value[channel][pos]) pool_pos = pos;
								}
							}
							para1->temp_grad_value[channel][pool_pos] += src->temp_grad_value[channel][i * RN + j];
						}
					}
				}
				para1->BackProp();
			}
		};

		template<typename T1>
		using MaxPoolImpl = MaxPool;
	};

	template<typename TensorNodeType1, typename TensorNodeType2, unsigned M, unsigned N, unsigned MM, unsigned NN>
	class ConvWrap {
	public:
		static_assert(M >= MM && N >= NN, "InvalidOperation.");

		class ConvValid {
		public:
			using para = TensorNodeType1;// TensorNode<C, M*N, V>;
			using para_ = TensorNodeType2;// TensorNode<C*CC, MM*NN, V>;
			static constexpr unsigned C = para::DM;
			static constexpr unsigned CC = para_::DM / C;
			static_assert(C * CC == para_::DM && para::DN == M * N && para_::DN == MM * NN, "InvalidOperation.");
			using return_type = TensorNode<CC, (M - MM + 1)*(N - NN + 1), typename para::value_type>;

			class ConvMap {
			public:
				unsigned index[MM*NN][(M - MM + 1)*(N - NN + 1)];
				ConvMap() {
					for (int i = 0; i < M - MM + 1; ++i) {
						for (int j = 0; j < N - NN + 1; ++j) {
							for (int ii = 0; ii < MM; ++ii) {
								for (int jj = 0; jj < NN; ++jj) {
									index[ii*NN + jj][i*(N - NN + 1) + j] = (i + ii)*N + j + jj;
								}
							}
						}
					}
				}
			};

			static const ConvMap conv_map;

			static void value(return_type* src, para* para1, para_* para2) {
				para1->Output();
				para2->Output();
				memset(src->value, 0, sizeof(src->value));
				
				for (int channel = 0; channel < C; ++channel) {
					for (int channel_ = 0; channel_ < CC; ++channel_) {
						int channel__ = channel * CC + channel_;
						for (int i = 0; i < (M - MM + 1)*(N - NN + 1); ++i) {
							for (int ii = 0; ii < MM*NN; ++ii) {
								src->value[channel_][i] +=
									para1->value[channel][conv_map.index[ii][i]] * para2->value[channel__][ii];
							}
						}
					}
				}
			}

			static void gradient(return_type* src, para* para1, para_* para2) {
				for (int channel = 0; channel < C; ++channel) {
					for (int channel_ = 0; channel_ < CC; ++channel_) {
						int channel__ = channel * CC + channel_;
						for (int i = 0; i < (M - MM + 1)*(N - NN + 1); ++i) {
							for (int ii = 0; ii < MM*NN; ++ii) {
								para1->temp_grad_value[channel][conv_map.index[ii][i]] +=
									src->temp_grad_value[channel_][i] * para2->value[channel__][ii];
								para2->temp_grad_value[channel*CC + channel_][ii] +=
									src->temp_grad_value[channel_][i] * para1->value[channel][conv_map.index[ii][i]];
							}
						}
					}
				}
				para1->BackProp();
				para2->BackProp();
			}

		};

		class ConvSame {
		public:
			using para = TensorNodeType1;// TensorNode<C, M*N, V>;
			using para_ = TensorNodeType2;// TensorNode<C*CC, MM*NN, V>;
			static constexpr unsigned C = para::DM;
			static constexpr unsigned CC = para_::DM / C;
			static constexpr int dM = MM >> 1;
			static constexpr int dN = NN >> 1;
			static_assert(C*CC == para_::DM && para::DN == M * N && para_::DN == MM * NN && para_::DN % 2 == 1, "InvalidOperation.");
			using return_type = TensorNode<CC, M * N, typename para::value_type>;

			class ConvMap {
			public:
				unsigned index[MM*NN][M*N];
				ConvMap() {
					for (int i = 0; i < M; ++i) {
						for (int j = 0; j < N; ++j) {
							for (int ii = 0; ii < MM; ++ii) {
								for (int jj = 0; jj <= NN; ++jj) {
									index[ii*NN + jj][i*N + j] =
										(i + ii < dM || j + jj < dN || i + ii >= dM + M || j + jj >= dN + N) ?
										-1 : ((i - dM + ii)*N + j - dN + jj);
								}
							}
						}
					}
				}
			};

			static const ConvMap conv_map;

			static void value(return_type* src, para* para1, para_* para2) {
				para1->Output();
				para2->Output();
				memset(src->value, 0, sizeof(src->value));
				for (int channel = 0; channel < C; ++channel) {
					typename para::value_type *p1 = para1->value[channel];
					for (int channel_ = 0; channel_ < CC; ++channel_) {
						int channel__ = channel * CC + channel_;
						typename para_::value_type *p2 = para2->value[channel__];
						typename return_type::value_type *s = src->value[channel_];
						for (int i = 0; i < M*N; ++i) {
							for (int ii = 0; ii < MM * NN; ++ii) {
								if (~conv_map.index[ii][i]) {
									s[i] +=
										p1[conv_map.index[ii][i]] * 
										p2[ii];
								}
							}
						}
					}
				}
			}

			static void gradient(return_type* src, para* para1, para_* para2) {
				for (int channel = 0; channel < C; ++channel) {
					typename para::value_type *p1 = para1->value[channel];
					typename para::value_type *ptg1 = para1->temp_grad_value[channel];
					for (int channel_ = 0; channel_ < CC; ++channel_) {
						int channel__ = channel * CC + channel_;
						typename para_::value_type *p2 = para2->value[channel__];
						typename para_::value_type *ptg2 = para2->temp_grad_value[channel__];
						typename return_type::value_type *stg = src->temp_grad_value[channel_];
						for (int i = 0; i < M*N; ++i) {
							for (int ii = 0; ii < MM * NN; ++ii) {
								if (~conv_map.index[ii][i]) {
									ptg1[conv_map.index[ii][i]] +=
										stg[i] * p2[ii];
									ptg2[ii] +=
										stg[i] * p1[conv_map.index[ii][i]];
								}
							}
						}
					}
				}
				para1->BackProp();
				para2->BackProp();
			}

		};

		template<typename T1, typename T2>
		using ConvValidImpl = ConvValid;

		template<typename T1, typename T2>
		using ConvSameImpl = ConvSame;

	};

	template<typename TensorNodeType1, typename TensorNodeType2, unsigned M, unsigned N, unsigned MM, unsigned NN>
	const typename ConvWrap<TensorNodeType1, TensorNodeType2, M, N, MM, NN>::ConvSame::ConvMap
		ConvWrap<TensorNodeType1, TensorNodeType2, M, N, MM, NN>::ConvSame::conv_map;

	template<typename TensorNodeType1, typename TensorNodeType2, unsigned M, unsigned N, unsigned MM, unsigned NN>
	const typename ConvWrap<TensorNodeType1, TensorNodeType2, M, N, MM, NN>::ConvValid::ConvMap
		ConvWrap<TensorNodeType1, TensorNodeType2, M, N, MM, NN>::ConvValid::conv_map;


#undef UNARY_OPERATION
#undef BINARY_OPERATION
#undef UNARY_VALUE
#undef BINARY_VALUE
#undef UNARY_GRAD
#undef BINARY_GRAD

	

}