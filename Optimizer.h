#pragma once
#include<vector>
#include<random>
#include<algorithm>
namespace Megumi {

	class BasicOptimizer {
	public:
		virtual void init(unsigned, unsigned) = 0;
		virtual void run() = 0;
		virtual bool finish() const = 0;
		virtual bool update() const = 0;
		virtual float rate() const = 0;
		virtual int id() const = 0;
		virtual void set_random_engine(std::default_random_engine*) = 0;
		virtual float momentum_factor() const = 0;
	};

	class NoneOptimizer :public BasicOptimizer {
	private:
		unsigned index;
		unsigned end;
		unsigned begin;
	public:
		NoneOptimizer() :index(0), begin(0), end(0) {}
		inline void init(unsigned begin, unsigned end) {
			this->begin = begin;
			this->end = end;
			this->index = begin;
		}
		inline void run() { ++index; }
		inline bool finish() const { return index >= end; }
		inline bool update() const { return false; }
		inline float rate() const { return 0.0; }
		inline int id() const { return index; }
		inline void set_random_engine(std::default_random_engine* engine) {}
		inline float momentum_factor() const { return 0.0; }
	};

	class BGDOptimizer :public BasicOptimizer {
	private:
		unsigned index;
		unsigned end;
		unsigned begin;
		const float study_rate;
		const float alpha;
	public:
		BGDOptimizer(float study_rate, float momentum_factor = 0.0) :index(0), begin(0), end(0), study_rate(study_rate), alpha(momentum_factor) {}
		inline void init(unsigned begin, unsigned end) {
			this->begin = begin;
			this->end = end;
			this->index = begin;
		}
		inline void run() { ++index; }
		inline bool finish() const { return index >= end; }
		inline bool update() const { return index + 1 == end; }
		inline float rate() const { return study_rate; }
		inline int id() const { return index; }
		inline void set_random_engine(std::default_random_engine* engine) {}
		inline float momentum_factor() const { return alpha; }
	};
	
	class SGDOptimizer :public BasicOptimizer {
	private:
		unsigned index;
		float study_rate;
		const float stable_study_rate;
		const float alpha;
		std::default_random_engine* random_engine;
		std::vector<unsigned> index_deque;
		unsigned batch_num;
	public:
		SGDOptimizer(unsigned batch_num, float study_rate, float momentum_factor = 0.0) :index(0), study_rate(study_rate),
			stable_study_rate(study_rate * 0.01), alpha(momentum_factor), random_engine(nullptr), batch_num(batch_num) {
		}
		
		inline void init(unsigned begin, unsigned end) {
#if defined(_DEBUG)
			_ASSERT(end >= batch_num + begin);
#endif
			for(unsigned i = begin; i < end; ++i) index_deque.push_back(i);
			std::shuffle(index_deque.begin(), index_deque.end(), *random_engine);
		}
		inline void run() { ++index; }
		inline bool finish() const { return index >= batch_num; }
		inline bool update() const { return index + 1 == batch_num; }
		inline float rate() const { return study_rate; }
		inline int id() const { return index_deque[index]; }
		inline void set_random_engine(std::default_random_engine* engine) { random_engine = engine; }
		inline float momentum_factor() const { return alpha; }
	};

}
