#include "./pipeline.h"

#include <catch2/catch.hpp>
#include <iostream>

// store the sink output in a vector
std::vector<int> vec;
// a simple source that generates the numbers 1 through 10
struct simple_source : ppl::source<int> {
	int current_value = 0;

	simple_source() = default;

	auto name() const -> std::string override {
		return "SimpleSource";
	}

	auto poll_next() -> ppl::poll override {
		if (current_value >= 10)
			return ppl::poll::closed;
		++current_value;
		return ppl::poll::ready;
	}

	auto value() const -> const int& override {
		return current_value;
	}
};

// a simple sink that writes the numbers to std::cout
struct simple_sink : ppl::sink<int> {
	const ppl::producer<int>* slot0 = nullptr;

	simple_sink() = default;

	auto name() const -> std::string override {
		return "SimpleSink";
	}

	void connect(const ppl::node* src, int slot) override {
		if (slot == 0) {
			slot0 = static_cast<const ppl::producer<int>*>(src);
		}
	}

	auto poll_next() -> ppl::poll override {
		std::cout <<"value "<< slot0->value() << '\n';
		return ppl::poll::ready;
	}
};

TEST_CASE("create_node"){
	auto pipeline = ppl::pipeline{};
//	int a;
//	const auto source = pipeline.create_node<simple_source>();
//	const auto sink = pipeline.create_node<simple_sink>();
//	std::cout<<pipeline.get_node(source)->name()<<'\n'<<sink;
//	CHECK(source==0);
//	CHECK(sink==1);
	CHECK_NOTHROW(pipeline.create_node<simple_source>());

//	std::cout<<pipeline.get_node(source)->name()<<source<<"\n"<<sink<<"\n";
//	CHECK(source);

}

TEST_CASE("erase_node"){
	auto pipeline = ppl::pipeline{};
	const auto source = pipeline.create_node<simple_source>();
//	const auto sink = pipeline.create_node<simple_sink>();
//	std::cout<< source<<'\n';
	pipeline.get_node(source);
	CHECK_NOTHROW(pipeline.erase_node(source));
	CHECK_THROWS_WITH(pipeline.erase_node(source), "invalid node ID");
	CHECK_THROWS_WITH(pipeline.erase_node(123), "invalid node ID");
}

TEST_CASE("get_node"){
	auto pipeline = ppl::pipeline{};
	const auto src = pipeline.create_node<simple_source>();
//	const auto sink = pipeline.create_node<simple_sink>();
//	std::cout<<src<<"\n";
	SECTION("Valid node ID") {
		auto node_1 = pipeline.get_node(src);
		CHECK(node_1 != nullptr);
	}
	SECTION("Invalid node ID") {
		auto node_2 = pipeline.get_node(123);
		CHECK(node_2 == nullptr);
	}
	SECTION("Const get_node") {
		const auto& const_pipeline = pipeline;
		const auto node_3 = const_pipeline.get_node(src);
		CHECK(node_3 != nullptr);
	}
}

TEST_CASE("connect"){
	auto pipeline = ppl::pipeline{};
	const auto source = pipeline.create_node<simple_source>();
	const auto sink = pipeline.create_node<simple_sink>();
//	std::cout<<pipeline.get_node(source)->name()<<source<<sink;
	CHECK_NOTHROW(pipeline.connect(source, sink, 0));
	CHECK_THROWS_WITH(pipeline.connect(source, sink, 0), "slot already used");
	CHECK_THROWS_WITH(pipeline.connect(source, sink, -1), "no such slot");
//	CHECK_THROWS_WITH(pipeline.connect(source, sink, 1), "no such slot");
//	pipeline.connect(source, sink, 0);
//	CHECK_THROWS_WITH(pipeline.connect(source, sink, 0), "no such slot");
//	auto node_3 = pipeline.get_node(0);
//	std::cout<<node_3->name()<<'\n';
}

TEST_CASE("disconnect"){
	auto pipeline = ppl::pipeline{};
	const auto source = pipeline.create_node<simple_source>();
	const auto sink = pipeline.create_node<simple_sink>();
	//	std::cout<<pipeline.get_node(source)->name()<<source<<sink;
	pipeline.connect(source, sink, 0);
	CHECK_NOTHROW(pipeline.disconnect(source, sink));
	//	auto node_3 = pipeline.get_node(0);
	//	std::cout<<node_3->name()<<'\n';
}

TEST_CASE("get_dependencies"){
	auto pipeline = ppl::pipeline{};
	const auto source = pipeline.create_node<simple_source>();
	const auto sink = pipeline.create_node<simple_sink>();
	pipeline.connect(source, sink, 0);
	auto dependencies = pipeline.get_dependencies(source);
	CHECK(dependencies.size() == 1);
	CHECK(dependencies[0] == std::pair<int, int>(sink, 0));
}

TEST_CASE("is_valid"){
	SECTION("at least one sink node fail"){
		auto pipeline = ppl::pipeline{};
		pipeline.create_node<simple_source>();
		CHECK_FALSE(pipeline.is_valid());
	}
	SECTION("at least one source node fail"){
		auto pipeline = ppl::pipeline{};
		pipeline.create_node<simple_sink>();
		CHECK_FALSE(pipeline.is_valid());
	}
	SECTION("All non-sink nodes must have at least one dependent"){
		auto pipeline = ppl::pipeline{};
		pipeline.create_node<simple_source>();
		pipeline.create_node<simple_sink>();
		CHECK_FALSE(pipeline.is_valid());
	}
	SECTION("All source slots for all nodes must be filled"){
		// a component that output the even numbers
		struct component_1 : ppl::component<std::tuple<int, int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_1() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};
		auto pipeline = ppl::pipeline{};
		const auto src_node = pipeline.create_node<simple_source>();
		const auto component1 = pipeline.create_node<component_1>();
		const auto dst_node = pipeline.create_node<simple_sink>();
		pipeline.connect(src_node, component1, 0);
		pipeline.connect(component1, dst_node, 0);
		CHECK_FALSE(pipeline.is_valid());
	}
	SECTION("check cycle return false"){
		// a simple source that generates the numbers 1 through 10
		struct source_1 : ppl::source<int> {
			int current_value = 0;

			//   int current_value = 5;

			source_1() = default;

			auto name() const -> std::string override {
				return "Source";
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value == 5) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

		// a simple sink that writes the numbers to vec
		struct sink_1 : ppl::sink<int> {
			const ppl::producer<int>* slot0 = nullptr;

			sink_1() = default;

			auto name() const -> std::string override {
				return "Sink";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				//			std::cout << slot0->value() << ' ';
				vec.emplace_back(slot0->value());
				return ppl::poll::ready;
			}
		};

		// a component that output the even numbers
		struct component_1 : ppl::component<std::tuple<int, int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_1() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

		// a component that output the even numbers
		struct component_2 : ppl::component<std::tuple<int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_2() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

		// a component that output the even numbers
		struct component_3 : ppl::component<std::tuple<int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_3() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

//		struct component_4 : ppl::component<std::tuple<int, int>, int> {
//			int current_value = 0;
//			const ppl::producer<int>* slot0 = nullptr;
//
//			//   int current_value = 5;
//
//			component_4() = default;
//
//			auto name() const -> std::string override {
//				return "simple_component";
//			}
//
//			void connect(const ppl::node* src, int slot) override {
//				if (slot == 0) {
//					slot0 = static_cast<const ppl::producer<int>*>(src);
//				}
//			}
//
//			auto poll_next() -> ppl::poll override {
//				if (current_value >= 10)
//					return ppl::poll::closed;
//				++current_value;
//				if (current_value%2 == 1) return ppl::poll::empty;
//				return ppl::poll::ready;
//			}
//
//			auto value() const -> const int& override {
//				return current_value;
//			}
//		};



		auto pipeline = ppl::pipeline{};
		const auto source = pipeline.create_node<source_1>();
		const auto component1 = pipeline.create_node<component_1>();
		const auto component2 = pipeline.create_node<component_2>();
		const auto component3 = pipeline.create_node<component_3>();
//		const auto component4 = pipeline.create_node<component_4>();
		const auto sink = pipeline.create_node<sink_1>();
		pipeline.connect(source, component1, 0);
		pipeline.connect(component1, component2, 0);
		pipeline.connect(component2, component3, 0);
		pipeline.connect(component3, component1, 1);
		pipeline.connect(component3, sink, 0);
//		pipeline.connect(source, component4, 0);
//		pipeline.connect(component4, component3, 1);
//		std::cout<<pipeline;
//		pipeline.is_valid();
		CHECK_FALSE(pipeline.is_valid());
	}

	SECTION("valid"){
		// a simple source that generates the numbers 1 through 10
		struct source_1 : ppl::source<int> {
			int current_value = 0;

			//   int current_value = 5;

			source_1() = default;

			auto name() const -> std::string override {
				return "Source";
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value == 5) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

		// a simple sink that writes the numbers to vec
		struct sink_1 : ppl::sink<int> {
			const ppl::producer<int>* slot0 = nullptr;

			sink_1() = default;

			auto name() const -> std::string override {
				return "Sink";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				//			std::cout << slot0->value() << ' ';
				vec.emplace_back(slot0->value());
				return ppl::poll::ready;
			}
		};

		// a component that output the even numbers
		struct component_1 : ppl::component<std::tuple<int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_1() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

		// a component that output the even numbers
		struct component_2 : ppl::component<std::tuple<int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_2() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};

		// a component that output the even numbers
		struct component_3 : ppl::component<std::tuple<int>, int> {
			int current_value = 0;
			const ppl::producer<int>* slot0 = nullptr;

			//   int current_value = 5;

			component_3() = default;

			auto name() const -> std::string override {
				return "simple_component";
			}

			void connect(const ppl::node* src, int slot) override {
				if (slot == 0) {
					slot0 = static_cast<const ppl::producer<int>*>(src);
				}
			}

			auto poll_next() -> ppl::poll override {
				if (current_value >= 10)
					return ppl::poll::closed;
				++current_value;
				if (current_value%2 == 1) return ppl::poll::empty;
				return ppl::poll::ready;
			}

			auto value() const -> const int& override {
				return current_value;
			}
		};




		auto pipeline = ppl::pipeline{};
		const auto source = pipeline.create_node<source_1>();
		const auto component1 = pipeline.create_node<component_1>();
		const auto component2 = pipeline.create_node<component_2>();
		const auto component3 = pipeline.create_node<component_3>();
		//		const auto component4 = pipeline.create_node<component_4>();
		const auto sink = pipeline.create_node<sink_1>();
		pipeline.connect(source, component1, 0);
		pipeline.connect(component1, component2, 0);
		pipeline.connect(component2, component3, 0);
//		pipeline.connect(component3, component4, 0);
		pipeline.connect(component3, sink, 0);
		//		pipeline.connect(source, component4, 0);
		//		pipeline.connect(component4, component3, 1);
		//		std::cout<<pipeline;
		//		pipeline.is_valid();
		CHECK(pipeline.is_valid());
	}

}

TEST_CASE("step"){
	// a simple source that generates the numbers 1 through 10
	struct source_1 : ppl::source<int> {
		int current_value = 0;

		//   int current_value = 5;

		source_1() = default;

		auto name() const -> std::string override {
			return "Source";
		}

		auto poll_next() -> ppl::poll override {
			if (current_value >= 10)
				return ppl::poll::closed;
			++current_value;
			if (current_value == 5) return ppl::poll::empty;
			return ppl::poll::ready;
		}

		auto value() const -> const int& override {
			return current_value;
		}
	};

	// a simple sink that writes the numbers to vec
	struct sink_1 : ppl::sink<int> {
		const ppl::producer<int>* slot0 = nullptr;

		sink_1() = default;

		auto name() const -> std::string override {
			return "Sink";
		}

		void connect(const ppl::node* src, int slot) override {
			if (slot == 0) {
				slot0 = static_cast<const ppl::producer<int>*>(src);
			}
		}

		auto poll_next() -> ppl::poll override {
//			std::cout << slot0->value() << ' ';
			vec.emplace_back(slot0->value());
			return ppl::poll::ready;
		}
	};

	// a component that output the even numbers
	struct component_1 : ppl::component<std::tuple<int>, int> {
		int current_value = 0;
		const ppl::producer<int>* slot0 = nullptr;

		//   int current_value = 5;

		component_1() = default;

		auto name() const -> std::string override {
			return "simple_component";
		}

		void connect(const ppl::node* src, int slot) override {
			if (slot == 0) {
				slot0 = static_cast<const ppl::producer<int>*>(src);
			}
		}

		auto poll_next() -> ppl::poll override {
			if (current_value >= 10)
				return ppl::poll::closed;
			++current_value;
			if (current_value%2 == 1) return ppl::poll::empty;
			return ppl::poll::ready;
		}

		auto value() const -> const int& override {
			return current_value;
		}
	};

	auto pipeline = ppl::pipeline{};

	const auto source = pipeline.create_node<source_1>();
	const auto component = pipeline.create_node<component_1>();
	const auto sink = pipeline.create_node<sink_1>();

	pipeline.connect(source, component, 0);
	pipeline.connect(component, sink, 0);
	vec.clear();
	pipeline.step();
	std::vector<int> expected_vec = {2};
	CHECK(vec == expected_vec);

}

TEST_CASE("run"){
	// a simple source that generates the numbers 1 through 10
	struct source_1 : ppl::source<int> {
		int current_value = 0;

		//   int current_value = 5;

		source_1() = default;

		auto name() const -> std::string override {
			return "Source";
		}

		auto poll_next() -> ppl::poll override {
			if (current_value >= 10)
				return ppl::poll::closed;
			++current_value;
			if (current_value == 5) return ppl::poll::empty;
			return ppl::poll::ready;
		}

		auto value() const -> const int& override {
			return current_value;
		}
	};

	// a simple sink that writes the numbers to vec
	struct sink_1 : ppl::sink<int> {
		const ppl::producer<int>* slot0 = nullptr;

		sink_1() = default;

		auto name() const -> std::string override {
			return "Sink";
		}

		void connect(const ppl::node* src, int slot) override {
			if (slot == 0) {
				slot0 = static_cast<const ppl::producer<int>*>(src);
			}
		}

		auto poll_next() -> ppl::poll override {
			//			std::cout << slot0->value() << ' ';
			vec.emplace_back(slot0->value());
			return ppl::poll::ready;
		}
	};

	// a component that output the even numbers
	struct component_1 : ppl::component<std::tuple<int>, int> {
		int current_value = 0;
		const ppl::producer<int>* slot0 = nullptr;

		//   int current_value = 5;

		component_1() = default;

		auto name() const -> std::string override {
			return "simple_component";
		}

		void connect(const ppl::node* src, int slot) override {
			if (slot == 0) {
				slot0 = static_cast<const ppl::producer<int>*>(src);
			}
		}

		auto poll_next() -> ppl::poll override {
			if (current_value >= 10)
				return ppl::poll::closed;
			++current_value;
			if (current_value%2 == 1) return ppl::poll::empty;
			return ppl::poll::ready;
		}

		auto value() const -> const int& override {
			return current_value;
		}
	};

	auto pipeline = ppl::pipeline{};

	const auto source = pipeline.create_node<source_1>();
	const auto component = pipeline.create_node<component_1>();
	const auto sink = pipeline.create_node<sink_1>();

	pipeline.connect(source, component, 0);
	pipeline.connect(component, sink, 0);
	vec.clear();
	pipeline.run();
	std::vector<int> expected_vec = {2, 4, 6, 8, 10};
	CHECK(vec == expected_vec);
}
