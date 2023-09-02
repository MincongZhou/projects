/**
* To help you get started and test that you've implemented the correct interface,
* here is a (very simple!) example usage of the API.
*
* It is important that this example compiles:
* otherwise, most of our automarking will fail on your assignment.
*
* We start off by including some libraries we'll need.
*/

#include "./pipeline.h"

#include <fstream>
#include <iostream>

/**
* First we'll make a very simple source,
* that just generates the numbers 1 through 10, and then stops.
* Note that `poll_next()` is used to advance to the next number (once),
* and then `value()` may be called many times repeatedly to retrieve that number.
* Because this is a `source`, we don't need to define `connect()`,
* as we have no inputs.
*/
std::vector<int> vec;
// a simple source that generates the numbers 1 through 10
struct simple_source : ppl::source<int> {
   int current_value = 0;

//   int current_value = 5;

   simple_source() = default;

   auto name() const -> std::string override {
	   return "SimpleSource";
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

/**
* We also need a sink to do something with our results.
* In this case, our sink will just write to std::cout.
* Because this is a sink, we don't need to define value(),
* as we have no outputs.
* That said, we _do_ need to define connect(), to learn about our input.
* Again, note that poll_next() performs the action.
*/

// a simple sink that writes the numbers to std::cout
struct simple_sink : ppl::sink<int> {
   const ppl::producer<int>* slot0 = nullptr;
//   std::vector<int> vec;

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
	   std::cout <<"value is "<< slot0->value() << '\n';
	   vec.emplace_back(slot0->value());
	   return ppl::poll::ready;
   }
};

struct simple_component : ppl::component<std::tuple<int>, int> {
   int current_value = 0;
   const ppl::producer<int>* slot0 = nullptr;

   //   int current_value = 5;

   simple_component() = default;

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

/**
* Note the unchecked downcast in `connect()`:
* it is up to your `pipeline` implementation to ensure that
* `connect()` is not called with bad arguments.
*
* From here, we just need to wire together our very simple pipeline,
* and run it to completion:
*/

int main() {
   auto pipeline = ppl::pipeline{};

   const auto source = pipeline.create_node<simple_source>();
   const auto source2 = pipeline.create_node<simple_source>();
   const auto component = pipeline.create_node<simple_component>();
   const auto sink = pipeline.create_node<simple_sink>();
   const auto sink2 = pipeline.create_node<simple_sink>();
   pipeline.connect(source, sink, 0);
   pipeline.connect(source2, component, 0);
   pipeline.connect(component, sink2, 0);
//   pipeline.connect(source, component, 0);
//   pipeline.connect(component, sink, 0);

//   std::cout<<source;
//   std::cout<<sink;
//   std::cout<<std::endl;

//   const auto component = pipeline.create_node<simple_component>();
//   std::cout<<component;
//   pipeline.connect(source, component, 0);
//   pipeline.connect(component, sink, 0);
//   if(pipeline.is_valid()){
//	   std::cout<<"pipeline is valid"<<std::endl;
//   }else{
//	   std::cout<<"pipeline is not valid"<<std::endl;
//   }
//   pipeline.step();
//   pipeline.step();
   pipeline.run();
//   for (auto i : vec) {
//	   std::cout << i << ' ';
//   }
   std::cout<<std::endl;
   std::cout<<pipeline<<std::endl;

   // Save the DOT representation to a file.
//   std::ofstream dot_file("pipeline.dot");
//   dot_file << pipeline;
//   dot_file.close();
//   auto output = std::ofstream("client.dot");
//   output << pipeline;
//   pipeline.run();
//   if (auto output = std::ofstream("client.dot")) {
//	   output << pipeline;
//   }
//   std::cout<<pipeline<<std::endl;

//   pipeline.run();
}

/**
* For the sake of the example, we also write out the dependency graph.
* This should create a file, `client.dot, in your current working directory.
* By running it through `dot -Tsvg client.dot -o client.svg`, you should see some
* nicely formatted output!
*/
