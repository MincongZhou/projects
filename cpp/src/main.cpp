#include "./pipeline.h"
#include <iostream>

#include <map>
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
int main() {
	auto pipeline = ppl::pipeline{};
//	std::map<std::string, int> myMap;
//
//	// Add a key-value pair to the map
//	myMap.insert(std::pair<std::string, int>("foo", 42));
//
//	// Or, use the make_pair function to create the pair
//	myMap.insert(std::make_pair("bar", 123));
//
//	// Print the contents of the map
//	for (const auto& [key, value] : myMap) {
//		std::cout << "Key: " << key << ", Value: " << value << std::endl;
//	}

	return 0;
}
////using namespace std;
//enum roll_no {
//	satya = 70,
//	aakanskah = 73,
//	sanket = 31,
//	aniket = 05,
//	avinash = 68,
//	shreya = 47,
//	nikita = 69,
//};
//int main(){
//	ppl::pipeline pipeline;
////	int a = 5;
//	int b = 0;
//	try{
//		if (b==0){
//			throw ppl::pipeline_error(ppl::pipeline_error_kind::invalid_node_id);
//		}
//	}catch (const ppl::pipeline_error& e){
//		std::cout<< "Error1: "<<e.what()<<std::endl;
//	}
//	return 0;
//}
