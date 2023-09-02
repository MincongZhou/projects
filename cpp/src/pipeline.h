#ifndef COMP6771_PIPELINE_H
#define COMP6771_PIPELINE_H

#include <type_traits>
#include <unordered_map>
#include <algorithm>
#include <any>
#include <cassert>
#include <concepts>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <vector>
#include <typeinfo>
#include <variant>
#include <unordered_set>
#include <unordered_map>

namespace ppl {
	namespace internal{

		template <typename... Ts, std::size_t... Is>
		std::vector<std::type_index> store_tuple_typeids(const std::tuple<Ts...>&, std::index_sequence<Is...>) {
			// Create a vector with the typeid of each type in the tuple
			std::vector<std::type_index> type_indices = {typeid(std::tuple_element_t<Is, std::tuple<Ts...>>) ...};
			return type_indices;
		}

		template <typename... Ts>
		std::vector<std::type_index> store_tuple_typeids(const std::tuple<Ts...>& t) {
			return store_tuple_typeids(t, std::index_sequence_for<Ts...>{});
		}



	}

 	//// 3.1 pipeline_error

 	// Errors that may occur in a pipeline.
 	enum class pipeline_error_kind {
 		// An expired node ID was provided.
 		invalid_node_id,
 		// Attempting to bind a non-existant slot.
 		no_such_slot,
 		// Attempting to bind to a slot that is already filled.
 		slot_already_used,
 		// The output type and input types for a connection don't match.
 		connection_type_mismatch,
 	};

 	struct pipeline_error : std::exception {
 		// Constructs an error with the given reason.
 		explicit pipeline_error(pipeline_error_kind kind);
 		// Returns: the kind of error we were constructed from.
 		auto kind() -> pipeline_error_kind;
 		// Returns: A string depending on the value of kind():
 		// invalid_node_id: Return "invalid node ID"
 		// no_such_slot: Return "no such slot"
 		// slot_already_used: Return "slot already used"
 		// connection_type_mismatch: Return "connection type mismatch"
 		[[nodiscard]] auto what() const noexcept -> const char * override;

 	 private:
 		pipeline_error_kind error_kind;
 	};

 	//// 3.2 node

 	// The result of a poll_next() operation.
 	enum class poll {
 		// A value is available.
 		ready,
 		// No value is available this time, but there might be one later.
 		empty,
 		// No value is available, and there never will be again:
 		// every future poll for this node will return `poll::closed` again.
 		closed,
 	};

 	class node {
 	 public:
 		// Returns: A human-readable name for the node.
 		// Notes: This is a pure virtual function,
 		// and must be overriden by derived classes.
 		virtual auto name() const -> std::string  = 0;
 		virtual ~node() = default;

 	 private:
 		// Process a single tick, preparing the next value.
 		// Returns: The state of this node; see poll.
 		// Notes: This is a pure virtual function,
 		// and must be overriden by derived classes.
 		virtual auto poll_next() -> poll = 0;
 		// Connect source as the input to the given slot.
 		// If source is nullptr, signifies that this node should disconnect the existing connection for slot.
		// A later call to connect with a non-null pointer will later fill that slot again.
 		// Preconditions: slot is a valid index, and
 		// source is either a pointer to a producer of the correct type, or nullptr.
 		// Notes: This is a pure virtual function, and must be overriden by derived classes.
 		virtual void connect(const node* source, int slot) = 0;

 		// You may add any other virtual functions you feel you may want here.


//		virtual std::unique_ptr<node> clone() const = 0;
 		friend class pipeline;
 	};

 	//// 3.3 producer
 	template <typename Output>
 	struct producer : node {
 		using output_type = Output;
//		explicit producer(const output_type& t):value_(t){}

		// Returns: an immutable reference to the node's constructed value.
		// Preconditions:  has been called and last returned .poll_next()poll::ready
		// Notes: This is a pure virtual function;
		// it must be overriden by derived classes.
 		virtual auto value() const -> const output_type& = 0; // only when `Output` is not `void`
	 private:
//		output_type value_;
//		std::unique_ptr<node> p_ = nullptr;
	};

 	// Specialization for void output_type
	// Because sink nodes produce no output,
	// you must specialise the producer type for when Output is void.
	// This specialisation should be identical to the normal template,
	// except that the value() function does not exist.
 	template <>
 	struct producer<void> : node {
 		using output_type = void;


 	};

 	//// 3.4 component
 	template <typename Input, typename Output>
 	struct component : producer<Output> {
 		using input_type = Input;
		using output_type = Output;
		// if output type is void, pass it into tuple with type std::monostate
		template <bool IncludeVoid>
		using io_tuple_type = std::tuple<input_type,
		      typename  std::conditional<IncludeVoid, std::monostate, output_type>::type>;
		io_tuple_type<std::is_same<output_type, void>::value> io_;
		int input_size = std::tuple_size_v<input_type>;
	 private:

 	};

 	//// 3.5 sink & source
 	//// sink
 	template <typename Input>
	// output type is void
 	struct sink : component<std::tuple<Input>, void> {
	 private:
//		[[maybe_unused]] std::tuple<Input> input_;
		std::tuple<Input> input_;
	};
	//// source
 	template <typename Output>
 	struct source : component<std::tuple<>, Output> {
		using output_type = Output;

 	 private:
 		void connect([[maybe_unused]] const node *source,[[maybe_unused]] int slot) override {
			// This should never be called, as a source should not have any input connections.
//			assert(false && "connect() should not be called on a source component");
 		}



	};


 	// The requirements that a type `N` must satisfy
 	// to be used as a component in a pipeline.
 	template <typename N>
 	//// 3.6.0
 	concept concrete_node = requires{
	    // publish the types it consumes through a public member type input_type
 		typename N::input_type;
	    // have a std::tuple input_type
//		requires std::is_same<std::tuple,typename N::input_type>;
//	    requires std::is_same_v<typename N::input_type,
//	                            std::tuple<typename std::tuple_element<0, typename N::input_type>::type>>;
	    // publish the types it produces through a public member type output_type
 		typename N::output_type;
	    // be derived from the node type
	    requires std::is_base_of_v<ppl::node, N>;
	    // also be derived from the appropriate producer type
	    // note that the requirement that a component be derived from node is automatically met if this requirement is met.
 		requires std::is_base_of_v <ppl::producer <typename N::output_type>, N>;
	    // not be an abstract class (i.e., we can construct it).
	    requires !std::is_abstract_v <N>;
//		std::tuple_size<typename N::input_type>::value > 0;
 	};

 	class pipeline {
 	 public:

 		//// 3.6.1 Types
 		// An opaque handle to a node in the pipeline.
	    // May be any type of your choice as long as it is "regular"
	    // that is, copyable, default-constructible, and equality-comparable.
	    // Note: we expect to be able to create a reasonable number of nodes.
	    // Your handle should be able to support at least 256 nodes in the pipeline.
	    // We refer to a node_id as "invalid" if it is not a valid handle;
	    // that is, it cannot be used to refer to a node that is currently in the pipeline.
 		using node_id = int/* unspecified */;

 		//// 3.6.2 Special Members
 		// The pipeline must be default constructible.
	    // The pipeline should not be copyable (any attempt to do so should be a compile error).
	    // The pipeline should be movable; after auto p2 = std::move(p1);,
	    // p2 should manage all the nodes and connections that p1 used to,
	    // and p1 should be left in a valid (but unspecified) empty state.
	    // In this state, the pipeline should logically contain 0 nodes.
	    // You may provide a destructor to clean up if necessary.

	    // The pipeline must be default constructible.
 		pipeline() = default;
	    // The pipeline should not be copyable (any attempt to do so should be a compile error).
 		pipeline(const pipeline &) = delete;
	    // The pipeline should be movable; after auto p2 = std::move(p1);,
// 		pipeline(pipeline&&) = default;
	    // Move constructor
		pipeline(pipeline&&) noexcept = default;

	    // // Delete the copy assignment operator
 		auto operator=(const pipeline &) -> pipeline& = delete;
	    // Move assignment operator
 		auto operator=(pipeline &&) -> pipeline& = default;
 		~pipeline() = default;

 		//// 3.6.3
 		template <typename N, typename... Args>
 		// Preconditions: N is a valid node type (see section 3.6.0),
		// and can be constructed from the arguments args.
 		    requires concrete_node<N> and std::constructible_from<N, Args...>
 		auto create_node(Args&& ...args) -> node_id{
			static_assert(std::is_base_of_v<node, N>, "N must be derived from the node type");


// 		    node_id id = next_node_id+1;
			// Generate a new unique node_id
			node_id new_id = generate_new_node_id();
			// Create a new node with the given type N and constructor arguments
			auto new_node = std::make_unique<N>(std::forward<Args>(args)...);

//			 Store the new node in the nodes_map with the generated node_id
//			nodes_map[new_id] = std::move(new_node);
			nodes_map[new_id] = std::make_unique<N>(std::forward<Args>(args)...);
			auto io = new_node.get()->io_;
//			auto n =static_cast<std::size_t> (new_node.get()->input_size);
			// Store the typeid of each type in the tuple into a vector
			std::vector<std::type_index> type_indices = internal::store_tuple_typeids(std::get<0>(io));
			// Store the output type at the last
			type_indices.emplace_back(typeid(std::get<1>(io)));

			type_info_map[new_id] = std::make_pair(type_indices,io);



			return new_id;
 		}
		// Remove the specified node from the pipeline.
		// Disconnects it from any nodes it is currently connected to.
 		void erase_node(node_id n_id){
 		    auto it = nodes_map.find(n_id);
 			if (it == nodes_map.end()){
 				throw pipeline_error(pipeline_error_kind::invalid_node_id);
 			}
			for (auto& kv : adjacencyMap){
				// Remove all pairs with a first element of n_id.
//				kv.second.remove_if([n_id](const auto& p){return p.first==n_id;});
				kv.second.erase(std::remove_if(kv.second.begin(), kv.second.end(),[n_id](const auto& p){return p.first==n_id;})
				                    , kv.second.end());
			}
			nodes_map.erase(n_id);
			adjacencyMap.erase(n_id);
			type_info_map.erase(n_id);
 		}

 		auto get_node(node_id n_id) -> node* {
 		    auto it = nodes_map.find(n_id);
 			if (it == nodes_map.end()){
//				std::cout<<"cannot find"<<std::endl;
 				return nullptr;
 			}

			return nodes_map[n_id].get();
 		};

		// Const version
		const node* get_node(node_id n_id) const {
			auto it = nodes_map.find(n_id);
			if (it == nodes_map.end()) {
				// std::cout << "cannot find" << std::endl;
				return nullptr;
			}
			return nodes_map.at(n_id).get();
		};

 		//// 3.6.4
 		// Connect src's output to dst's input for the given slot.
		//
		// Throws: in order, if either handle is invalid,
		// the destination node's slot is already full,
		// the slot number indicated by slot does not exist,
		// the source output type does not match the destination slot's input type,
		// throw the appropriate pipeline_error.
 		void connect(node_id src, node_id dst, int slot){
		    // Check if both src and dst node_ids are valid
			auto src_it = nodes_map.find(src);
			auto dst_it = nodes_map.find(dst);
			if (src_it == nodes_map.end() || dst_it == nodes_map.end()){
				throw pipeline_error(pipeline_error_kind::invalid_node_id);
			}
			node* source_node = src_it->second.get();
			node* dest_node = dst_it->second.get();

			if (find_connect(dst, slot)){
//				std::cout<<"find"<<std::endl;
				throw pipeline_error(pipeline_error_kind::slot_already_used);
			}

			// the slot number indicated by slot does not exist
//			auto n = get_tuple_size(type_info_map[dst]);
//			std::cout<<"n: "<< type_info_map[dst].first<< std::endl;
			if (slot < 0 || static_cast<std::size_t>(slot) >= (type_info_map[dst].first.size()-1)){
				throw pipeline_error(pipeline_error_kind::no_such_slot);
			}

			// the source output type does not match the destination slot's input type
			// Validate that the source output type matches the destination slot's input type
			if (type_info_map[src].first.back() != type_info_map[dst].first[static_cast<std::size_t>(slot)]){
				throw pipeline_error(pipeline_error_kind::connection_type_mismatch);
			}
			// Connect the source node's output to the destination node's input slot
			dest_node->connect(source_node, slot);
			addEdge(src,dst, slot);
 		};

		// Remove all immediate connections between the given two nodes.
		// If the provided nodes are not connected, nothing is done.
		// Throws: a pipeline_error if either handle is invalid.
 		void disconnect(node_id src, node_id dst){
			// Check if both src and dst node_ids are valid
			auto src_it = nodes_map.find(src);
			auto dst_it = nodes_map.find(dst);
			if (src_it == nodes_map.end() || dst_it == nodes_map.end()){
				throw pipeline_error(pipeline_error_kind::invalid_node_id);
			}
			// Disconnect the source node's output from the destination node's input slots
			auto src_it2 = adjacencyMap.find(src);
			if (src_it2 == adjacencyMap.end()){
				return;
			}

			// Remove all pairs with a first element = dst.
			adjacencyMap[src].erase(std::remove_if(adjacencyMap[src].begin(),adjacencyMap[src].end(),
			    [dst](const auto& p){return p.first==dst;}), adjacencyMap[src].end());

 		};
		// Returns: A list of all nodes immediately depending on src.
		// Each element is a pair (node, slot), where src's output
		// is connected to the given slot for the node.
		// Throws: A pipeline_error if source is invalid.
 		auto get_dependencies(node_id src) -> std::vector<std::pair<node_id, int>>{
			auto it = nodes_map.find(src);
			if (it == nodes_map.end()){
				throw pipeline_error(pipeline_error_kind::invalid_node_id);
			}
		    return adjacencyMap[src];
 		};

 		//// 3.6.5
 		// Preconditions: None.
		// Validate that this is a sensible pipeline. In particular:
		// All source slots for all nodes must be filled.
		// All non-sink nodes must have at least one dependent.
		// There is at least 1 source node.
		// There is at least 1 sink node.
		// There are no subpipelines i.e. completely disconnected sections of the dataflow from the main pipeline.
		// There are no cycles.
		// Returns: true if all above conditions hold, and false otherwise.
 		auto is_valid() -> bool{
		    // There is at least 1 source node.
			bool flag = false;
		    for (const auto& kv : type_info_map){
				if (kv.second.first.size()==1){ flag = true; break; }
			}
			if (!flag){ return false; }

			// There is at least 1 sink node.
			flag = false;
			for (const auto& kv : type_info_map){
				if (kv.second.first.back()==typeid(std::monostate)){ flag = true; break;}
			}


			// All source slots for all nodes must be filled.
			std::unordered_map<node_id, int> input_count_map;
			for (const auto& kv:nodes_map){
				input_count_map[kv.first] = 0;
			}
			for (const auto& kv:adjacencyMap){
				for (auto pair : kv.second){
					input_count_map[pair.first]+=1;
				}
			}

			for (const auto& kv:input_count_map){
//				std::cout<<get_input_num(kv.first)<<std::endl;
				if (kv.second != get_input_num(kv.first)){
					return false; }
			}
			if (!flag){  return false; }

			// There are no cycles
			flag = is_cyclic(adjacencyMap);
			if (flag){ return false; }

		    return true;
 		};

		// Preconditions: is_valid() is true.
		// Perform one tick of the pipeline.
		// Initially source nodes shall be polled, and will prepare a value.
		// According to the poll result:
		//
		// If the node is closed, close all nodes that depend on it.
		// If the node has no value, skip all nodes that depend on it.
		// Otherwise, the node has a value,
		// and all nodes that depend on it should be polled,
		// and so on recursively.
		// The tick ends once every node has been either polled, skipped, or closed.
		//
		//
		//
		// Returns: true if all sink nodes are now closed,
		// or false otherwise.
		//
		// Notes: you are allowed to (but don't have to)
		// avoid polling a node if all its dependent sink nodes are closed.
 		auto step() -> bool{
			if (!is_valid()){ return false; }
//			std::cout<<"step"<<std::endl;
			// Track the status of the sink nodes
			bool all_sinks_closed = false;
//			auto poll_status = poll::ready;
			// Helper function to recursively traverse the pipeline
//			std::function<void(node_id)> traverse_pipeline = [&](node_id current_id){
			std::function<void(node_id)> traverse_pipeline = [&](node_id current_id){
				node* current_node = get_node(current_id);
				auto poll_status = current_node->poll_next();
//				poll_status = current_node->poll_next();
				if (is_source_node(current_id) && poll_status==poll::closed){ all_sinks_closed = true;}
				for (auto& p : adjacencyMap[current_id]){
					// connect
					auto dst_id = p.first;
					int slot = p.second;
					node* src_node = get_node(current_id);
					node* dst_node = get_node(dst_id);
					dst_node->connect(src_node, slot);
					// skip
					if (poll_status == poll::empty){
						poll_status = current_node->poll_next();
					}
					// has value
					if (poll_status == poll::ready){
						traverse_pipeline(dst_id);
//						poll_status = current_node->poll_next();
					}
				}



			};

			for (const auto& [id, node_ptr] : nodes_map){
				if (is_source_node(id)){ traverse_pipeline(id); }
			}
			return all_sinks_closed;
 		};

 		void run(){
			if (!is_valid()){ return; }
 			while (!step()) {}
 		};

 		//// 3.6.6
 		friend std::ostream &operator<<(std::ostream &os, const pipeline &p){
			std::map<node_id, int> id_map;
			int num = 1;
			for (const auto &it : p.nodes_map) {
				id_map[it.first] = num;
				num+=1;
			}

			os << "digraph G {" << std::endl;

			// Print nodes
			for (const auto &[id, node_ptr] : p.nodes_map) {
				os << "  \"" << id_map[id] << " " << node_ptr->name() << "\"" << std::endl;
			}
//			for (const auto &[id, node_ptr] : p.nodes_map) {
//				os << "  \"" << id+1 << " " << node_ptr->name() << "\"" << std::endl;
//			}

			os << std::endl;

			// Print edges
			for (const auto &[src_id, connections] : p.adjacencyMap) {
				for (const auto &[dst_id, slot] : connections) {
					os << "  \"" << id_map[src_id] << " " << p.nodes_map.at(src_id)->name() << "\" -> \""
					   << id_map[dst_id] << " " << p.nodes_map.at(dst_id)->name() << "\"" << std::endl;
				}
			}
//			for (const auto &[src_id, connections] : p.adjacencyMap) {
//				for (const auto &[dst_id, slot] : connections) {
//					os << "  \"" << src_id+1 << " " << p.nodes_map.at(src_id)->name() << "\" -> \""
//					   << dst_id+1 << " " << p.nodes_map.at(dst_id)->name() << "\"" << std::endl;
//				}
//			}

			os << "}" << std::endl;
			return os;
 		};

 	 private:
		// Data structure to store nodes
		std::map<node_id, std::unique_ptr<node>> nodes_map;
		std::map<node_id, std::vector<std::pair<node_id, int>>> adjacencyMap;
		std::unordered_map<node_id,std::pair<std::vector<std::type_index> ,std::any>> type_info_map;
		void addEdge(int src, int dest, int slot) {
			adjacencyMap[src].emplace_back(std::make_pair(dest, slot));
		}


		bool find_connect(int dst,int slot) {
			for (const auto& kv : adjacencyMap){
				auto it = std::find(kv.second.begin(), kv.second.end(), std::make_pair(dst, slot));
				if (it != kv.second.end()){
//					std::cout<<"find"<<it->first<<std::endl;
					return true;
				}
//				if(kv.second == std::make_pair(dst, slot))
			}
			return false;
		}

		int get_input_num(node_id n_id){
//			std::cout<<"false"<<std::endl<<std::endl;
			return static_cast<int>(type_info_map[n_id].first.size()-1);
		}


		static node_id generate_new_node_id() {
			static node_id current_id = 0;
			return current_id++;
		}

		using Graph = std::map<node_id, std::vector<std::pair<node_id, int>>>;

		bool is_cyclic_helper(node_id node, const Graph &graph, std::unordered_set<node_id> &visited, std::unordered_set<node_id> &recursion_stack) {
			visited.insert(node);
			recursion_stack.insert(node);

			if (graph.find(node) != graph.end()) {
				for (const auto &[neighbor, _] : graph.at(node)) {
					if (visited.find(neighbor) == visited.end()) {
						if (is_cyclic_helper(neighbor, graph, visited, recursion_stack)) {
							return true;
						}
					} else if (recursion_stack.find(neighbor) != recursion_stack.end()) {
						return true;
					}
				}
			}

			recursion_stack.erase(node);
			return false;
		}

		bool is_cyclic(const Graph &graph) {
			std::unordered_set<node_id> visited;
			std::unordered_set<node_id> recursion_stack;

			for (const auto &[node, _] : graph) {
				if (visited.find(node) == visited.end()) {
					if (is_cyclic_helper(node, graph, visited, recursion_stack)) {
						return true;
					}
				}
			}

			return false;
		}

		bool is_source_node(node_id n_id){
			return type_info_map[n_id].first.size() == 1;
		}

		bool is_sink_node(node_id n_id){
			return type_info_map[n_id].first.back() == typeid(std::monostate);
		}



     };

}

 #endif  // COMP6771_PIPELINE_H
