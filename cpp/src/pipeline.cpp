 #include "./pipeline.h"

 ppl::pipeline_error::pipeline_error(ppl::pipeline_error_kind kind) : error_kind(kind) {}

 auto ppl::pipeline_error::kind() -> pipeline_error_kind {
 	return error_kind;
 }

 auto ppl::pipeline_error::what() const noexcept -> const char* {
 	switch (error_kind){
 	case pipeline_error_kind::invalid_node_id:
 		return "invalid node ID";
 	case pipeline_error_kind::no_such_slot:
 		return "no such slot";
 	case pipeline_error_kind::slot_already_used:
 		return "slot already used";
 	case pipeline_error_kind::connection_type_mismatch:
 		return "connection type mismatch";
 	default:
 		return "unknown error";
 	}
 }