
//#include "memory_planning_pi.h"

#include <torch/types.h> // torch::tensor

#include <torch/csrc/jit/passes/freeze_module.h>
//#include <torch/csrc/jit/passes/memory_planning.h>

#include <torch/csrc/jit/runtime/profiling_graph_executor_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <torch/csrc/jit/passes/remove_mutation.h> // RemoveMutation

#include <torch/csrc/jit/ir/irparser.h> // for parseIR

#include <torch/csrc/jit/ir/alias_analysis.h>

#include <torch/csrc/jit/passes/tensorexpr_fuser.h> // RemoveProfileNodesAndSpecializeTypes

#include <torch/csrc/jit/jit_log.h> // getHeader

#include <torch/csrc/jit/serialization/import.h> // module load

#include <torch/csrc/autograd/profiler_kineto.h>
//#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>

#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/api/module.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/ir/ir.h> // Graph
#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>
#include <c10/util/irange.h>
//#include <torch/csrc/jit/passes/memory_planning.h> // LiveRange

#include <torch/types.h> // torch::tensor

namespace torch {
namespace jit {


inline Stack createStack(std::vector<at::Tensor>&& list) {
  return Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}


inline std::string ReplaceString(
    std::string subject,
    const std::string& search,
    const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
  return subject;
};

// void test_models(std::string model_name, Strategy strat);

const int BATCH_N = 1;

inline std::vector<IValue> unet_stack() {
  auto input = at::randn({BATCH_N, 3, 256, 256}, at::kCPU);
  auto stack = createStack({input});
  return stack;
}

inline std::vector<IValue> dcgan_stack() {
  auto input = at::randn({BATCH_N, 64, 64, 64}, at::kCPU);
  auto stack = createStack({input});
  return stack;
}

using StackMaker = std::function<std::vector<IValue>()>;

inline std::vector<IValue> bert_stack() {
  auto tokens_tensor = torch::tensor(
      {{101,
        2040,
        2001,
        3958,
        27227,
        1029,
        102,
        3958,
        103,
        2001,
        1037,
        13997,
        11510,
        102}},
      {torch::kLong});
  auto segments_tensor = torch::tensor(
      {{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1}}, {torch::kLong});

  for (int i = 1; i < BATCH_N; ++i) {
    tokens_tensor = torch::cat({tokens_tensor, tokens_tensor}, 1);
    segments_tensor = torch::cat({segments_tensor, segments_tensor}, 1);
  }

  auto stack = createStack({tokens_tensor, segments_tensor});
  return stack;
}

inline std::vector<IValue> resnet_stack() {
  auto input = at::randn({BATCH_N, 3, 244, 244}, at::kCPU);
  auto stack = createStack({input});
  return stack;
}



std::pair<std::shared_ptr<Graph>, torch::jit::script::Module> get_graph_from_model(std::string pt_path) {
  torch::jit::script::Module module;

  module = torch::jit::load(pt_path);
  std::shared_ptr<Module> module_ptr;
  module.eval();
  module_ptr = std::make_shared<Module>(freeze_module(module));
  auto forward = module_ptr->get_method("forward");
  auto graph = forward.graph();
  graph->eraseInput(0);
  for (const auto& inp : graph->inputs()) {
    auto name = inp->debugName();
    if (name.find('.') != std::string::npos)
      inp->setDebugName(ReplaceString(name, ".", "_"));
  }
  for (const auto& node : graph->nodes()) {
    for (const auto& inp : node->inputs()) {
      auto name = inp->debugName();
      if (name.find('.') != std::string::npos)
        inp->setDebugName(ReplaceString(name, ".", "_"));
    }
    for (const auto& inp : node->outputs()) {
      auto name = inp->debugName();
      if (name.find('.') != std::string::npos)
        inp->setDebugName(ReplaceString(name, ".", "_"));
    }
  }
  for (const auto& inp : graph->outputs()) {
    auto name = inp->debugName();
    if (name.find('.') != std::string::npos)
      inp->setDebugName(ReplaceString(name, ".", "_"));
  }

  torch::jit::Inline(*graph);
  jit::RemoveTensorMutation(graph);
  return std::make_pair(graph, module);
}

void prep_test_models(std::string model_name) {
  std::map<std::string, StackMaker> models = {
      {"unet", unet_stack},
      {"resnet18", resnet_stack},
      {"resnet34", resnet_stack},
      {"resnet50", resnet_stack},
      {"resnet101", resnet_stack},
      {"resnet152", resnet_stack},
      {"dcgan", dcgan_stack},
      {"small_bert", bert_stack},
      {"bert", bert_stack},
  };
  // run once to type info
  std::vector<IValue> stack;
  stack = models[model_name]();
  auto g = get_graph_from_model(
      "/home/mlevental/dev_projects/pytorch_dev/memory_allocator/models/" +
      model_name + ".pt").first;

  auto pr = jit::ProfilingRecord::instrumentGraph(g);
  auto graph = pr->profiled_graph_;

  Code cd(graph, model_name);
  InterpreterState is{cd};
  is.run(stack);

  // plan
  ProfilingRecord::removeProfileCounter(graph->block());
  jit::RemoveProfileNodesAndSpecializeTypes(graph);
  graph->dump();
//  return graph;
}

TORCH_LIBRARY(my_ops, m) {
    m.def("prep_test_models",
           []() {
            return prep_test_models("resnet");
           });
}

}}