// Copyright 2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NBLA_UTILS_NNB_HPP_
#define NBLA_UTILS_NNB_HPP_

#include <nbla/api_levels.hpp>
#include <nbla_utils/nnp.hpp>

namespace nbla {
namespace utils {
namespace nnb {
using namespace nnp;

enum { NN_BINARY_FORMAT_VERSION = 2 };

struct Settings {
  map<string, map<string, int>> functions;
  map<string, string> variables;
};

typedef int VariableId;
typedef int BufferIndex;
typedef int ApiLevel;

typedef int64_t SizeType;
typedef string FunctionArgCode;
typedef map<string, FunctionPtr> FunctionMap;
typedef map<string, VariablePtr> ParametersMap;
typedef map<string, Network::Variable> VariablesMap;
typedef map<string, VariablePtr> GeneratorsMap;
typedef map<BufferIndex, vector<VariableId>> BufferIndexMap;
typedef map<BufferIndex, SizeType> BufferSizeMap;
typedef map<VariableId, BufferIndex> BufferIdMap;
typedef unordered_set<string> TranposedWeightsSet;
typedef unordered_map<string, TranposedWeightsSet> ConvertContextMap;

struct NnablartInfo {
  int batch_size_;
  string network_name_;
  vector<string> input_variables_;
  vector<SizeType> input_buffer_sizes_;
  vector<string> output_variables_;
  vector<SizeType> output_buffer_sizes_;
  vector<string> param_variables_;
  vector<SizeType> variable_sizes_;
  BufferIndexMap variable_buffer_index_;
  BufferSizeMap variable_buffer_size_;
  BufferIdMap buffer_ids_;
  GeneratorsMap generator_variables_;
  ParametersMap parameters_;
  VariablesMap variables_;
  shared_ptr<Network> network_;
  FunctionMap function_info_;
  ConvertContextMap convert_context_;
};

class NBLA_API NnbExporter {
public:
  NnbExporter(const nbla::Context &ctx, Nnp &nnp, int batch_size = -1,
              int nnb_version = NN_BINARY_FORMAT_VERSION, int api_level = -1);

  void execute(const string &nnb_output_filename,
               const char *const settings_filename = nullptr,
               const string &default_type = "FLOAT32");

private:
  struct List;
  struct Variable;

  const nbla::Context &ctx_;
  NnablartInfo info_;
  int api_level_;
  int nnb_version_;
};

} // namespace nnb
} // namespace utils
} // namespace nbla

#endif // NBLA_UTILS_NNB_HPP_
