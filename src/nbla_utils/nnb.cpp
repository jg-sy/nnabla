// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

#include <nbla/context.hpp>
#include <nbla/function/transpose.hpp>
#include <nbla_utils/nnb.hpp>

#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace nbla {
namespace utils {
namespace nnb {

using nbla::utils::nnp::Executor;
using nbla::utils::nnp::Network;
using nbla::utils::nnp::Nnp;

typedef uint8_t NN_DATA_TYPE;
static constexpr NN_DATA_TYPE NN_DATA_TYPE_FLOAT = 0;
static constexpr NN_DATA_TYPE NN_DATA_TYPE_INT16 = 1;
static constexpr NN_DATA_TYPE NN_DATA_TYPE_INT8 = 2;
static constexpr NN_DATA_TYPE NN_DATA_TYPE_SIGN = 3;

static constexpr int FP_POS_MAX_INT16 = 15;
static constexpr int FP_POS_MAX_INT8 = 7;

// ----------------------------------------------------------------------
// ApiLevelInfo
// ----------------------------------------------------------------------

using ApiLevelFunctions = nbla::functions::ApiLevelFunctions;
using FunctionInfo = nbla::functions::FunctionInfo;
using FunctionInfoMap = nbla::functions::FunctionInfoMap;
using FunctionId = nbla::functions::FunctionId;

// sony/nnabla/python/src/nnabla/utils/converter/utils.py(243)
struct ApiLevelInfo {
  ApiLevelFunctions api_level_functions_;
  ApiLevel api_level_;
  FunctionInfoMap function_info_map_;

  ApiLevelInfo();
  ApiLevelInfo(const ApiLevel level);

  void set_api_level(const ApiLevel level);
  ApiLevel get_current_level() const { return api_level_; }
  vector<string> get_function_list() const;
  FunctionId get_function_id(const string &func_name) const;
  string get_argument_code(const string &func_name) const;
  string get_func_unique_name(const string &func_name) const;
};

ApiLevelInfo::ApiLevelInfo()
    : api_level_functions_(nbla::functions::api_level_functions()),
      api_level_(1), function_info_map_() {}

ApiLevelInfo::ApiLevelInfo(const ApiLevel level) : ApiLevelInfo() {
  set_api_level(level);
}

void ApiLevelInfo::set_api_level(const ApiLevel level) {
  api_level_ = api_level_functions_.size();
  if (level != -1 && level <= api_level_) {
    api_level_ = level;
  }
  FunctionInfoMap function_map;
  for (int i = 0; i < api_level_; ++i) {
    const FunctionInfoMap &api_level_functions = api_level_functions_[i];
    for (const std::pair<std::string, FunctionInfo> &api_level_it :
         api_level_functions) {
      function_map[api_level_it.first] = api_level_it.second;
    }
  }
  function_info_map_.swap(function_map);
}

vector<string> ApiLevelInfo::get_function_list() const {
  vector<string> ret;
  ret.reserve(function_info_map_.size());
  for (auto it : function_info_map_) {
    ret.push_back(it.first);
  }
  return ret;
}

FunctionId ApiLevelInfo::get_function_id(const string &func_name) const {
  FunctionId func_id = FunctionId::END_OF_FUNCTION_ID;
  auto it = function_info_map_.find(func_name);
  if (it != function_info_map_.end()) {
    func_id = it->second.func_id_;
  }
  return func_id;
}

string ApiLevelInfo::get_argument_code(const string &func_name) const {
  string arg_code;
  auto it = function_info_map_.find(func_name);
  if (it != function_info_map_.end()) {
    arg_code = it->second.argument_;
    if (arg_code == "Empty") {
      arg_code.clear();
    }
  }
  return arg_code;
}

string ApiLevelInfo::get_func_unique_name(const string &func_name) const {
  string unique_name;
  auto it = function_info_map_.find(func_name);
  if (it != function_info_map_.end()) {
    unique_name = it->first + "_" + it->second.argument_;
  }
  return unique_name;
}

// ----------------------------------------------------------------------
// MemoryBlock
// ----------------------------------------------------------------------

class MemoryBlockStorage;
class MemoryBlock {
public:
  friend MemoryBlockStorage;
  using storage_type = vector<uint8_t>;
  using storage_ptr = shared_ptr<storage_type>;
  using size_type = typename storage_type::size_type;
  using index_type = size_t;
  using offset_type = size_t;

  static constexpr index_type InvalidIndex = static_cast<index_type>(-1);
  static constexpr offset_type InvalidOffset = static_cast<offset_type>(-1);

  template <typename T, typename... Args>
  static MemoryBlock pack(const char *format, const T &value,
                          Args &&... values);

  template <typename T>
  static MemoryBlock pack(const char format, const T *const data, size_t size);

  template <typename T>
  static MemoryBlock pack(const char format, const vector<T> &data);

  MemoryBlock();

  MemoryBlock(MemoryBlock &&other) noexcept
      : storage_(), index_(InvalidIndex), size_(0) {
    this->operator=(std::move(other));
  }

  MemoryBlock &operator=(MemoryBlock &&other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      index_ = other.index_;
      size_ = other.size_;

      other.storage_ = {};
      other.index_ = InvalidIndex;
      other.size_ = 0;
    }
    return *this;
  }

  MemoryBlock &operator+=(MemoryBlock &&other);

  inline offset_type offset() const { return storage_.off_; }

  size_t fwrite(std::FILE *f) const;

private:
  struct StorageRef {

    operator bool() const { return !!ref_; }
    inline const uint8_t *const data() const { return ref_->data() + off_; }
    inline uint8_t *const data() { return ref_->data() + off_; }
    inline size_t size() const { return ref_->size(); }
    inline void resize(const size_t newsize) { ref_->resize(newsize, 0); }

    storage_ptr ref_;
    offset_type off_;
  };

  MemoryBlock(const index_type index, storage_ptr storage) noexcept
      : storage_({nullptr, InvalidOffset}), index_(InvalidIndex), size_(0) {
    move(index, storage);
  }

  void move(const index_type new_index, storage_ptr const new_storage);

  template <typename T_SRC, typename T_PACKED>
  void pack_data(const T_SRC *const src, const size_t size, const bool align);

  template <typename T>
  void pack_data(const char fmt, const T *src, const size_t size,
                 const bool align = false);

  // No-op base function to end recursion
  void pack_values(const char *) {}

  template <typename T, typename... Args>
  void pack_values(const char *format, const T &value, Args &&... values);

  template <size_t T_ALIGN = 4>
  inline static size_t align_size(const size_t size) {
    return ((size + (T_ALIGN - 1)) & ~(T_ALIGN - 1));
  }

  StorageRef storage_;
  index_type index_;
  size_type size_;
};

template <typename T, typename... Args>
inline MemoryBlock MemoryBlock::pack(const char *format, const T &value,
                                     Args &&... args) {
  MemoryBlock memory_block;
  memory_block.pack_values(format, value, std::forward<Args>(args)...);
  return memory_block;
}

template <typename T>
inline MemoryBlock MemoryBlock::pack(const char format, const T *const data,
                                     size_t size) {
  MemoryBlock memory_block;
  memory_block.pack_data(format, data, size);
  return memory_block;
}

template <typename T>
inline MemoryBlock MemoryBlock::pack(const char format, const vector<T> &data) {
  MemoryBlock memory_block;
  memory_block.pack_data(format, data.data(), data.size());
  return memory_block;
}

MemoryBlock::MemoryBlock()
    : MemoryBlock(InvalidIndex, std::make_shared<storage_type>()) {}

MemoryBlock &MemoryBlock::operator+=(MemoryBlock &&other) {
  NBLA_CHECK(this != &other, error_code::value,
             "Cannot append memory block data to itself");
  NBLA_CHECK(index_ == InvalidIndex, error_code::value,
             "Cannot append to memory block already assigned a block index");
  NBLA_CHECK(other.index_ == InvalidIndex, error_code::value,
             "Cannot append from memory block already assigned a block index");

  if (other.storage_ && other.size_) {
    if (storage_) {
      storage_.resize(storage_.size() + other.size_);
      memcpy(storage_.data() + size_, other.storage_.data(), other.size_);
      size_ += other.size_;
    } else {
      this->operator=(std::move(other));
    }
  }
  other.storage_ = {};
  other.size_ = 0;
  return *this;
}

void MemoryBlock::move(const index_type new_index,
                       storage_ptr new_storage_ptr) {
  NBLA_CHECK(!(storage_ && !new_storage_ptr), error_code::value,
             "Moving storage location will result in lost data");

  StorageRef new_storage = {new_storage_ptr, new_storage_ptr
                                                 ? new_storage_ptr->size()
                                                 : InvalidOffset};
  if (storage_ && size_) {
    new_storage.resize(new_storage.size() + size_);
    memmove(new_storage.data(), storage_.data(), size_);
  }
  storage_ = new_storage;
  index_ = new_index;
}

template <typename T, typename... Args>
inline void MemoryBlock::pack_values(const char *format, const T &value,
                                     Args &&... args) {
  pack_data(format[0], &value, 1, /*align=*/true);
  pack_values(format + 1, std::forward<Args>(args)...);
}

template <typename T_SRC, typename T_PACKED>
inline void MemoryBlock::pack_data(const T_SRC *const src, const size_t cnt,
                                   const bool align) {
  const size_t size_bytes = sizeof(T_PACKED) * cnt;
  const size_t alloc_size_bytes =
      size_bytes + (align ? (align_size(size_bytes) - size_bytes) : 0);
  storage_.resize(storage_.size() + alloc_size_bytes);

  uint8_t *const dst =
      storage_.data() +
      size_; // Needs to be defined after call to storage_.resize
  size_ += alloc_size_bytes;

  if (std::is_same<T_SRC, T_PACKED>::value) {
    memcpy(dst, src, size_bytes);
  } else {
    for (size_t i = 0, j = 0; i < size_bytes; i += sizeof(T_PACKED), ++j) {
      T_PACKED &packed = reinterpret_cast<T_PACKED &>(*(dst + i));
      packed = static_cast<T_PACKED>(src[j]);
    }
  }
}

template <typename T>
inline void MemoryBlock::pack_data(const char fmt, const T *src,
                                   const size_t size,
                                   const bool align /*= false*/) {
  switch (fmt) {
  case 'b':
    pack_data<T, int8_t>(src, size, align);
    break;
  case 'B':
    pack_data<T, uint8_t>(src, size, align);
    break;
  case 'c':
    pack_data<T, uint8_t>(src, size, align);
    break;
  case 'h':
    pack_data<T, int16_t>(src, size, align);
    break;
  case 'H':
    pack_data<T, uint16_t>(src, size, align);
    break;
  case 'i':
    pack_data<T, int32_t>(src, size, align);
    break;
  case 'l':
    pack_data<T, int32_t>(src, size, align);
    break;
  case 'I':
    pack_data<T, uint32_t>(src, size, align);
    break;
  case 'L':
    pack_data<T, uint32_t>(src, size, align);
    break;
  case 'q':
    pack_data<T, int64_t>(src, size, align);
    break;
  case 'Q':
    pack_data<T, uint64_t>(src, size, align);
    break;
  case 'f':
    pack_data<T, float>(src, size, align);
    break;
  case 'd':
    pack_data<T, double>(src, size, align);
    break;
  default:
    NBLA_ERROR(error_code::value, "Unsupported format type '%c", fmt);
    break;
  }
}

size_t MemoryBlock::fwrite(std::FILE *f) const {
  static constexpr size_t ElementSize = sizeof(storage_type::value_type);

  size_t bytes_written = 0;
  if (storage_ && size_) {
    bytes_written = ::fwrite(storage_.data(), ElementSize, size_, f);
  }
  return bytes_written;
}

// ----------------------------------------------------------------------
// MemoryBlockStorage
// ----------------------------------------------------------------------

typedef typename MemoryBlock::size_type MemoryBlockSize;
typedef typename MemoryBlock::index_type MemoryBlockIndex;
typedef typename MemoryBlock::offset_type MemoryOffset;

class MemoryBlockStorage {
public:
  MemoryBlockStorage();

  MemoryBlockIndex alloc(MemoryBlock &&memory_block);

  vector<MemoryOffset> memory_offsets() const;

  inline size_t size() const { return storage_->size(); }

  size_t fwrite(std::FILE *f) const;

private:
  MemoryBlock::storage_ptr storage_;
  vector<MemoryBlock> memory_blocks_;
};

MemoryBlockStorage::MemoryBlockStorage()
    : storage_(make_shared<MemoryBlock::storage_type>()), memory_blocks_() {}

MemoryBlockIndex MemoryBlockStorage::alloc(MemoryBlock &&memory_block) {
  const const MemoryBlockIndex index = memory_blocks_.size();
  memory_block.move(index, storage_);
  memory_blocks_.emplace_back(std::move(memory_block));
  return index;
}

vector<MemoryOffset> MemoryBlockStorage::memory_offsets() const {
  vector<MemoryOffset> offsets;
  offsets.reserve(memory_blocks_.size());

  for (const MemoryBlock &memory_block : memory_blocks_) {
    offsets.push_back(memory_block.offset());
  }
  return offsets;
}

size_t MemoryBlockStorage::fwrite(std::FILE *f) const {
  size_t bytes_written = 0;
  for (const MemoryBlock &memory_block : memory_blocks_) {
    bytes_written += memory_block.fwrite(f);
  }
  return bytes_written;
}

// ----------------------------------------------------------------------
// Helper functions
// ----------------------------------------------------------------------

template <typename T, T Lo, T Hi> inline const T &clamp_value(const T &value) {
  return (value < Lo) ? Lo : ((Hi < value) ? Hi : value);
}

NN_DATA_TYPE get_data_type_from_type_name(const std::string &type_name) {
  static const unordered_map<string, NN_DATA_TYPE> type_names_map = {
      {"FLOAT32", NN_DATA_TYPE_FLOAT},
      {"FIXED16", NN_DATA_TYPE_INT16},
      {"FIXED8", NN_DATA_TYPE_INT8},
      {"SIGN", NN_DATA_TYPE_SIGN}};

  return type_names_map.find(type_name)->second;
}

size_t get_type_size(const std::string &type_name, size_t default_value) {
  static const unordered_map<string, size_t> size_mapping = {
      {"FLOAT32", 4}, {"FIXED16", 2}, {"FIXED8", 1}};

  auto it = size_mapping.find(type_name);
  const size_t retVal = (it != size_mapping.end()) ? it->second : default_value;
  return retVal;
}

string type_to_pack_format(const Network::FunctionArg &arg) {
  string fmt;
  switch (arg.type_) {
  case Network::FunctionArg::BOOL:
    fmt = "B";
    break;
  case Network::FunctionArg::DOUBLE:
  case Network::FunctionArg::FLOAT:
    fmt = "f";
    break;
  case Network::FunctionArg::INT64:
    fmt = "i";
    break;
  case Network::FunctionArg::ARRAY_INT64:
  case Network::FunctionArg::SHAPE:
    fmt = "iI";
    break;
  case Network::FunctionArg::STRING:
    fmt = "i";
  case Network::FunctionArg::ARRAY_FLOAT:
  default:
    NBLA_ERROR(error_code::value, "Unhandled function arg type: %d", arg.type_);
    break;
  }
  return fmt;
}

// Based on numpy.packbits
inline uint8_t packbits(const uint8_t *const src) {
  const uint8_t packed_bits = ((bool)src[0]) << 7 | ((bool)src[1]) << 6 |
                              ((bool)src[2]) << 5 | ((bool)src[3]) << 4 |
                              ((bool)src[4]) << 3 | ((bool)src[5]) << 2 |
                              ((bool)src[6]) << 1 | ((bool)src[7]) << 0;
  return packed_bits;
}

// sony/nnabla/python/src/nnabla/utils/converter/utils.py(36)
Settings load_yaml_ordered(std::FILE *f) {
  // TODO
  return Settings();
}

// sony/nnabla/python/src/nnabla/utils/converter/utils.py(332)
int64_t calc_shape_size(Shape_t shape, int batch_size) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= (dim < 0) ? batch_size : dim;
  }
  return size;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(125)
void revise_buffer_size(NnablartInfo &info, const Settings &settings) {
  /*
  This function is used to revise buffer size, use byte
  as its unit, instead of data item.
  This is only used for nnb, not for csrc.
  When settings contains user customized data type, not pure
  FLOAT32, it affects the memory consumption.
  */

  const map<string, string> &var_map = settings.variables;

  const auto size_mapping = [&](const string &name) -> size_t {
    string type = "FLOAT32";
    auto it = var_map.find(name);
    if (it != var_map.end()) {
      type = it->second;
      type = type.substr(0, type.find('_'));
    }
    return get_type_size(type, /*default_value=*/4);
  };

  size_t buffer_index = 0;
  info.variable_sizes_.clear();
  info.variable_buffer_index_.clear();
  info.variable_buffer_size_.clear();
  info.buffer_ids_.clear();

  const vector<Network::Variable> variables = info.network_->get_variables();
  for (VariableId n = 0; n < (VariableId)variables.size(); ++n) {
    const Network::Variable &v = variables[n];
    const size_t byte_per_item = size_mapping(v.name);
    const int64_t size =
        calc_shape_size(v.shape, info.batch_size_) * byte_per_item;
    info.variable_sizes_.push_back(size);
    if (v.type == "Buffer") {
      info.variable_buffer_index_[buffer_index] = {n};
      vector<VariableId> &var_ids = info.variable_buffer_index_[buffer_index];
      for (VariableId vid : var_ids) {
        info.buffer_ids_[vid] = buffer_index;
      }
      info.variable_buffer_size_[buffer_index] = size;
      ++buffer_index;
    }
  }
}

// sony/nnabla/python/src/nnabla/utils/converter/utils.py(321)
shared_ptr<Executor> select_executor(Nnp &nnp) {
  shared_ptr<Executor> executor;
  vector<string> names = nnp.get_executor_names();
  if (!names.empty()) {
    executor = nnp.get_executor(names[0]);
  }
  return executor;
}

// sony/nnabla/python/src/nnabla/utils/converter/utils.py(325)
shared_ptr<Network> search_network(Nnp &nnp, const std::string &name) {
  shared_ptr<Network> network;
  for (const string &network_name : nnp.get_network_names()) {
    if (network_name == name) {
      network = nnp.get_network(network_name);
      break;
    }
  }
  return network;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(22)
VariablePtr generate_value(const nbla::Context &ctx,
                           const Executor::GeneratorVariable &x,
                           const Shape_t &shape,
                           const std::random_device::result_type seed = -1) {
  static std::random_device seed_gen;
  static std::uniform_real_distribution<> uniform(0.0, 1.0);
  static std::normal_distribution<> normal(0.0, 1.0);

  std::default_random_engine engine(seed == -1 ? seed_gen() : seed);
  VariablePtr v = make_shared<Variable>(shape);
  float_t *generator = v->cast_data_and_get_pointer<float_t>(ctx);
  if (x.type == "Normal") {
    for (int i = 0; i < v->size(); i++)
      generator[i] = x.multiplier * normal(engine);
  } else if (x.type == "Uniform") {
    for (int i = 0; i < v->size(); i++) {
      generator[i] = x.multiplier * uniform(engine);
    }
  } else if (x.type == "Constant") {
    for (int i = 0; i < v->size(); i++) {
      generator[i] = x.multiplier;
    }
  }
  return v;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(35)
bool initialize_nnablart_info(const nbla::Context &ctx, Nnp &nnp,
                              int batch_size, NnablartInfo &info) {
  shared_ptr<Executor> executor = select_executor(nnp);
  if (!executor) {
    printf("Failed to locate nnp executor\n");
    return false;
  }

  const string network_name = executor->network_name();
  shared_ptr<Network> network = search_network(nnp, network_name);
  if (!network) {
    printf("Network for executor [%s] is not found.\n", network_name.c_str());
    return false;
  }
  printf("Using network [%s].\n", network_name.c_str());

  info.batch_size_ = batch_size < 0 ? network->batch_size() : batch_size;
  info.network_name_ = executor->network_name();

  ParametersMap parameters;
  {
    vector<pair<string, VariablePtr>> params = nnp.get_parameters();
    parameters = ParametersMap(params.begin(), params.end());
  }

  VariablesMap variables;
  const vector<Network::Variable> network_variables = network->get_variables();
  {
    for (const Network::Variable &it : network_variables) {
      variables.insert({it.name, it});
    }
  }

  {
    vector<Executor::GeneratorVariable> generator_variables =
        executor->get_generator_variables();
    for (const Executor::GeneratorVariable &v : generator_variables) {
      const Network::Variable &v_info = variables[v.variable_name];
      Shape_t shape = v_info.shape;
      std::transform(shape.begin(), shape.end(), shape.begin(),
                     [b = info.batch_size_](const Shape_t::value_type &v) {
                       return (v < 0) ? b : v;
                     });
      VariablePtr data = generate_value(ctx, v, shape);
      info.generator_variables_[v.variable_name] = data;
    }
  }

  {
    vector<Executor::DataVariable> data_variables =
        executor->get_data_variables();
    info.input_variables_.reserve(data_variables.size());
    info.input_buffer_sizes_.reserve(data_variables.size());
    for (const Executor::DataVariable &dv : data_variables) {
      const Network::Variable &v = variables[dv.variable_name];

      info.input_variables_.push_back(dv.variable_name);
      info.input_buffer_sizes_.push_back(
          calc_shape_size(v.shape, info.batch_size_));
    }
  }
  {
    vector<Executor::OutputVariable> output_variables =
        executor->get_output_variables();
    info.output_variables_.reserve(output_variables.size());
    info.output_buffer_sizes_.reserve(output_variables.size());
    for (const Executor::OutputVariable &o : output_variables) {
      const Network::Variable &v = variables[o.variable_name];

      info.output_variables_.push_back(o.variable_name);
      info.output_buffer_sizes_.push_back(
          calc_shape_size(v.shape, info.batch_size_));
    }
  }
  {
    vector<Executor::ParameterVariable> parameter_variables =
        executor->get_parameter_variables();
    info.param_variables_.reserve(parameter_variables.size());
    for (const Executor::ParameterVariable &pv : parameter_variables) {
      info.param_variables_.push_back(pv.variable_name);
    }
  }
  {
    BufferIndex buffer_index = 0;
    info.variable_sizes_.reserve(network_variables.size());
    for (BufferIndex n = 0; n < network_variables.size(); ++n) {
      const Network::Variable &v = network_variables[n];
      const SizeType size = calc_shape_size(v.shape, info.batch_size_);
      info.variable_sizes_.push_back(size);
      if (v.type == "Buffer") {
        vector<VariableId> buffer_indices = {n};
        info.variable_buffer_index_.insert({buffer_index, buffer_indices});
        for (const VariableId &vid : buffer_indices) {
          info.buffer_ids_.insert_or_assign(vid, buffer_index);
        }

        auto it = info.variable_buffer_size_.find(buffer_index);
        if (it == info.variable_buffer_size_.end() || size > it->second) {
          info.variable_buffer_size_.insert_or_assign(buffer_index, size);
        }

        buffer_index++;
      }
    }
  }
  info.parameters_ = parameters;
  info.variables_ = variables;
  info.network_ = network;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(160)
void affine_transpose_weight(const std::vector<int> params,
                             const nbla::Context &ctx, NnablartInfo &info,
                             const Network::Function &func) {
  using Transpose = nbla::Transpose<float>;

  TranposedWeightsSet &transposed = info.convert_context_["Affine"];

  for (int idx : params) {
    const string &weight_name = func.inputs[idx];
    if (transposed.find(weight_name) != transposed.end()) {
      return;
    }

    VariablePtr w_data;
    if (info.parameters_.find(weight_name) != info.parameters_.end()) {
      w_data = info.parameters_[weight_name];
      transposed.insert(weight_name);
    } else {
      NBLA_ERROR(error_code::not_implemented,
                 "Affine weight is not transposed. Since it is not included in "
                 ".nntxt/.nnp");
    }

    VariablePtr var = Variable::create();
    Variables inputs = {w_data.get()};
    Variables outputs = {var.get()};

    FunctionPtr transpose_function = create_Transpose(ctx, {1, 0});
    transpose_function->setup(inputs, outputs);
    transpose_function->forward(inputs, outputs);
    NdArrayPtr data = var->data();
    w_data->reshape(data->shape(), /*force=*/true);
    w_data->set_data(data);
  }
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(186)
void pack_bin_conv_unused_weight(int index, const nbla::Context & /*ctx*/,
                                 NnablartInfo &info,
                                 const Network::Function &func) {
  string weight_name = func.inputs[index];
  VariablePtr w = info.parameters_[weight_name];
  NdArrayPtr data = w->data();
  NdArrayPtr truncated = data->narrow(0, 0, 1); // TRUNC TO 1
  w->reshape(truncated->shape(), /*force=*/true);
  w->set_data(truncated);
}

using PreprocessFunction = std::function<void(
    const nbla::Context &, NnablartInfo &, const Network::Function &)>;
using PreprocessFunctionMap = unordered_map<string, PreprocessFunction>;

#define NNB_PREPROCESSOR(preprocessor, ...)                                    \
  [](const nbla::Context &ctx, NnablartInfo &info,                             \
     const Network::Function &func) {                                          \
    preprocessor(__VA_ARGS__, ctx, info, func);                                \
  }

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(194)
static const PreprocessFunctionMap NNB_PREPROCESS_MAP = {
    {"Affine", NNB_PREPROCESSOR(affine_transpose_weight, {1})},
    {"BinaryConnectAffine", NNB_PREPROCESSOR(affine_transpose_weight, {1, 2})},
    {"BinaryWeightAffine", NNB_PREPROCESSOR(affine_transpose_weight, {1, 2})},
    {"BinaryWeightConvolution",
     NNB_PREPROCESSOR(pack_bin_conv_unused_weight, 1)},
    {"BinaryConnectConvolution",
     NNB_PREPROCESSOR(pack_bin_conv_unused_weight, 1)}};
#undef NNB_PREPROCESSOR

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/utils.py(214)
void preprocess_for_exporter(const nbla::Context &ctx, NnablartInfo &info) {
  vector<Network::Function> functions = info.network_->get_functions();
  for (const Network::Function &f : functions) {
    auto it = NNB_PREPROCESS_MAP.find(f.type);
    if (it != NNB_PREPROCESS_MAP.end()) {
      it->second(ctx, info, f);
    }
  }
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(18)
struct LifeSpan {
  bool needed_at(int func_idx) const {
    const bool needed =
        (begin_func_idx_ <= func_idx) && (end_func_idx_ >= func_idx);
    return needed;
  }

  int begin_func_idx_ = -1;
  int end_func_idx_ = -1;
};

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(29)
vector<LifeSpan> make_buf_var_lives(const NnablartInfo &info) {
  // buf_var_lives is to remember from when and until when each
  // Buffer Variables must be alive
  const vector<Network::Variable> network_variables =
      info.network_->get_variables();
  const vector<Network::Function> network_functions =
      info.network_->get_functions();
  const size_t num_var = network_variables.size();
  const size_t num_func = network_functions.size();
  const size_t buf_var_num = info.variable_buffer_index_.size();

  vector<LifeSpan> buf_var_lives(buf_var_num);
  unordered_map<string, VariableId> name_to_vidx;
  unordered_map<string, Network::Variable> name_to_var;
  name_to_vidx.reserve(num_var);
  name_to_var.reserve(num_var);
  for (VariableId i = 0; i < (VariableId)num_var; ++i) {
    Network::Variable var = network_variables[i];
    name_to_vidx.insert({var.name, i});
    name_to_var.insert({var.name, network_variables[i]});
  }

  // set LifeSpan.begin_func_idx and .end_func_idx along info.network
  const int final_func_idx = (int)num_func;
  for (int func_idx = 0; func_idx < (int)num_func; ++func_idx) {
    const Network::Function &func = network_functions[func_idx];
    vector<string> var_names;
    var_names.insert(var_names.end(), func.inputs.begin(), func.inputs.end());
    var_names.insert(var_names.end(), func.outputs.begin(), func.outputs.end());
    for (const string &var_name : var_names) {
      // No need to assign buffer for generator data
      // Ignore 'Parameter'
      if (name_to_var[var_name].type == "Buffer") {
        const VariableId var_idx = name_to_vidx[var_name];
        const BufferIndex buf_idx = info.buffer_ids_.find(var_idx)->second;
        LifeSpan &buf_var_life = buf_var_lives[buf_idx];

        // Only identify a Function which first refers to the Variable
        if (buf_var_life.begin_func_idx_ < 0) {
          auto it = std::find(info.input_variables_.begin(),
                              info.input_variables_.end(), var_name);
          buf_var_life.begin_func_idx_ =
              (it != info.input_variables_.end()) ? 0 : func_idx;
        }
        {
          auto it = std::find(info.output_variables_.begin(),
                              info.output_variables_.end(), var_name);
          buf_var_life.end_func_idx_ =
              (it != info.output_variables_.end()) ? final_func_idx : func_idx;
        }
      }
    }
  }
  return buf_var_lives;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(68)
size_t count_actual_buf(const NnablartInfo &info,
                        const vector<LifeSpan> &buf_var_lives) {
  size_t actual_buf_num = 0;
  const vector<Network::Function> network_functions =
      info.network_->get_functions();
  const size_t num_functions = network_functions.size();
  const size_t num_buf_var_lives = buf_var_lives.size();
  for (size_t func_idx = 0; func_idx < num_functions; ++func_idx) {
    size_t buf_num = 0;
    for (size_t buf_idx = 0; buf_idx < num_buf_var_lives; ++buf_idx) {
      const LifeSpan &buf_var_life = buf_var_lives[buf_idx];
      buf_num += (size_t)buf_var_life.needed_at(func_idx);
    }
    actual_buf_num = std::max(actual_buf_num, buf_num);
  }
  return actual_buf_num;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(99)
vector<SizeType>
compute_actual_buf_sizes(const NnablartInfo &info,
                         const vector<LifeSpan> &buf_var_lives) {
  const size_t actual_buf_num = count_actual_buf(info, buf_var_lives);
  // buf_size_array is to store size values of each actual buffer
  vector<SizeType> buf_size_array(actual_buf_num, 0);

  const size_t num_network_functions = info.network_->get_functions().size();
  for (size_t func_idx = 0; func_idx < num_network_functions; ++func_idx) {
    // tmp_size_array is size values when only focusing on a single Function
    vector<SizeType> tmp_size_array(actual_buf_num, -1);
    int crsr = 0;
    for (BufferIndex buf_idx = 0; buf_idx < buf_var_lives.size(); ++buf_idx) {
      const LifeSpan &buf_var_life = buf_var_lives[buf_idx];
      // Only focus on buffers used in this func
      if (buf_var_life.needed_at(func_idx)) {
        tmp_size_array[crsr] = info.variable_buffer_size_.find(buf_idx)->second;
        ++crsr;
      }
    }
    // Update sizes of actual buffers
    std::sort(tmp_size_array.begin(), tmp_size_array.end());
    for (size_t i = 0; i < actual_buf_num; ++i) {
      buf_size_array[i] = std::max(buf_size_array[i], tmp_size_array[i]);
    }
  }
  return buf_size_array;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(79)
vector<vector<BufferIndex>>
make_buf_var_refs(const NnablartInfo &info,
                  const vector<LifeSpan> &buf_var_lives) {
  // Returns buffer indices of buffers required in each Function
  const size_t num_network_functions = info.network_->get_functions().size();
  const size_t num_buf_var_lives = buf_var_lives.size();
  const size_t actual_buf_num = count_actual_buf(info, buf_var_lives);

  // buf_var_refs is to store buffer indices of buffers required in each
  // Function shape = (num_network_functions, actual_buf_num)
  vector<vector<BufferIndex>> buf_var_refs(
      num_network_functions, vector<BufferIndex>(actual_buf_num, -1));

  // fill buf_var_refs based on buf_var_lives
  for (size_t func_idx = 0; func_idx < num_network_functions; ++func_idx) {
    size_t crsr = 0;
    for (BufferIndex buf_idx = 0; buf_idx < (BufferIndex)num_buf_var_lives;
         ++buf_idx) {
      const LifeSpan &buf_var_life = buf_var_lives[buf_idx];

      // Only focus on buffers used in this func
      if (buf_var_life.needed_at(func_idx)) {
        buf_var_refs[func_idx][crsr] = buf_idx;
        ++crsr;
      }
    }
  }
  return buf_var_refs;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(124)
BufferIdMap
assign_actual_buf_to_variable(const NnablartInfo &info,
                              vector<SizeType> &actual_buf_sizes,
                              const vector<vector<BufferIndex>> &buf_var_refs) {
  // create a dictionary to store assignment of actual buffers to Variables
  const size_t num_network_functions = info.network_->get_functions().size();

  // vidx_to_abidx is short for variable index to actual buffer index
  BufferIdMap vidx_to_abidx;

  const size_t actual_buf_num = actual_buf_sizes.size();
  for (size_t func_idx = 0; func_idx < num_network_functions; ++func_idx) {
    // actual_assigned_flags is to remember if actual buffers are assigned or
    // not
    vector<bool> actual_assigned_flags(actual_buf_num, false);
    for (size_t ref_crsr = 0; ref_crsr < actual_buf_num; ++ref_crsr) {
      // minus buf_idx means the corresponding buffer is not needed
      BufferIndex buf_idx = buf_var_refs[func_idx][ref_crsr];
      if (buf_idx < 0) {
        continue;
      }

      // restore assignment determined in the previous func_idx
      const VariableId vidx =
          info.variable_buffer_index_.find(buf_idx)->second[0];
      auto it = vidx_to_abidx.find(vidx);
      if (it != vidx_to_abidx.end()) {
        const BufferIndex abidx = it->second;
        actual_assigned_flags[abidx] = true;
      } else {
        // determine assignment for this vidx in the following for loop
      }
    }

    // determine new assignments of actual buffers to Variables
    for (size_t ref_crsr = 0; ref_crsr < actual_buf_num; ++ref_crsr) {
      // minus buf_idx means the corresponding buffer is not needed
      BufferIndex buf_idx = buf_var_refs[func_idx][ref_crsr];
      if (buf_idx < 0) {
        continue;
      }

      // skip Variables to which an actual buffer is already assigned
      const VariableId vidx =
          info.variable_buffer_index_.find(buf_idx)->second[0];
      if (vidx_to_abidx.find(vidx) != vidx_to_abidx.end()) {
        continue;
      }

      // search for actual buffers vacant and large enough
      const SizeType needed_size =
          info.variable_buffer_size_.find(buf_idx)->second;
      BufferIndex abidx = 0;
      while (abidx != (BufferIndex)actual_buf_num) {
        const bool cond = !actual_assigned_flags[abidx] &&
                          needed_size <= actual_buf_sizes[abidx];
        if (cond) {
          actual_assigned_flags[abidx] = true;
          vidx_to_abidx[vidx] = abidx;
          break;
        } else {
          ++abidx;
        }
      }

      // increase size if buffers large enough was NOT found
      if (abidx == (BufferIndex)actual_buf_num) {
        for (abidx = 0; abidx < (BufferIndex)actual_buf_num; ++abidx) {
          if (!actual_assigned_flags[abidx]) {
            actual_buf_sizes[abidx] = needed_size;
            actual_assigned_flags[abidx] = true;
            vidx_to_abidx[vidx] = abidx;
            break;
          }
        }
      }
    }
  }
  return vidx_to_abidx;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/save_variable_buffer.py(187)
std::tuple<vector<SizeType>, BufferIdMap>
save_variable_buffer(const NnablartInfo &info) {
  // make the followings to save memory usage for Variable Buffer:
  //  - actual_buf_sizes(list): sizes of actual buffers, which lie under
  //  Variable Buffer.
  //                            indices in this list are hereinafter called
  //                            'actual buffer index'
  //  - vidx_to_abidx(dict): assignment of actual buffers to Variable Buffer.
  //                         the key and the value are Variable index and actual
  //                         buffer index, respectively
  const vector<LifeSpan> buf_var_lives = make_buf_var_lives(info);
  vector<SizeType> actual_buf_sizes =
      compute_actual_buf_sizes(info, buf_var_lives);
  vector<vector<BufferIndex>> buf_var_refs =
      make_buf_var_refs(info, buf_var_lives);
  BufferIdMap vidx_to_abidx =
      assign_actual_buf_to_variable(info, actual_buf_sizes, buf_var_refs);

  return std::make_tuple(actual_buf_sizes, vidx_to_abidx);
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(76)
inline int compute_int_bit_num(const float *const param_array,
                               const size_t size) {
  int int_bit_num = 1; //  1 is needed to represent sign

  vector<float> abs_array(size);
  std::transform(param_array, param_array + size, abs_array.begin(),
                 [](const float &a) { return std::abs(a); });
  const auto max_it = std::max_element(abs_array.begin(), abs_array.end());
  const float max_abs = *max_it;
  if (max_abs >= 1.f) {
    const ptrdiff_t max_idx = (max_it - abs_array.begin());
    const float max_log2 = std::log2(max_abs);
    // sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(82)
    if ((std::trunc(max_log2) == max_log2) && param_array[max_idx] > 0.f) {
      // almost impossible
      int_bit_num = static_cast<int>(max_log2) + 2;
    } else {
      int_bit_num = static_cast<int>(std::ceil(max_log2)) + 1;
    }
  }

  return int_bit_num;
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(69)
struct NnbExporter::List {
  MemoryBlockSize size;        // Size of data.
  MemoryBlockIndex list_index; // Index of data.
};

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(145)
struct NnbExporter::Variable {
  uint32_t id;        // Identifier
  List shape;         // Shape
  NN_DATA_TYPE type;  // Type of param values
  uint32_t fp_pos;    // Floating point position.
  int32_t data_index; // Location of data. If negative, it means data
                      // buffer index. Otherwise it means location of data
                      // in memory.
};

// ----------------------------------------------------------------------
// NnbExporter
// ----------------------------------------------------------------------
// sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(61)
NnbExporter::NnbExporter(const nbla::Context &ctx, Nnp &nnp,
                         int batch_size /*=-1*/,
                         int nnb_version /*= NN_BINARY_FORMAT_VERSION*/,
                         int api_level /*= -1*/)
    : ctx_(ctx), info_(), api_level_(api_level), nnb_version_(nnb_version) {
  if (batch_size < 0) {
    printf("NNB: Batch size adjust to 1.\n");
    batch_size = 1;
  }
  initialize_nnablart_info(ctx, nnp, batch_size, info_);
  preprocess_for_exporter(ctx_, info_);
}

// sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(90)
void NnbExporter::execute(const string &nnb_output_filename,
                          const char *const settings_filename /*= nullptr*/,
                          const string &default_type /*= "FLOAT32"*/) {
  Settings settings;
  if (settings_filename && settings_filename[0] != '\0') {
    if (std::FILE *f = std::fopen(settings_filename, "r")) {
      settings = load_yaml_ordered(f);
      std::fclose(f);
    }
  }

  ApiLevelInfo api_level_info_(api_level_);
  MemoryBlockStorage memory_data;

  vector<Network::Variable> network_variables = info_.network_->get_variables();
  const int num_network_variables = network_variables.size();
  const int num_input_variables = info_.input_variables_.size();
  const int num_output_variables = info_.output_variables_.size();

  //////////////////////////////////////////////////////////////////////////
  // Version
  const int api_level = api_level_info_.get_current_level();

  //////////////////////////////////////////////////////////////////////////
  // Variables name index
  unordered_map<string, unsigned int> vindexes_by_name;
  for (unsigned int i = 0; i < num_network_variables; ++i) {
    vindexes_by_name[network_variables[i].name] = i;
  }

  //////////////////////////////////////////////////////////////////////////
  // Inputs
  List inputs = {};
  {
    vector<uint32_t> input_list;
    input_list.reserve(num_input_variables);
    for (unsigned int i = 0; i < num_input_variables; ++i) {
      const string &name = info_.input_variables_[i];
      input_list.push_back(vindexes_by_name[name]);
    }
    const MemoryBlockIndex index =
        memory_data.alloc(MemoryBlock::pack('I', input_list));
    inputs = {input_list.size(), index};
  }

  //////////////////////////////////////////////////////////////////////////
  // Outputs
  List outputs = {};
  {
    vector<uint32_t> output_list;
    output_list.reserve(num_output_variables);
    for (unsigned int i = 0; i < num_output_variables; ++i) {
      const string &name = info_.output_variables_[i];
      output_list.push_back(vindexes_by_name[name]);
    }
    const MemoryBlockIndex index =
        memory_data.alloc(MemoryBlock::pack('I', output_list));
    outputs = {output_list.size(), index};
  }

  //////////////////////////////////////////////////////////////////////////
  // revise buffer size by bytes instead of data item.
  if (nnb_version_ > NN_BINARY_FORMAT_VERSION) {
    revise_buffer_size(info_, settings);
  }

  //////////////////////////////////////////////////////////////////////////
  // make 2 data to save Variable Buffers in inference
  vector<SizeType> actual_buf_sizes;
  BufferIdMap vidx_to_abidx;
  std::tie(actual_buf_sizes, vidx_to_abidx) = save_variable_buffer(info_);

  //////////////////////////////////////////////////////////////////////////
  // Variable buffers
  List buffers = {};
  {
    const vector<SizeType> blist = actual_buf_sizes;
    const MemoryBlockIndex index =
        memory_data.alloc(MemoryBlock::pack('I', blist));
    buffers = {blist.size(), index};
  }

  //////////////////////////////////////////////////////////////////////////
  // Variables
  List variables = {};
  {
    vector<int32_t> vindexes;
    for (uint32_t n = 0; n < num_network_variables; ++n) {
      const Network::Variable &v = network_variables[n];
      Variable var = {};
      var.id = n;

      // set var.shape and store into NNB
      {
        Shape_t shape = v.shape;
        for (int j = 0; j < shape.size(); ++j) {
          if (shape[j] < 0) {
            shape[j] = info_.batch_size_;
          }
        }
        const MemoryBlockIndex index =
            memory_data.alloc(MemoryBlock::pack('I', shape));
        var.shape = {(uint32_t)shape.size(), index};
      }

      // parse a type option in YAML given via -settings
      if (settings.variables.find(v.name) == settings.variables.end()) {
        settings.variables[v.name] = default_type;
      }
      const std::string type_option = settings.variables[v.name];
      const size_t opt_split_pos = type_option.find('_');
      const string type_name = type_option.substr(0, opt_split_pos);
      int fp_pos = (opt_split_pos != string::npos)
                       ? atoi(type_option.substr(opt_split_pos + 1).c_str())
                       : -1;

      // set var.type, var.data_index, and var.fp_pos in this paragraph
      var.type = get_data_type_from_type_name(type_name);
      auto it = info_.generator_variables_.find(v.name);
      if (it != info_.generator_variables_.end()) {
        VariablePtr pvar = it->second;
        const float *data = pvar->get_data_pointer<float>(ctx_);
        const size_t size = pvar->size();
        const MemoryBlockIndex index =
            memory_data.alloc(MemoryBlock::pack('f', data, size));
        var.data_index = index;
      } else if (v.type == "Parameter") {
        // store parameter into NNB
        VariablePtr pvar = info_.parameters_[v.name];
        MemoryBlock memory_block;
        if (type_name == "FLOAT32") {
          const float *data = pvar->get_data_pointer<float>(ctx_);
          const size_t size = pvar->size();
          memory_block = MemoryBlock::pack('f', data, size);
        } else if (type_name == "SIGN") {
          NdArrayPtr data = NdArray::create(pvar->shape());
          Array *dst = data->cast(dtypes::UBYTE, ctx_, /*write_only=*/true);

          {
            const dtypes dtype = pvar->data()->array()->dtype();
            const Array *src = pvar->data()->get(dtype, ctx_);
            dst->copy_from(src);
          }

          const size_t size = data->size();
          NBLA_CHECK(size % 8 == 0, error_code::value,
                     "Variable has invalid size (%d) for 'SIGN' type - must be "
                     "divisible by 8",
                     size);

          // sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(183)
          {
            uint8_t *const buffer = dst->pointer<uint8_t>();
            std::transform(buffer, buffer + size, buffer,
                           [](const uint8_t &v) { return (v == 255) ? 0 : v; });
          }

          // sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(184)
          data->reshape(Shape_t{((int64_t)size / 8), 8});

          // sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(185)
          {
            const int64_t stride = 8;
            const int64_t size0 = data->shape()[0];
            uint8_t *const buffer = dst->pointer<uint8_t>();
            for (int64_t i = 0; i < size0; ++i) {
              uint8_t *const first = buffer + (i * stride);
              std::reverse(first, first + stride);
            }
          }
          // sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(189)
          {
            const int64_t stride = 8;
            const int64_t size0 = data->shape()[0];
            NdArrayPtr packed_bits_nd = NdArray::create(Shape_t{size0});
            Array *packed_bits_arr =
                packed_bits_nd->cast(dtypes::UBYTE, ctx_, /*write_only=*/true);
            uint8_t *packed_bits = packed_bits_arr->pointer<uint8_t>();
            uint8_t *const buffer = dst->pointer<uint8_t>();
            for (int64_t i = 0; i < size0; ++i) {
              uint8_t *const packed_src = buffer + (i * stride);
              packed_bits[i] = packbits(packed_src);
            }
            memory_block = MemoryBlock::pack('B', packed_bits, size0);
          }
        } else {
          // type_name == "FIXED16" or type_name == "FIXED8"
          const dtypes dtype =
              (type_name == "FIXED16") ? dtypes::SHORT : dtypes::BYTE;
          const Array *src =
              pvar->data()->get(pvar->data()->array()->dtype(), ctx_);
          const float *const pSrc = src->const_pointer<float>();
          const Size_t size = src->size();

          // if fp_pos is not specified, compute it looking at its distribution
          if (fp_pos == -1) {
            const int int_bit_num = compute_int_bit_num(pSrc, size);
            fp_pos = int((dtype == dtypes::SHORT) ? FP_POS_MAX_INT16
                                                  : FP_POS_MAX_INT8) +
                     1;
            fp_pos -= int_bit_num;
          }

          // convert float to fixed point values
          {
            vector<int> values(size, 0);
            const float scale = (float)(1 << fp_pos);
            std::transform(
                pSrc, pSrc + size, values.begin(),
                [scale](const float &v) { return std::lround(v * scale); });
            if (dtype == dtypes::SHORT) {
              std::transform(values.begin(), values.end(), values.begin(),
                             &clamp_value<int, (-0x7fff - 1), 0x7fff>);
            } else {
              std::transform(values.begin(), values.end(), values.begin(),
                             &clamp_value<int, -128, 127>);
            }
            const char fmt = (dtype == dtypes::SHORT) ? 'h' : 'b';
            memory_block = MemoryBlock::pack(fmt, values.data(), size);
          }
        }

        const MemoryBlockIndex index =
            memory_data.alloc(std::move(memory_block));
        var.data_index = index;
      } else if (v.type == "Buffer") {
        NBLA_CHECK(var.type != NN_DATA_TYPE_SIGN, error_code::value,
                   "Unsupport SIGN type for Buffer Variable.");

        // check fp_pos
        NBLA_CHECK(!(var.type != NN_DATA_TYPE_FLOAT && fp_pos == -1),
                   error_code::value,
                   "fp_pos must be specified for Buffer Variable");

        // sony/nnabla/python/src/nnabla/utils/converter/nnablart/nnb.py(217)
        // FIXME: remove the following workaround
        {
          if (vidx_to_abidx.find(n) != vidx_to_abidx.end()) {
            // n which is NOT in vidx_to_abidx can appear
            // since NnpExpander doesn't handle --nnp-expand-network correctly
            var.data_index = (vidx_to_abidx[n] + 1) * -1;
          } else {
            // this var doesn't make sense, but add  it
            // so that nn_network_t::variables::size is conserved
            var.data_index = -1;
          }
        }
      }
      // check fp_pos and set var.fp_pos
      if (var.type == NN_DATA_TYPE_INT16 || var.type == NN_DATA_TYPE_INT8) {
        const int fp_pos_max = (var.type == NN_DATA_TYPE_INT16)
                                   ? FP_POS_MAX_INT16
                                   : FP_POS_MAX_INT8;
        if (0 <= fp_pos || fp_pos <= fp_pos_max) {
          var.fp_pos = fp_pos;
        } else {
          NBLA_ERROR(error_code::value, "invalid fp_pos was given");
        }
      } else {
        var.fp_pos = 0;
      }

      MemoryBlock variable = MemoryBlock::pack(
          "IiIBi", var.id, var.shape.size, var.shape.list_index,
          ((var.fp_pos & 0xf) << 4 | (var.type & 0xf)), var.data_index);
      const MemoryBlockIndex index = memory_data.alloc(std::move(variable));
      vindexes.push_back(index);
    }

    {
      const MemoryBlockIndex index =
          memory_data.alloc(MemoryBlock::pack('I', vindexes));
      variables = {vindexes.size(), index};
    }
  } // Variables

  //////////////////////////////////////////////////////////////////////////
  // Functions
  List functions = {};
  {
    vector<int32_t> findexes;
    vector<Network::Function> network_functions =
        info_.network_->get_functions();
    const size_t num_network_functions = network_functions.size();
    for (size_t n = 0; n < num_network_functions; ++n) {
      const Network::Function &f = network_functions[n];
      {
        const vector<string> api_level_functions =
            api_level_info_.get_function_list();
        auto api_level_function = std::find(api_level_functions.begin(),
                                            api_level_functions.end(), f.type);
        NBLA_CHECK(api_level_function != api_level_functions.end(),
                   error_code::value,
                   "%s() is not supported in current API level(=%d).", f.type,
                   api_level_info_.get_current_level());
      }

      const FunctionId id = api_level_info_.get_function_id(f.type);
      MemoryBlock function_data = MemoryBlock::pack('H', &id, 1);

      // Default function implementation is 0(float)
      {
        auto it = settings.functions.find(f.name);
        if (it == settings.functions.end()) {
          std::map<string, int> &fsetting = settings.functions[f.name];
          fsetting["implement"] = 0;
        }
      }
      {
        const int implement = settings.functions[f.name]["implement"];
        function_data += MemoryBlock::pack('H', &implement, 1);
      }

      {
        vector<uint32_t> finputs(f.inputs.size(), 0);
        std::transform(f.inputs.begin(), f.inputs.end(), finputs.begin(),
                       [&vindexes_by_name](const string &i) {
                         return vindexes_by_name[i];
                       });

        const MemoryBlockIndex index =
            memory_data.alloc(MemoryBlock::pack('I', finputs));
        function_data += MemoryBlock::pack("iI", finputs.size(), index);
      }

      {
        vector<uint32_t> foutputs(f.outputs.size(), 0);
        std::transform(f.outputs.begin(), f.outputs.end(), foutputs.begin(),
                       [&vindexes_by_name](const string &o) {
                         return vindexes_by_name[o];
                       });

        const MemoryBlockIndex index =
            memory_data.alloc(MemoryBlock::pack('I', foutputs));
        function_data += MemoryBlock::pack("iI", foutputs.size(), index);
      }

      const string argcode = api_level_info_.get_argument_code(f.type);
      size_t argcode_pos = 0;
      const size_t num_arguments = f.arguments.size();
      if (num_arguments > 0) {
        for (size_t an = 0; an < num_arguments; ++an) {
          const Network::FunctionArg &arg = f.arguments[an];
          const string arg_type_id = type_to_pack_format(arg);

          // omit the parameter that is not supported
          // we only down - version by omitting the tail - appended parameters.
          if (argcode_pos >= argcode.size()) {
            printf("%s.%lld is omitted for lower API Level:%d\n",
                   f.type.c_str(), an, api_level_info_.get_current_level());
            continue;
          } else {
            // If argument type is changed, this function will be
            // unable to down-version.
            const size_t arg_type_id_size = arg_type_id.size();
            NBLA_CHECK(
                argcode.substr(argcode_pos, arg_type_id_size) == arg_type_id,
                error_code::value, "%s is not supported by API Level:%d.",
                api_level_info_.get_func_unique_name(f.type).c_str(),
                api_level_info_.get_current_level());
            argcode_pos += arg_type_id_size;
          }

          // N.B. using const char* here ensures the overloaded
          // form of 'pack' is used that aligns the data
          const char *const pack_format = arg_type_id.c_str();
          switch (arg.type_) {
          case Network::FunctionArg::BOOL:
            function_data += MemoryBlock::pack(pack_format, arg.bool_);
            break;
          case Network::FunctionArg::DOUBLE:
            function_data += MemoryBlock::pack(pack_format, arg.double_);
            break;
          case Network::FunctionArg::FLOAT:
            function_data += MemoryBlock::pack(pack_format, arg.float_);
            break;
          case Network::FunctionArg::INT64:
            function_data += MemoryBlock::pack(pack_format, arg.int64_t_);
            break;
          case Network::FunctionArg::ARRAY_FLOAT: {
            MemoryBlockSize size = arg.afloat_.size();
            const MemoryBlockIndex index =
                memory_data.alloc(MemoryBlock::pack('f', arg.afloat_));
            function_data += MemoryBlock::pack(pack_format, size, index);
            break;
          }
          case Network::FunctionArg::ARRAY_INT64: {
            MemoryBlockSize size = arg.aint64_t_.size();
            const MemoryBlockIndex index =
                memory_data.alloc(MemoryBlock::pack('i', arg.aint64_t_));
            function_data += MemoryBlock::pack(pack_format, size, index);
            break;
          }
          case Network::FunctionArg::SHAPE: {
            MemoryBlockSize size = arg.aint64_t_.size();
            const MemoryBlockIndex index =
                memory_data.alloc(MemoryBlock::pack('i', arg.aint64_t_));
            function_data += MemoryBlock::pack(pack_format, size, index);
            break;
          }
          case Network::FunctionArg::STRING:
            function_data +=
                MemoryBlock::pack(pack_format, std::get<int>(arg.choice_));
            break;
          }
        }
      } else {
        // Check if old version requires argument.
        // If it is true, down-version is not allowed.
        NBLA_CHECK(argcode.size() == 0, error_code::value,
                   "%s is not supported by API Level: %d",
                   api_level_info_.get_func_unique_name(f.type).c_str(),
                   api_level_info_.get_current_level());
      }

      {
        const MemoryBlockIndex index =
            memory_data.alloc(std::move(function_data));
        findexes.push_back(index);
      }
    }

    {
      const MemoryBlockIndex index =
          memory_data.alloc(MemoryBlock::pack('I', findexes));
      functions = {findexes.size(), index};
    }
  } // Functions

  const vector<MemoryOffset> memory_offsets = memory_data.memory_offsets();
  MemoryBlock network = MemoryBlock::pack("IIiIiIiIiIiIII",
                                           nnb_version_,
                                           api_level,
                                           buffers.size,
                                           buffers.list_index,
                                           variables.size,
                                           variables.list_index,
                                           functions.size,
                                           functions.list_index,
                                           inputs.size,
                                           inputs.list_index,
                                           outputs.size,
                                           outputs.list_index,
                                           memory_offsets.size(),
                                           memory_data.size());

  network +=
      MemoryBlock::pack('I', memory_offsets.data(), memory_offsets.size());

  if (std::FILE *f = std::fopen(nnb_output_filename.c_str(), "wb")) {
    size_t bytes_written = network.fwrite(f);
    bytes_written += memory_data.fwrite(f);
    std::fclose(f);
  }
}

} // namespace nnb
} // namespace utils
} // namespace nbla