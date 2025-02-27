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

#include <nbla/logger.hpp>
#include <nbla_utils/nnb.hpp>
#include <nbla_utils/nnp.hpp>

#include <string>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nnp_impl.hpp"

#include <archive.h>
#include <archive_entry.h>

namespace nbla {
namespace utils {
namespace nnp {

// ----------------------------------------------------------------------
// Network
// ----------------------------------------------------------------------
Network::Network(NetworkImpl *impl) : impl_(unique_ptr<NetworkImpl>(impl)) {}

void Network::replace_variable(const string &name, CgVariablePtr variable) {
  impl_->replace_variable(name, variable);
}

CgVariablePtr Network::get_variable(const string &name) {
  return impl_->get_variable(name);
}

vector<Network::Variable> Network::get_variables() {
  return impl_->get_variables();
}

vector<Network::Function> Network::get_functions() {
  return impl_->get_functions();
}

string Network::name() const { return impl_->name(); }

void Network::set_batch_size(int batch_size) {
  impl_->set_batch_size(batch_size);
}

int Network::batch_size() const { return impl_->batch_size(); }

// ----------------------------------------------------------------------
// Network::FunctionArg
// ----------------------------------------------------------------------

inline int get_index_of(const string &value,
                        const vector<string> &available_values) {
  auto it = std::find(available_values.begin(), available_values.end(), value);
  const int index = (it == available_values.end())
                        ? -1
                        : (int)(it - available_values.begin());
  return index;
}

Network::FunctionArg::FunctionArg(const bool value) : type_(BOOL) {
  bool_ = value;
}
Network::FunctionArg::FunctionArg(const double value) : type_(DOUBLE) {
  double_ = value;
}
Network::FunctionArg::FunctionArg(const float value) : type_(FLOAT) {
  float_ = value;
}
Network::FunctionArg::FunctionArg(const int64_t value) : type_(INT64) {
  int64_t_ = value;
}
Network::FunctionArg::FunctionArg(const ::Shape &value) : type_(SHAPE) {
  new (&aint64_t_) vector<int64_t>(value.dim().begin(), value.dim().end());
}
Network::FunctionArg::FunctionArg(const string &value,
                                  const vector<string> &available_values)
    : type_(STRING) {
  new (&choice_) Choice(value, get_index_of(value, available_values));
}
Network::FunctionArg::FunctionArg(const FunctionArg &rhs) : type_(rhs.type_) {
  switch (type_) {
  case nbla::utils::nnp::Network::FunctionArg::BOOL:
    bool_ = rhs.bool_;
    break;
  case nbla::utils::nnp::Network::FunctionArg::DOUBLE:
    double_ = rhs.double_;
    break;
  case nbla::utils::nnp::Network::FunctionArg::FLOAT:
    float_ = rhs.float_;
    break;
  case nbla::utils::nnp::Network::FunctionArg::INT64:
    int64_t_ = rhs.int64_t_;
    break;
  case nbla::utils::nnp::Network::FunctionArg::SHAPE:
    new (&aint64_t_) vector<int64_t>(rhs.aint64_t_);
    break;
  case nbla::utils::nnp::Network::FunctionArg::ARRAY_INT64:
    new (&aint64_t_) vector<int64_t>(rhs.aint64_t_);
    break;
  case nbla::utils::nnp::Network::FunctionArg::ARRAY_FLOAT:
    new (&afloat_) vector<float>(rhs.afloat_);
    break;
  case nbla::utils::nnp::Network::FunctionArg::STRING:
    new (&choice_) Choice(rhs.choice_);
    break;
  default:
    break;
  }
}

Network::FunctionArg::~FunctionArg() {

  switch (type_) {
  case SHAPE:
    aint64_t_.~vector();
    break;
  case ARRAY_INT64:
    aint64_t_.~vector();
    break;
  case ARRAY_FLOAT:
    afloat_.~vector();
    break;
  case STRING:
    choice_.~tuple();
    break;
  default:
    break;
  }
}

// ----------------------------------------------------------------------
// Executor
// ----------------------------------------------------------------------
Executor::Executor(ExecutorImpl *impl)
    : impl_(unique_ptr<ExecutorImpl>(impl)) {}
string Executor::name() const { return impl_->name(); }
string Executor::network_name() const { return impl_->network_name(); }
void Executor::set_batch_size(int batch_size) {
  impl_->set_batch_size(batch_size);
}
int Executor::batch_size() const { return impl_->batch_size(); }
vector<Executor::DataVariable> Executor::get_data_variables() {
  return impl_->get_data_variables();
}
vector<Executor::OutputVariable> Executor::get_output_variables() {
  return impl_->get_output_variables();
}
vector<Executor::GeneratorVariable> Executor::get_generator_variables() {
  return impl_->get_generator_variables();
}
vector<Executor::ParameterVariable> Executor::get_parameter_variables() {
  return impl_->get_parameter_variables();
}
shared_ptr<Network> Executor::get_network() { return impl_->get_network(); }
void Executor::execute() { impl_->execute(); }

// ----------------------------------------------------------------------
// Optimizer
// ----------------------------------------------------------------------
Optimizer::Optimizer(OptimizerImpl *impl)
    : impl_(unique_ptr<OptimizerImpl>(impl)) {}

string Optimizer::name() const { return impl_->name(); }
string Optimizer::network_name() const { return impl_->network_name(); }
const int Optimizer::update_interval() const {
  return impl_->update_interval();
}
shared_ptr<Network> Optimizer::get_network() { return impl_->get_network(); }
const float Optimizer::update(const int iter) { return impl_->update(iter); }

// ----------------------------------------------------------------------
// Monitor
// ----------------------------------------------------------------------
Monitor::Monitor(MonitorImpl *impl) : impl_(unique_ptr<MonitorImpl>(impl)) {}

string Monitor::name() const { return impl_->name(); }
string Monitor::network_name() const { return impl_->network_name(); }
shared_ptr<Network> Monitor::get_network() { return impl_->get_network(); }
const float Monitor::monitor_epoch() { return impl_->monitor_epoch(); }

// ----------------------------------------------------------------------
// TrainingConfig
// ----------------------------------------------------------------------
TrainingConfig::TrainingConfig(TrainingConfigImpl *impl)
    : impl_(unique_ptr<TrainingConfigImpl>(impl)) {}

const long long int TrainingConfig::max_epoch() const {
  return impl_->max_epoch();
}

const long long int TrainingConfig::iter_per_epoch() const {
  return impl_->iter_per_epoch();
}

const bool TrainingConfig::save_best() const { return impl_->save_best(); }

// ----------------------------------------------------------------------
// Nnp
// ----------------------------------------------------------------------
Nnp::Nnp(const nbla::Context &ctx) : impl_(NBLA_NEW_OBJECT(NnpImpl, ctx)) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

Nnp::~Nnp() {}

bool Nnp::add(const string &filename) {
  int ep = filename.find_last_of(".");
  string extname = filename.substr(ep, filename.size() - ep);

  if (extname == ".prototxt" || extname == ".nntxt") {
    return impl_->add_prototxt(filename);
  } else if (extname == ".protobuf") {
    return impl_->add_protobuf(filename);
  } else if (extname == ".h5") {
    std::ifstream file(filename.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
      return impl_->add_hdf5(buffer.data(), size);
    }
  } else if (extname == ".nnp") {
    struct archive *a = archive_read_new();
    assert(a);
    archive_read_support_format_zip(a);
    int r = ARCHIVE_OK;
    r = archive_read_open_filename(a, filename.c_str(), 10240);
    assert(r == ARCHIVE_OK);
    if (r != ARCHIVE_OK) {
      return false;
    }
    bool ret = impl_->add_archive(a);
    archive_read_free(a);
    return ret;
  } else {
    std::cerr << "Error: No available file." << std::endl;
    return false;
  }

  return false;
}

bool Nnp::add(char *buffer, unsigned int size) {
  struct archive *a = archive_read_new();
  assert(a);
  archive_read_support_format_zip(a);
  int r = ARCHIVE_OK;
  r = archive_read_open_memory(a, buffer, size);
  if (r != ARCHIVE_OK) {
    return false;
  }
  bool ret = impl_->add_archive(a);
  archive_read_free(a);
  return ret;
}

vector<string> Nnp::get_network_names() { return impl_->get_network_names(); }

shared_ptr<Network> Nnp::get_network(const string &name) {
  return impl_->get_network(name);
}

vector<string> Nnp::get_executor_names() { return impl_->get_executor_names(); }

shared_ptr<Executor> Nnp::get_executor(const string &name) {
  return impl_->get_executor(name);
}

vector<pair<string, VariablePtr>> Nnp::get_parameters() {
  return impl_->get_parameters();
}

bool Nnp::save_parameters(const string &filename) {
  return impl_->save_parameters(filename);
}

bool Nnp::export_network(const string &filename) {
  int ep = filename.find_last_of(".");
  string extname = filename.substr(ep, filename.size() - ep);

  bool success = false;
  if (extname == ".nnb") {
    nbla::utils::nnb::NnbExporter exporter(impl_->ctx_, *this);
    exporter.execute(filename);
    success = true;
  } else {
    std::cerr << "Error: '" << extname << "' is not a supported format." << std::endl;
  }
  return success;
}

vector<string> Nnp::get_optimizer_names() {
  return impl_->get_optimizer_names();
}

shared_ptr<Optimizer> Nnp::get_optimizer(const string &name) {
  return impl_->get_optimizer(name);
}

vector<string> Nnp::get_monitor_names() { return impl_->get_monitor_names(); }

shared_ptr<Monitor> Nnp::get_monitor(const string &name) {
  return impl_->get_monitor(name);
}

shared_ptr<TrainingConfig> Nnp::get_training_config() {
  return impl_->get_training_config();
}
}
}
}
