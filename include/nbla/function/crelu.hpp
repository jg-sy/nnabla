// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

/** cReLU
 */
#ifndef __NBLA_FUNCTION_CRELU_HPP__
#define __NBLA_FUNCTION_CRELU_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

#include <memory>
#include <string>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(CReLU, int);

/** Concatenated Rectified Linear Unit (CReLU) concatenates ReLU outputs of
positive inputs and negative inputs together at specified axis.

Inputs:
- N-D array.

Outputs:
- N-D array where axis dimension is doubled by concatenating.

@tparam T Data type for computation.
@param axis The ReLU activations of positive inputs and negative inputs are
concatenated at axis.

@sa Clevert et al. Fast and Accurate Deep Network Learning by Exponential Linear
Units (ELUs). https://arxiv.org/abs/1511.07289

\ingroup FunctionImplGrp
 */
template <typename T> class CReLU : public BaseFunction<int> {
protected:
  int axis_;
  Size_t size0_, size1_;

public:
  CReLU(const Context &ctx, int axis) : BaseFunction(ctx, axis), axis_(axis) {}
  virtual ~CReLU() {}
  virtual shared_ptr<Function> copy() const {
    return create_CReLU(ctx_, axis_);
  }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual string name() { return "CReLU"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cpu>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return false; }

protected:
  NBLA_API virtual void setup_impl(const Variables &inputs,
                                   const Variables &outputs);
  NBLA_API virtual void forward_impl(const Variables &inputs,
                                     const Variables &outputs);
  NBLA_API virtual void backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum);
  virtual bool grad_depends_input_data_impl(int i, int j) const { return true; }
};
}
#endif
