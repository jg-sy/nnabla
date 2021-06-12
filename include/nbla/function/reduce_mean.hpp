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

/** ReduceMean
 */
#ifndef __NBLA_FUNCTION_REDUCEMEAN_HPP__
#define __NBLA_FUNCTION_REDUCEMEAN_HPP__

#include <nbla/cpu.hpp>
#include <nbla/function.hpp>
#include <nbla/function_registry.hpp>

namespace nbla {

NBLA_REGISTER_FUNCTION_HEADER(ReduceMean);

/** Reduction with mean along all dimensions defined as
@f[
y = \frac{1}{N} \sum_{i,j,...} x_{i,j,...}.
@f]

Inputs:
- N-D array.
Outputs:
- Scalar (0-shape array).

where N is size of the input array.
@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
template <typename T> class ReduceMean : public BaseFunction<> {

public:
  ReduceMean(const Context &ctx) : BaseFunction(ctx) {}
  virtual ~ReduceMean() {}
  virtual shared_ptr<Function> copy() const { return create_ReduceMean(ctx_); }
  virtual vector<dtypes> in_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual int min_inputs() { return 1; }
  virtual int min_outputs() { return 1; }
  virtual string name() { return "ReduceMean"; }
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
  virtual bool grad_depends_input_data_impl(int i, int j) const {
    return false;
  }
};
}
#endif
