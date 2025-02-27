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

/** minimum2
 */
#ifndef __NBLA_FUNCTION_MINIMUM2_HPP__
#define __NBLA_FUNCTION_MINIMUM2_HPP__

#include <nbla/function/utils/base_transform_binary.hpp>

namespace nbla {

/** @class Minimum2
@brief Elementwise minimum defined as
@f[
y_i = {\rm minimum}\left(x^{(0)}_i, x^{(1)}_i\right).
@f]

Inputs:
- N-D array.
- N-D array.

Outputs:
- N-D array.

@tparam T Data type for computation.
\ingroup FunctionImplGrp
 */
NBLA_DEFINE_TRANSFORM_BINARY(Minimum2, (x0 < x1) ? x0 : x1, (x0 < x1) * dy,
                             (x0 >= x1) * dy, false, false, true, true);
}
#endif
