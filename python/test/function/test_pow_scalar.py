# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
import nnabla.functions as F
from nbla_test_utils import list_context

ctxs = list_context('PowScalar')


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, 1, 2])
def test_pow_scalar_forward_backward(seed, val, ctx, func_name):
    from nbla_test_utils import function_tester
    rng = np.random.RandomState(seed)
    inputs = [rng.rand(2, 3, 4).astype(np.float32) + 0.5]
    function_tester(rng, F.pow_scalar, lambda x, y: x ** y, inputs,
                    func_args=[val], atol_b=5e-2,
                    ctx=ctx, func_name=func_name)


@pytest.mark.parametrize("ctx, func_name", ctxs)
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("val", [0.5, 1, 2])
@pytest.mark.parametrize("inplace", [False, True])
def test_pow_scalar_double_backward(seed, val, ctx, func_name, inplace):
    from nbla_test_utils import backward_function_tester
    rng = np.random.RandomState(seed)
    inputs = [(rng.randint(5, size=(2, 3)).astype(np.float32) + 1.0) * 0.2]
    backward_function_tester(rng, F.pow_scalar,
                             inputs=inputs,
                             func_args=[val, inplace], func_kwargs={},
                             atol_accum=1e-2,
                             dstep=1e-3,
                             ctx=ctx)
