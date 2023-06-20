# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
# Copyright 2022 Sony Group Corporation.
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

from six import iteritems

import nnabla as nn
import nnabla.solvers as S
from nnabla.testing import assert_allclose
import numpy as np
from collections import OrderedDict
import os


class RefSolver(object):

    def __init__(self, weight_decay_rate=0.0):
        self.default_weight_decay_rate = np.asarray(
            weight_decay_rate, dtype=np.float32)
        self.weight_decay_rate = self.default_weight_decay_rate.copy()

    def set_parameters(self, params):
        if not hasattr(self, 'params'):
            self.params = OrderedDict()
        for key, param in iteritems(params):
            param = param.d.copy()
            if key in self.params:
                continue
            self.params[key] = param
            self._set_state_impl(key, param)

    def _set_state_impl(self, key, param):
        pass

    def weight_decay_is_fused(self):
        return False

    def update(self, grads):
        for key, grad in iteritems(grads):
            param = self.params[key]
            self._update_impl(key, param, grad)
        self.weight_decay_rate = self.default_weight_decay_rate.copy()

    def weight_decay(self, grads, decay_rate):
        decay_rate = np.asarray(decay_rate, dtype=np.float32)
        if self.weight_decay_is_fused():
            self.weight_decay_rate = decay_rate
            return
        if decay_rate == 0:
            return
        for key, grad in iteritems(grads):
            param = self.params[key]
            grad[...] = grad + decay_rate * param

    def clip_grad_by_norm(self, grads, clip_norm):
        clip_norm = np.asarray(clip_norm, dtype=np.float32)
        for key, grad in iteritems(grads):
            norm = np.sqrt(np.sum(grad ** 2))
            grad[...] = clip_norm * grad / max(clip_norm, norm)


class MixinWeightDecayFused(object):
    def weight_decay_is_fused(self):
        return True


def solver_tester(rng, solver, ref_solver, solver_args=[], solver_kwargs={},
                  num_itr=5, decay=1e-4, clip_norm=0.5, atol=1e-6, rtol=1e-5,
                  ctx=None, solver_name=None,
                  weight_decay_interval=2,
                  hook_solver_update=None):
    if ctx is None:
        ctx = nn.Context()

    if hook_solver_update is None:
        def hook_solver_update(itr, s, r): return None

    # Create params
    p1 = nn.Variable([2, 3, 4])
    p2 = nn.Variable([3, 4, 1, 2])
    p3 = nn.Variable([])

    params = OrderedDict([('zZzZ', p1), ('bbb', p2), ('asdfadfdasd', p3)])
    for p in params.values():
        p.d = np.asarray(rng.randn(*p.shape), dtype=np.float32)
        p.g = np.asarray(rng.randn(*p.shape), dtype=np.float32)

    with nn.context_scope(ctx):
        s = solver(*solver_args, **solver_kwargs)
    s.set_parameters(params)
    if solver_name is not None:
        assert s.name == solver_name

    ref_s = ref_solver(*solver_args, **solver_kwargs)
    ref_s.set_parameters(params)

    # Get params (unordered_map is used in C++, thus check in both directions)
    params_ = s.get_parameters()
    for k0, v0 in iteritems(ref_s.params):
        v1 = params_[k0]
        assert_allclose(v0, v1.d, atol=atol, rtol=rtol)
    for k1, v1 in iteritems(params_):
        v0 = ref_s.params[k1]
        assert_allclose(v0, v1.d, atol=atol, rtol=rtol)

    # Check weight decay.
    if not s.weight_decay_is_fused():
        grad_copy = OrderedDict([(k, p.g.copy())
                                 for k, p in iteritems(params)])
        s.weight_decay(decay)
        ref_s.weight_decay(grad_copy, decay)
        for p, ref_p in zip(params.values(), grad_copy.values()):
            assert_allclose(ref_p, p.g, atol=atol, rtol=rtol)

    # Check clip grad by norm.
    grad_copy = OrderedDict([(k, p.g.copy())
                             for k, p in iteritems(params)])
    s.clip_grad_by_norm(clip_norm)
    ref_s.clip_grad_by_norm(grad_copy, clip_norm)
    for p, ref_p in zip(params.values(), grad_copy.values()):
        assert np.allclose(ref_p, p.g, atol=atol, rtol=rtol)

    # Check solver update.
    for i in range(num_itr):
        grads = OrderedDict([(k, np.asarray(rng.randn(*p.shape), dtype=np.float32))
                             for k, p in iteritems(params)])
        for k, g in iteritems(grads):
            params[k].g = g
        decay_value = decay * \
            ((i % weight_decay_interval) / weight_decay_interval)  # Python3
        s.weight_decay(decay_value)
        ref_s.weight_decay(grads, decay_value)
        s.update()
        ref_s.update(grads)
        # update check
        hook_solver_update(i, s, ref_s)
        for (k, p), (ref_k, ref_p) in zip(params.items(), ref_s.params.items()):
            assert_allclose(ref_p, p.d, atol=atol, rtol=rtol,
                            err_msg=f'i={i}, p="{k}" decay_value={decay_value}')
        # iteration state incrementation check
        for state in s.get_states().values():
            assert state.t == (i + 1)

    # Check inf, nan, and inf/nan
    for v, method in zip([[np.inf], [np.nan], [np.inf, np.nan]],
                         [lambda s: s.check_inf_grad(),
                          lambda s: s.check_nan_grad(),
                          lambda s: s.check_inf_or_nan_grad()]):
        def set_value(p):
            p.g[...] = rng.choice(v + [-1, 0, 1],
                                  size=int(np.prod(p.shape)),
                                  replace=True).reshape(p.shape)
            if v[0] not in p.g:
                p.g.flat[rng.choice(np.arange(int(np.prod(p.shape))))] = v[0]
        for p in params.values():
            assert method(s) == False
            g = p.g.copy()
            set_value(p)
            assert method(s) == True
            p.g[...] = g

    # Rescale grad
    scale = 10.
    ref_grad = [p.g.copy() for p in params.values()]
    for p in params.values():
        p.g *= scale
    s.scale_grad(1. / scale)
    for ref, p in zip(ref_grad, params.values()):
        assert_allclose(ref, p.g, atol=1e-4)

    # Save/Load Test
    def test_save_load(s, name):
        # Save states
        import tempfile
        tmpdir = tempfile.mkdtemp("solver-test")
        tmpfile = os.path.join(tmpdir, name)
        states0 = s.get_states()
        s.save_states(tmpfile)
        # Load states
        with nn.context_scope(ctx):
            s1 = solver(*solver_args, **solver_kwargs)
            s1.set_parameters(params)
            s1.load_states(tmpfile)
        # Check save/load states
        states1 = s1.get_states()
        for k0, s0 in iteritems(states0):
            s1 = states1[k0]
            for sname, vx0 in iteritems(s0.pstate):
                vx1 = s1.pstate[sname]
                assert_allclose(vx0.d, vx1.d)
            assert s1.t == s0.t
    test_save_load(s, "states.h5")
    test_save_load(s, "states.protobuf")

    # Check if remove_state_impl work correctly.
    s.clear_parameters()
