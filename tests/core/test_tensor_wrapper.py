# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
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
#

import pytest
import torch

from kornia.core import Tensor
from kornia.core.tensor_wrapper import TensorWrapper, unwrap, wrap

from testing.base import BaseTester


class TestTensorWrapper(BaseTester):
    def test_smoke(self, device, dtype):
        data = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        tensor = wrap(data, TensorWrapper)
        assert isinstance(tensor, TensorWrapper)
        assert isinstance(tensor.data, Tensor)
        assert tensor.shape == (1, 2, 3, 4)
        assert tensor.device == device
        assert tensor.dtype == dtype
        self.assert_close(data, tensor.unwrap())

    def test_serialization(self, device, dtype, tmp_path):
        data = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        tensor: TensorWrapper = wrap(data, TensorWrapper)

        file_path = tmp_path / "tensor.pt"
        torch.save(tensor, file_path)
        file_path.is_file()

        loaded_tensor: TensorWrapper = torch.load(file_path, weights_only=False)
        assert isinstance(loaded_tensor, TensorWrapper)

        self.assert_close(loaded_tensor.unwrap(), tensor.unwrap())

    def test_wrap_list(self, device, dtype):
        data_list = [
            torch.rand(2, device=device, dtype=dtype),
            torch.rand(3, device=device, dtype=dtype),
            TensorWrapper(torch.rand(3, device=device, dtype=dtype)),
            1,
            0.5,
        ]
        tensor_list = wrap(data_list, TensorWrapper)
        assert isinstance(tensor_list, list)
        assert len(tensor_list) == 5

        tensor_list_data = unwrap(tensor_list)
        assert len(tensor_list_data) == 5

        self.assert_close(tensor_list_data[0], data_list[0])
        self.assert_close(tensor_list_data[1], data_list[1])
        self.assert_close(tensor_list_data[2], data_list[2])
        assert tensor_list_data[3] == data_list[3]
        assert tensor_list_data[4] == data_list[4]

        for i in range(len(tensor_list_data[:3])):
            self.assert_close(tensor_list[i].unwrap(), data_list[i])

    def test_accessors(self, device, dtype):
        data = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        x = wrap(data, TensorWrapper)
        self.assert_close(x[1], torch.tensor(1.0, device=device, dtype=dtype))
        y = x[0]
        self.assert_close(y, torch.tensor(0.0, device=device, dtype=dtype))
        x[1] = 0.0
        self.assert_close(x, torch.zeros_like(x))

    def test_unary_ops(self, device, dtype):
        data = torch.rand(2, device=device, dtype=dtype)
        x = TensorWrapper(data)
        x1 = TensorWrapper(data)
        x2 = TensorWrapper(data)

        self.assert_close(x.add(x), x + x)
        self.assert_close(x.add(1), 1 + x)
        self.assert_close(x.add(1), x + 1)
        self.assert_close(x.mul(x), x * x)
        self.assert_close(x.mul(1), 1 * x)
        self.assert_close(x.mul(1), x * 1)
        self.assert_close(x.sub(x), x - x)
        self.assert_close(x.sub(1), x - 1)
        self.assert_close(x.div(x), x / x)
        self.assert_close(x.true_divide(x), x / x)
        self.assert_close(x.floor_divide(x), x // x)
        self.assert_close(x1.ge(x2), x1 >= x2)
        self.assert_close(x1.gt(x2), x1 > x2)
        self.assert_close(x1.lt(x2), x1 < x2)
        self.assert_close(x1.le(x2), x1 <= x2)
        self.assert_close(x1.eq(x2), x1 == x2)
        self.assert_close(x1.ne(x2), x1 != x2)
        self.assert_close(-x, -data)

    def test_callable(self, device, dtype):
        data = torch.ones(2, device=device, dtype=dtype)
        x = TensorWrapper(data)
        y = (x * x).sum(-1, True)
        self.assert_close(y, torch.ones_like(y) * 2)

    @pytest.mark.skip(reason="not implemented yet")
    def test_cardinality(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_jit(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_exception(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_module(self, device, dtype):
        pass

    @pytest.mark.skip(reason="not implemented yet")
    def test_gradcheck(self, device):
        pass
