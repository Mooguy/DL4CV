import unittest  # noqa
import torch
from functional import multi_head_attention
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange

class TestAttentionFunction(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float64

  def _test_forward(self, batch_size, num_heads, sequence_size, num_dims):
    q = torch.rand(batch_size, num_heads, sequence_size, num_dims)
    k = torch.rand(batch_size, num_heads, sequence_size, num_dims)
    v = torch.rand(batch_size, num_heads, sequence_size, num_dims)
    y = multi_head_attention(q, k, v)

    y_ = scaled_dot_product_attention(q,k,v)
    y_ = rearrange(y_, "b h s d -> b s (h d)")

    dbg = (f'\ngot: {y}\nexpected: {y_}')
    torch.testing.assert_close(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backwards(self, batch_size, num_heads, sequence_size, num_dims):
    q = torch.rand(batch_size, num_heads, sequence_size, num_dims, requires_grad=True)
    k = torch.rand(batch_size, num_heads, sequence_size, num_dims, requires_grad=True)
    v = torch.rand(batch_size, num_heads, sequence_size, num_dims, requires_grad=True)

    y = multi_head_attention(q, k, v)
    y.sum().backward()

    q_ = q.clone().detach()
    q_.requires_grad = True
    k_ = k.clone().detach()
    k_.requires_grad = True
    v_ = v.clone().detach()
    v_.requires_grad = True
    y_ = scaled_dot_product_attention(q_,k_,v_)
    y_ = rearrange(y_, "b h s d -> b s (h d)")
    y_.sum().backward()

    dbg = f'got: {q.grad}\nexpected: {q_.grad}'
    torch.testing.assert_close(q.grad, q_.grad, rtol=self.rtol, atol=self.atol, msg=dbg)
    dbg = f'got: {k.grad}\nexpected: {k_.grad}'
    torch.testing.assert_close(k.grad, k_.grad, rtol=self.rtol, atol=self.atol, msg=dbg)
    dbg = f'got: {v.grad}\nexpected: {v_.grad}'
    torch.testing.assert_close(v.grad, v_.grad, rtol=self.rtol, atol=self.atol, msg=dbg)

  def testSingleHead(self):
    batch_size = 2
    num_heads = 1
    sequence_size = 8
    num_dims = 16
    self._test_forward(batch_size, num_heads, sequence_size, num_dims)

  def testMultiHead(self):
    batch_size = 2
    num_heads = 4 
    sequence_size = 8
    num_dims = 16
    self._test_forward(batch_size, num_heads, sequence_size, num_dims)

  def testSingleHeadBackward(self):
    batch_size = 2
    num_heads = 1
    sequence_size = 8
    num_dims = 16
    self._test_backwards(batch_size, num_heads, sequence_size, num_dims)

  def testMultiHeadBackward(self):
    batch_size = 2
    num_heads = 4 
    sequence_size = 8
    num_dims = 16
    self._test_backwards(batch_size, num_heads, sequence_size, num_dims)



    

