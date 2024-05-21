import numpy as np

import minitorch.operators
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def out_position_to_in_position(out_position, out_shape, out_strides, in_shape, in_strides):
    """
    Convert an out position to an in position.

    Args:
        out_position (int): position in out tensor
        out_shape (np.array): shape for out tensor
        out_strides (np.array): strides for out tensor
        in_shape (np.array): shape for in tensor
        in_strides (np.array): strides for in tensor

    Returns:
        int : position at in tensor
    """
    out_index = np.zeros(out_shape.shape, dtype=int)
    to_index(out_position, out_shape, out_index)
    in_index = np.zeros(in_shape.shape, dtype=int)
    broadcast_index(out_index, out_shape, in_shape, in_index)
    in_position = index_to_position(in_index, in_strides)
    assert isinstance(in_position, int), f"in_position is not an int: {in_shape, in_index, in_position}"
    return in_position


def tensor_map(fn):
    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        """
        Low-level implementation of tensor map between
        tensors with *possibly different strides*.

        Simple version:

        * Fill in the `out` array by applying `fn` to each
          value of `in_storage` assuming `out_shape` and `in_shape`
          are the same size.

        Broadcasted version:

        * Fill in the `out` array by applying `fn` to each
          value of `in_storage` assuming `out_shape` and `in_shape`
          broadcast. (`in_shape` must be smaller than `out_shape`).

        Args:
            fn: function from float-to-float to apply
            out (np.array): storage for out tensor
            out_shape (np.array): shape for out tensor
            out_strides (np.array): strides for out tensor
            in_storage (np.array): storage for in tensor
            in_shape (np.array): shape for in tensor
            in_strides (np.array): strides for in tensor

        Returns:
            None : Fills in `out`
        """
        mapped_in_storage = np.array([fn(x) for x in in_storage])
        for out_position in range(len(out)):
            in_position = out_position_to_in_position(out_position, out_shape, out_strides, in_shape, in_strides)
            out[out_position] = mapped_in_storage[in_position]
    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      fn_map(a, out)
      out

    Simple version::

        for i:
            for j:
                out[i, j] = fn(a[i, j])

    Broadcasted version (`a` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0])

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        """
        Low-level implementation of tensor zip between
        tensors with *possibly different strides*.

        Simple version:

        * Fill in the `out` array by applying `fn` to each
          value of `a_storage` and `b_storage` assuming `out_shape`
          and `a_shape` are the same size.

        Broadcasted version:

        * Fill in the `out` array by applying `fn` to each
          value of `a_storage` and `b_storage` assuming `a_shape`
          and `b_shape` broadcast to `out_shape`.

        Args:
            fn: function mapping two floats to float to apply
            out (np.array): storage for `out` tensor
            out_shape (np.array): shape for `out` tensor
            out_strides (np.array): strides for `out` tensor
            a_storage (np.array): storage for `a` tensor
            a_shape (np.array): shape for `a` tensor
            a_strides (np.array): strides for `a` tensor
            b_storage (np.array): storage for `b` tensor
            b_shape (np.array): shape for `b` tensor
            b_strides (np.array): strides for `b` tensor

        Returns:
            None : Fills in `out`
        """
        for out_position in range(len(out)):
            a_position = out_position_to_in_position(
                out_position, out_shape, out_strides, a_shape, a_strides
            )
            b_position = out_position_to_in_position(
                out_position, out_shape, out_strides, b_shape, b_strides
            )
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      out = fn_zip(a, b)

    Simple version ::

        for i:
            for j:
                out[i, j] = fn(a[i, j], b[i, j])

    Broadcasted version (`a` and `b` might be smaller than `out`) ::

        for i:
            for j:
                out[i, j] = fn(a[i, 0], b[0, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    def _reduce(out, out_shape, out_strides, a_storage, a_shape, a_strides, reduce_dim):
        """
        Low-level implementation of tensor reduce.

        * `out_shape` will be the same as `a_shape`
           except with `reduce_dim` turned to size `1`

        Args:
            fn: reduction function mapping two floats to float
            out (np.array): storage for `out` tensor
            out_shape (np.array): shape for `out` tensor
            out_strides (np.array): strides for `out` tensor
            a_storage (np.array): storage for `a` tensor
            a_shape (np.array): shape for `a` tensor
            a_strides (np.array): strides for `a` tensor
            reduce_dim (int): dimension to reduce out

        Returns:
            None : Fills in `out`
        """
        if reduce_dim is None:
            return minitorch.operators.reduce(fn, out[0])(a_storage)
        for out_position in range(len(out)):
            out_index = np.zeros(out_shape.shape, dtype=np.int64)
            to_index(out_position, out_shape, out_index)
            for i in range(a_shape[reduce_dim]):
                a_index = out_index
                a_index[reduce_dim] = i
                a_position = index_to_position(a_index, a_strides)
                out[out_position] = fn(out[out_position], a_storage[a_position])

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      out = fn_reduce(a, dim)

    Simple version ::

        for j:
            out[1, j] = start
            for i:
                out[1, j] = fn(out[1, j], a[i, j])


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dim (int): int of dim to reduce

    Returns:
        :class:`TensorData` : new tensor
    """

    f = tensor_reduce(fn)

    def ret(a, dim):
        out_shape = list(a.shape)
        out_shape[dim] = 1

        # Other values when not sum.
        out = a.zeros(tuple(out_shape))
        out._tensor._storage[:] = start

        f(*out.tuple(), *a.tuple(), dim)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
