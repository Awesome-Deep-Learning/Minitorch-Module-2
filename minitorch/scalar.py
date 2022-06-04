from .autodiff import FunctionBase, Variable, History
from . import operators
import numpy as np


# ## Task 1.1
# Central Difference calculation


def central_difference(f, *vals, arg=0, epsilon=1e-6):
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`
        arg (int): the number :math:`i` of the arg to compute the derivative
        epsilon (float): a small constant

    Returns:
        float : An approximation of :math:`f'_i(x_0, \ldots, x_{n-1})`
    """
    minus_epsilon_vals = [val - epsilon if i == arg else val for (i, val) in enumerate(list(vals))]
    plus_epsilon_vals = [val + epsilon if i == arg else val for (i, val) in enumerate(list(vals))]
    derivative = (f(*plus_epsilon_vals) - f(*minus_epsilon_vals)) / (2 * epsilon)
    return derivative

# ## Task 1.2 and 1.4
# Scalar Forward and Backward


class Scalar(Variable):
    """
    A reimplementation of scalar values for autodifferentiation
    tracking.  Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    :class:`ScalarFunction`.

    Attributes:
        data (float): The wrapped scalar value.
    """

    def __init__(self, v, back=History(), name=None):
        super().__init__(back, name=name)
        self.data = float(v)

    def __repr__(self):
        return "Scalar(%f)" % self.data

    def __mul__(self, b):
        return Mul.apply(self, b)

    def __truediv__(self, b):
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b):
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b):
        return Add.apply(self, b)

    def __bool__(self):
        return bool(self.data)

    def __lt__(self, b):
        return LT.apply(self, b)

    def __gt__(self, b):
        return LT.apply(b, self)

    def __eq__(self, b):
        return EQ.apply(self, b)

    def __sub__(self, b):
        return Add.apply(self, Neg.apply(b))

    def __neg__(self):
        return Neg.apply(self)

    def log(self):
        return Log.apply(self)

    def exp(self):
        return Exp.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def get_data(self):
        """Returns the raw float value"""
        return self.data


class ScalarFunction(FunctionBase):
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    def forward(ctx, *inputs):
        r"""
        Forward call, compute :math:`f(x_0 \ldots x_{n-1})`.

        Args:
            ctx (:class:`Context`): A container object to save
                                    any information that may be needed
                                    for the call to backward.
            *inputs (list of floats): n-float values :math:`x_0 \ldots x_{n-1}`.

        Should return float the computation of the function :math:`f`.
        """
        pass  # pragma: no cover

    @staticmethod
    def backward(ctx, d_out):
        r"""
        Backward call, computes :math:`f'_{x_i}(x_0 \ldots x_{n-1}) \times d_{out}`.

        Args:
            ctx (Context): A container object holding any information saved during in the corresponding `forward` call.
            d_out (float): :math:`d_out` term in the chain rule.

        Should return the computation of the derivative function
        :math:`f'_{x_i}` for each input :math:`x_i` times `d_out`.

        """
        pass  # pragma: no cover

    # Checks.
    variable = Scalar
    data_type = float

    @staticmethod
    def data(a):
        return a


# Examples
class Add(ScalarFunction):
    """Addition function :math:`f(x, y) = x + y`"""

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, d_output):
        return d_output, d_output


class Log(ScalarFunction):
    """Log function :math:`f(x) = log(x)`"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.

class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        # super usage ref: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        return super(Mul, Mul).data_type(operators.mul(a, b))

    @staticmethod
    def backward(ctx, d_output):
        a, b = ctx.saved_values
        return operators.mul_back(a, b, d_output)


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return super(Inv, Inv).data_type(operators.inv(a))

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx, a):
        return super(Neg, Neg).data_type(operators.neg(a))

    @staticmethod
    def backward(ctx, d_output):
        return operators.neg_back(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return super(Sigmoid, Sigmoid).data_type(operators.sigmoid(a))

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.sigmoid_back(a, d_output)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return super(ReLU, ReLU).data_type(operators.relu(a))

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return super(Exp, Exp).data_type(operators.exp(a))

    @staticmethod
    def backward(ctx, d_output):
        a = ctx.saved_values
        return operators.exp_back(a, d_output)


class LT(ScalarFunction):
    """Less-than function :math:`f(x) =` 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx, a, b):
        return super(LT, LT).data_type(operators.lt(a, b))

    @staticmethod
    def backward(ctx, d_output):
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function :math:`f(x) =` 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx, a, b):
        return super(EQ, EQ).data_type(operators.eq(a, b))

    @staticmethod
    def backward(ctx, d_output):
        return 0.0, 0.0


def derivative_check(f, *scalars):
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f (function) : function from n-scalars to 1-scalar.
        *scalars (list of :class:`Scalar`) : n input scalar values.
    """
    for x in scalars:
        x.requires_grad_(True)
    out = f(*scalars)
    out.backward()

    vals = [v for v in scalars]
    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *vals, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
