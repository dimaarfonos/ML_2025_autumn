import numpy as np
from numpy.linalg import LinAlgError
import scipy
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for phi(alpha) = f(x_k + alpha d_k).
        """
        method = self._method
        # Helper closures
        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        dphi = lambda a: oracle.grad_directional(x_k, d_k, a)

        if method == 'Constant':
            return self.c

        if method == 'Wolfe':
            # Try strong Wolfe via SciPy; fall back to Armijo backtracking if None.
            try:
                from scipy.optimize.linesearch import scalar_search_wolfe2
                alpha, _, _ = scalar_search_wolfe2(phi, dphi, c1=self.c1, c2=self.c2)
            except Exception:
                alpha = None
            if alpha is None:
                # fall back to Armijo backtracking starting from alpha_0
                alpha0 = self.alpha_0
                c1 = self.c1
                alpha = alpha0
                phi0 = phi(0.0)
                dphi0 = dphi(0.0)
                while phi(alpha) > phi0 + c1 * alpha * dphi0:
                    alpha *= 0.5
                return alpha
            return alpha

        if method == 'Armijo':
            c1 = self.c1
            alpha = previous_alpha if (previous_alpha is not None) else self.alpha_0
            phi0 = phi(0.0)
            dphi0 = dphi(0.0)
            if alpha <= 0:
                alpha = self.alpha_0
            while phi(alpha) > phi0 + c1 * alpha * dphi0:
                alpha *= 0.5
            return alpha

        raise ValueError('Unknown method {}'.format(method))


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    # Initial values
    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g0_norm_sq = np.dot(g_k, g_k)

    if trace:
        history['time'].append(0)
        history['func'].append(f_k)
        history['grad_norm'].append(np.linalg.norm(g_k))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))

    # Check stopping at start
    if np.dot(g_k, g_k) <= tolerance * g0_norm_sq:
        return x_k, 'success', history

    previous_alpha = None
    for it in range(1, max_iter + 1):
        d_k = -g_k
        try:
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=previous_alpha)
        except Exception:
            return x_k, 'computational_error', history
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history
        x_k = x_k + alpha_k * d_k

        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)

        if trace:
            history['time'].append(it)
            history['func'].append(f_k)
            history['grad_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if not np.isfinite(f_k) or not np.isfinite(np.linalg.norm(g_k)):
            return x_k, 'computational_error', history

        if np.dot(g_k, g_k) <= tolerance * g0_norm_sq:
            return x_k, 'success', history

        previous_alpha = alpha_k

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    f_k = oracle.func(x_k)
    g_k = oracle.grad(x_k)
    g0_norm_sq = np.dot(g_k, g_k)

    if trace:
        history['time'].append(0)
        history['func'].append(f_k)
        history['grad_norm'].append(np.linalg.norm(g_k))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))

    if np.dot(g_k, g_k) <= tolerance * g0_norm_sq:
        return x_k, 'success', history

    for it in range(1, max_iter + 1):
        try:
            H = oracle.hess(x_k)
            c, lower = scipy.linalg.cho_factor(H, overwrite_a=False, check_finite=True)
            d_k = scipy.linalg.cho_solve((c, lower), -g_k)
        except Exception:
            return x_k, 'newton_direction_error', history

        try:
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        except Exception:
            return x_k, 'computational_error', history
        if alpha_k is None or not np.isfinite(alpha_k):
            return x_k, 'computational_error', history

        x_k = x_k + alpha_k * d_k

        f_k = oracle.func(x_k)
        g_k = oracle.grad(x_k)

        if trace:
            history['time'].append(it)
            history['func'].append(f_k)
            history['grad_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if not np.isfinite(f_k) or not np.isfinite(np.linalg.norm(g_k)):
            return x_k, 'computational_error', history

        if np.dot(g_k, g_k) <= tolerance * g0_norm_sq:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history
