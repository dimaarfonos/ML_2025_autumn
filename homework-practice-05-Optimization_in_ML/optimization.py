import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
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
        Находит длину шага alpha.
        """
        if self._method == 'Constant':
            return self.c

        if previous_alpha is not None:
            alpha = previous_alpha
        else:
            alpha = self.alpha_0

        phi_0 = oracle.func_directional(x_k, d_k, 0)
        phi_prime_0 = oracle.grad_directional(x_k, d_k, 0)

        if self._method == 'Armijo':
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                
                alpha *= 0.5
                
                if alpha < 1e-10:
                    return alpha

        elif self._method == 'Wolfe':
            try:
                from scipy.optimize import line_search
                
                result = line_search(
                    lambda x: oracle.func(x),
                    lambda x: oracle.grad(x),
                    x_k,
                    d_k,
                    c1=self.c1,
                    c2=self.c2,
                    amax=50
                )
                
                alpha_wolfe = result[0]
                
                if alpha_wolfe is not None and alpha_wolfe > 0:
                    return alpha_wolfe
            
            except:
                pass
            
            # Fallback to Armijo
            alpha = self.alpha_0
            while True:
                phi_alpha = oracle.func_directional(x_k, d_k, alpha)
                
                if phi_alpha <= phi_0 + self.c1 * alpha * phi_prime_0:
                    return alpha
                
                alpha *= 0.5
                
                if alpha < 1e-10:
                    return alpha

        return alpha


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
    Метод градиентного спуска для оптимизации.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    grad_0 = oracle.grad(x_0)
    grad_0_norm_sq = np.dot(grad_0, grad_0)
    
    start_time = datetime.now()
    
    for iteration in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_k_norm_sq = np.dot(grad_k, grad_k)
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_k_norm_sq))
            if x_0.size <= 2:
                history['x'].append(np.copy(x_k))
        
        if grad_k_norm_sq <= tolerance * grad_0_norm_sq:
            if display:
                print(f'Итерация {iteration}: достигнут критерий останова')
            return x_k, 'success', history
        
        d_k = -grad_k
        
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        
        if alpha_k is None or np.isnan(alpha_k) or np.isinf(alpha_k):
            if display:
                print(f'Итерация {iteration}: вычислительная ошибка в line search')
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha_k * d_k
        
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            if display:
                print(f'Итерация {iteration}: вычислительная ошибка в обновлении x')
            return x_k, 'computational_error', history
        
        if display and (iteration + 1) % 100 == 0:
            print(f'Итерация {iteration + 1}: f(x) = {oracle.func(x_k):.6e}')
    
    # Final check
    grad_k = oracle.grad(x_k)
    grad_k_norm_sq = np.dot(grad_k, grad_k)
    
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_k_norm_sq))
        if x_0.size <= 2:
            history['x'].append(np.copy(x_k))
    
    if grad_k_norm_sq <= tolerance * grad_0_norm_sq:
        return x_k, 'success', history
    
    if display:
        print(f'Достигнуто максимальное число итераций: {max_iter}')
    
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Метод Ньютона для оптимизации.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    grad_0 = oracle.grad(x_0)
    grad_0_norm_sq = np.dot(grad_0, grad_0)
    
    start_time = datetime.now()
    
    for iteration in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_k_norm_sq = np.dot(grad_k, grad_k)
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_k_norm_sq))
            if x_0.size <= 2:
                history['x'].append(np.copy(x_k))
        
        if grad_k_norm_sq <= tolerance * grad_0_norm_sq:
            if display:
                print(f'Итерация {iteration}: достигнут критерий останова')
            return x_k, 'success', history
        
        hess_k = oracle.hess(x_k)
        
        try:
            if scipy.sparse.issparse(hess_k):
                hess_k_dense = hess_k.toarray()
            else:
                hess_k_dense = hess_k
            
            # Check condition number to detect near-singular matrices
            try:
                cond = np.linalg.cond(hess_k_dense)
                if cond > 1e12:  # Matrix is nearly singular
                    if display:
                        print(f'Итерация {iteration}: гессиан плохо обусловлен (cond={cond:.2e})')
                    return x_k, 'computational_error', history
            except:
                pass
            
            c, lower = scipy.linalg.cho_factor(hess_k_dense)
            d_k = scipy.linalg.cho_solve((c, lower), -grad_k)
        except LinAlgError:
            if display:
                print(f'Итерация {iteration}: гессиан не положительно определён')
            return x_k, 'computational_error', history
        except Exception as e:
            if display:
                print(f'Итерация {iteration}: ошибка при вычислении направления Ньютона')
            return x_k, 'computational_error', history
        
        if np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)):
            if display:
                print(f'Итерация {iteration}: вычислительная ошибка при решении системы')
            return x_k, 'computational_error', history
        
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        
        if alpha_k is None or np.isnan(alpha_k) or np.isinf(alpha_k):
            if display:
                print(f'Итерация {iteration}: вычислительная ошибка в line search')
            return x_k, 'computational_error', history
        
        x_k = x_k + alpha_k * d_k
        
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            if display:
                print(f'Итерация {iteration}: вычислительная ошибка в обновлении x')
            return x_k, 'computational_error', history
        
        if display and (iteration + 1) % 10 == 0:
            print(f'Итерация {iteration + 1}: f(x) = {oracle.func(x_k):.6e}')
    
    # Final check
    grad_k = oracle.grad(x_k)
    grad_k_norm_sq = np.dot(grad_k, grad_k)
    
    if trace:
        history['time'].append((datetime.now() - start_time).total_seconds())
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.sqrt(grad_k_norm_sq))
        if x_0.size <= 2:
            history['x'].append(np.copy(x_k))
    
    if grad_k_norm_sq <= tolerance * grad_0_norm_sq:
        return x_k, 'success', history
    
    if display:
        print(f'Достигнуто максимальное число итераций: {max_iter}')
    
    return x_k, 'iterations_exceeded', history