"""automatic differentiation utilities for minitorch"""

from typing import Callable, List , Tuple, Any, Set
from dataclasses import dataclass
from typing import Optional, Sequence

def central_difference(f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6) -> float:
    """compute numerical derivative of f with respect to argument 1`arg`. Uses central difference: (f(x+h) - f(x-h)) / (2h)
    Args : 
        f : function to differentiate
        *vals : Input values to f 
        arg:which argument to differentiate with respect to (0 - indexed )
        epsilon : step size for numerical differentiation

    returns :
        Approximate derivative


    example:
        >>> def mul(x, y):return x * y
        >>> central_difference(mul, 3, 4, arg=0) # df/dx at (3,4) 4.0
        >>> central_difference(mul, 3,4, arg= 0) # df/dy at (3,4) 3.0
    """
    vals_list = list(vals)

    #Create vals with arg incremented by epsilon
    vals_plus = vals_list.copy()
    vals_plus[arg] = vals_plus[arg] + epsilon

    #create vals with arg decremented by epsilon
    vals_minus = vals_list.copy()
    vals_minus[arg] = vals_minus[arg] - epsilon 

    #Compute central defference
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)


@dataclass
class Variable:
    """
       A node in the computation graph.

    Attributes:
      history: Record of the operation that created this variable.
      derivative: Gradient accumulated during backpropagation.
      name: Optional name for debugging.
    """

    history: Optional["History"] = None
    derivative: Optional[float] = None
    name: Optional[str] = None

    def is_leaf(self) -> bool :
        """A leaf varible has no history(was not created by an operation). """
        return self.history is None
    
    def is_constant(self) -> bool:
        """A constant has no history and will not receive gradients."""
        return self.history is None
    
    def requires_grad_(self, requires_grad: bool = True) -> "Variable":
        "Set whether this variable should track gradients"
        if requires_grad:
            self.history = History()
        else:
            self.history = None
        return self

@dataclass
class History:
    """records the opeation that created a variable.

    Attributes:
    last_fn: The function class that created this varible 
    ctx: context object storing values needed for backward 
    inputs: The input variables to the operation

    """

    last_fn: Optional[type] = None
    ctx: Optional["Context"] = None
    inputs: Sequence["Variable"] = ()



def topological_sort(var: Variable) -> List[Variable]:
    """
    Return variables in topological order.
    Children always appear before parents.
    """

    #create empty order list and visited set
    order: List[Variable] = []

    #This prevents visiting the same variable twice.
    visited: Set[int] = set()


    #Nested DFS function
    def visit(var: Variable) -> None:

        var_id = id(var)

        if var_id in visited:
            return

        visited.add(var_id)

        if var.history is not None and var.history.inputs:
            for input_var in var.history.inputs:
                visit(input_var)

        order.append(var)

    visit(var)
    return order


def backpropagate(variable: Variable, deriv: float = 1.0) -> None:
    """Run backpropagation starting from variable.

       computes gadients for all leat variable in the computation graph .
       Args:
           variable:Output variable to differentiate
           deriv: Gradient of variable(default 1.0 for scalar output)
    """

    #Get variables in topological order
    topo_order  = topological_sort(variable)

    #process in REVERSE topological order (output first)
    reverse_order  = list(reversed(topo_order))

    #Initailize gradient of output
    variable.derivative = deriv

    for var in reverse_order:
        if var.is_leaf():
            # Leaf variables just accumulate gradients, nothing to propagate
            continue

        if var.derivative is None:
            # No gradient reached this node (disconnected)
            continue

        # Get the function that created this variable
        history = var.history
        if history is None or history.last_fn is None:
            continue

        # Call backward to get gradients for inputs
        backward_fn = history.last_fn.backward 
        ctx = history.ctx
        input_grads = backward_fn(ctx, var.derivative)

        # Accumulate gradients to input variables
        for input_var, grad in zip(history.inputs, input_grads):
            if grad is not None:
                input_var.accumulate_derivative(grad)
            
@dataclass
class Context:
    """
    Context class is used by `function` to store inforamtion during the forwad pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backword(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation"
        if self.no_grad:
            return 
        self.saved_values = values


    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

    