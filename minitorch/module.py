"""Module system for organizing neural network components"""

from typing import Dict, List, Tuple, Any, Sequence


class Parameter:
    """A Trainable parameter in a neural network"""

    #define constructor that takes any value and stores it as the parameter value. The value can be a scalar, list, tensor, etc.
    def __init__(self, value:any):
        self.value = value
    
    @property
    def shape(self):
        """Return parameter shape if available"""
        #if the value has a shape attribute, return it. Otherwise, return empty tuple for scalars and None
        if hasattr(self.value, "shape"):
            return self.value.shape
        return ()
    
    #if helps clearly show the parameter value when printed
    def __repr__(self):
        return f"Parameter({self.value})"
    
    #allow updating the parameter value after creation
    def update(self, value:Any) -> None:
        """Update the parameter value"""
        self.value = value  



class Module:
    """Base class for all neural network modules"""

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True
        
    def modules(self) -> Sequence["Module"]:
        """Return all sub-modules recursively (deepth-first)"""
        results = []
        for module in self.__modules__.values():
            results.append(module)
            results.extend(module.modules())
        return results
        
    def train(self):
        """Set training mode for this module and all sub-modules"""
        self.training = True
        for module in self.modules():
            module.training = True
        
    def eval(self):
        """Set evaluation mode for this module and all sub-modules"""
        self.training = False
        for module in self.modules():
            module.training = False
        
    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Return all parameters in this module and sub-modules as (name, parameter) tuples"""
        results = []

        #Add direct parameters
        for name, param in self.__parameters__.items():
            results.append((name, param))
            
        #Add parameters from sub-modules with prefixed names
        for module_name, module in self.__modules__.items():
            for sub_name, param in module.named_parameters():
                results.append((f"{module_name}.{sub_name}", param))
        return results
        
    def parameters(self) -> Sequence[Parameter]:
        """Return all parameters in this module and sub-modules as a list"""
        return [param for _, param in self.named_parameters()]
        
    def add_parameter(self, name:str, value:Any)-> Parameter:
        """Add a parameter to this module"""
        if isinstance(value, Parameter):
            param = value
        else:
            param = Parameter(value)

        self.__parameters__[name] = param
        return param
        
    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        raise AttributeError(f"{type(self).__name__} has no attribute {key}")
