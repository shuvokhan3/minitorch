"""Module system for organizing neural network components"""

from typing import Dict, List, Tuple, Any, Sequence


class Parameter:
    """A Trainable parameter in a neural network"""
    def __init__(self,value:any):
        self.value - value
    
    @property
    def shape(self):
        """Return parameter shape if available"""
        if hasattr(self.value, "shape"):
            return self.value.shape
        return ()
    
    def __repr__(self):
        return f"Parameter({self.value})"
    
    def update(self, value:Any) -> None:
        """Update the parameter value"""
        self.value = value  

    class Module:
        """Base class for all neural network modules"""

        def __init__(self):
            self.__module__ = {}
            self.__parameters__ = {}
            self.training = True
        
        def modules(self) -> Sequence["Module"]:
            """Return all sub-modules recursively (deepth-first)"""
            results = []
            for module in self.__modules.values():
                results.append(module)
                results.extend(module.modules())
            return results
        
        def train(self):
            """Set training mode for this module and all sub-modules"""
            self.training = True
            for module in self.modules():
                module.training = True
        
        def evel(self):
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
        
        def __setattr__(self, key: str, value: Any):
            if isinstance(value, Parameter):
               self._parameters[key] = value
            elif isinstance(value, Module):
               self._modules[key] = value

            super().__setattr__(key, value)
