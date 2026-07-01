from minitorch.scalar import Scalar

# def square(x):
#     return x * x

# def mul(x, y):
#     return x * y

# val = central_difference(square, 3.0)
# print(val)

# val2 = central_difference(mul , 3.0, 4.0 ,arg=0)

# print(val2)
# print(central_difference(mul, 3.0, 4.0, arg=1))  # ∂(xy)/∂y at (3,4)



# a = Scalar(2.0, name="a")
# b = Scalar(3.0, name="b")


# print(f"a.data = {a.data}")
# print(f"a.is_leaf() = {a.is_leaf()}")
# print(f"a.derivative = {a.derivative}")

# a.accumulate_derivative(1.0)
# a.accumulate_derivative(2.0)
# print(f"After accumulating: a.derivative = {a.derivative}")

# print(a.requires_grad_())
# print(a.is_leaf())

# a.zero_grad_()
# print(a.derivative)



a = Scalar(2.0)
a.requires_grad_(True)
b = Scalar(3.0)
b.requires_grad_(True)

c = a * b  # c = 6.0

print(f"c.data = {c.data}")
print(f"c.is_leaf() = {c.is_leaf()}")
print(f"c.history.last_fn = {c.history.last_fn}")