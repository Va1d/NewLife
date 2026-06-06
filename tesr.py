class A:
    data = []
class B(A):
    pass 
class C(A):
    pass 

B.data.append(1)
print(A.data)  # Output: [1]
print(B.data)  # Output: [1]
print(C.data)  # Output: [1]