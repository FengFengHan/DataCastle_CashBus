class A:
    f1 = 'f1'
    def __init__(self,val):
        self.f2 = val
a = A('f2')
print(a.f2)
print(a.f1)
a.f3 = 'feng'
print(a.f3)
b = a
b.f1 = b.f1 + 'fb'
print(b.f1)
