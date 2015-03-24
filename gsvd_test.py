#This example matches with the result of GSVD of Matlab
from numpy import *


def magic(n):
    A = zeros((n,n))
    for i in range(n):
        for j in range(n):
            row=i+1; col=j+1
            A[i,j] = n*((row + col - 1 + n // 2) % n) + ((row + 2*col - 2) % n) + 1
    return A

from gsvd import gsvd

##EXAMPLE 1
A = arange(1,16).reshape((5,3),order='f')
#B = asarray([[8,1,6],[3,5,7],[4,9,2]])
B = magic(3)

print
print "EXAMPLE1: TEST for U,V,X,C,S = gsvd(A,B)"
U,V,X,C,S = gsvd(A,B)
print "U:"
print U
print "V:"
print V
print "X:"
print X
print "C:"
print C
print "S:"
print S

print "Test for A = U*C*X^t and B = V*S*X^t"
print allclose(A, dot(U, dot(C, X.T)))
print allclose(B, dot(V, dot(S, X.T)))

print
print "EXAMPLE1: TEST for U,V,X,C,S = gsvd(B,A)"
U,V,X,C,S = gsvd(B,A)
print "U:"
print U
print "V:"
print V
print "X:"
print X
print "C:"
print C
print "S:"
print S

print "Test for B = U*C*X^t and A = V*S*X^t"
print allclose(B, dot(U, dot(C, X.T)))
print allclose(A, dot(V, dot(S, X.T)))

##EXAMPLE 2
A = arange(1,16).reshape((3,5),order='f')
B = magic(5)

print
print "EXAMPLE2: TEST for U,V,X,C,S = gsvd(A,B)"
U,V,X,C,S = gsvd(A,B)
print "U:"
print U
print "V:"
print V
print "X:"
print X
print "C:"
print C
print "S:"
print S

print "Test for A = U*C*X^t and B = V*S*X^t"
print allclose(A, dot(U, dot(C, X.T)))
print allclose(B, dot(V, dot(S, X.T)))
