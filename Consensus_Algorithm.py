
"""
CONSENSUS ALGORITHM FOR SOLVING LINEAR ALGEBRAIC EQUATION Ax=b
 
"""

import numpy as np
import matplotlib.pyplot as plt

"""
creating random matrix A and random vector b
for the system Ax=b
"""
A=np.random.rand(1000,1000)
b=np.random.rand(1000,1)

#dividing A into 250x4 submatrices
A_split=np.split(A,4)

#similar partition to the vector b
b_split=np.split(b,4)

#the submatrices that the agents have to operate on
#Submatrix Ai for agent i
A1=A_split[0]
A2=A_split[1]
A3=A_split[2]
A4=A_split[3]

#the corresponding bi's for agent i
b1=b_split[0]
b2=b_split[1]
b3=b_split[2]
b4=b_split[3]

"""
Using SVD to find the Basis for the NullSpace
Colums r+1 to n;the last n-r columns of V form the basis for the NullSpace

"""
U1,S1,Vh1=np.linalg.svd(A1,full_matrices="True")
V1=np.transpose(Vh1)
#Basis for the NullSpace of A1
N1=V1[:,250:]

U2,S2,Vh2=np.linalg.svd(A2,full_matrices="True")
V2=np.transpose(Vh2)
#Basis for the NullSpace of A2
N2=V2[:,250:]

U3,S3,Vh3=np.linalg.svd(A3,full_matrices="True")
V3=np.transpose(Vh3)
#Basis for the NullSpace of A3
N3=V3[:,250:]

U4,S4,Vh4=np.linalg.svd(A4,full_matrices="True")
V4=np.transpose(Vh4)
#Basis for the NullSpace of A4
N4=V4[:,250:]

"""
# Finding the Projection matrices onto the NullSpace
# For matrix N with columns as the Basis Vectors for the NullSpace
# Projection matrix P=N(inv(N'N))N'

"""
#P1:Projection Matrix onto N(A1)
N1t=np.transpose(N1)
T1=np.matmul(N1t,N1)
T1_inv=np.linalg.inv(T1)
Inter_1=np.matmul(N1,T1_inv)
P1=np.matmul(Inter_1,N1t)
#P2:Projection Matrix onto N(A2)
N2t=np.transpose(N2)
T2=np.matmul(N2t,N2)
T2_inv=np.linalg.inv(T2)
Inter_2=np.matmul(N2,T2_inv)
P2=np.matmul(Inter_2,N2t)
#P3:Projection Matrix onto N(A3)
N3t=np.transpose(N3)
T3=np.matmul(N3t,N3)
T3_inv=np.linalg.inv(T3)
Inter_3=np.matmul(N3,T3_inv)
P3=np.matmul(Inter_3,N3t)
#P4:Projection Matrix onto N(A4)
N4t=np.transpose(N4)
T4=np.matmul(N4t,N4)
T4_inv=np.linalg.inv(T4)
Inter_4=np.matmul(N4,T4_inv)
P4=np.matmul(Inter_4,N4t)

"""
 Algorithm:
 1)Initialize: xi=min-norm solution to Ai x = bi
 2)Repeat:Until convergence
       #Use the update equation specified
       #Determine the 2-norm of (Axi-b) during each iteration 
       (To check how close the estimate xi is to the true solution to Ax=b)
"""   

"""
Finding the min-norm solution to Ai x =bi
x_init= Ai' inv((Ai Ai')) bi

"""
#Transpose each Ai
A1t=np.transpose(A1)
A2t=np.transpose(A2)
A3t=np.transpose(A3)
A4t=np.transpose(A4)

#Finding each inv(AiAi')
M1=np.linalg.inv(np.matmul(A1,A1t))
M2=np.linalg.inv(np.matmul(A2,A2t))
M3=np.linalg.inv(np.matmul(A3,A3t))
M4=np.linalg.inv(np.matmul(A4,A4t))

#The min-norm solutions of A1,A2,A3 and A4  
x1_init=np.matmul(np.matmul(A1t,M1),b1)
x2_init=np.matmul(np.matmul(A2t,M2),b2)
x3_init=np.matmul(np.matmul(A3t,M3),b3)
x4_init=np.matmul(np.matmul(A4t,M4),b4)

x1_array=np.hsplit(np.zeros((1000,500)),500)
x2_array=np.hsplit(np.zeros((1000,500)),500)
x3_array=np.hsplit(np.zeros((1000,500)),500)
x4_array=np.hsplit(np.zeros((1000,500)),500)
e1=np.zeros((1,500))
e2=np.zeros((1,500))
e3=np.zeros((1,500))
e4=np.zeros((1,500))
count_array=np.zeros((1,500))
for index in range(500):
    count_array[:,index]=index+1
i=4
ir=0.25
e1[:,0]=np.linalg.norm(np.subtract(np.matmul(A,x1_init),b))
e2[:,0]=np.linalg.norm(np.subtract(np.matmul(A,x1_init),b))
e3[:,0]=np.linalg.norm(np.subtract(np.matmul(A,x1_init),b))
e4[:,0]=np.linalg.norm(np.subtract(np.matmul(A,x1_init),b))
count=0
"""
Update equation:
    xi[count]=xi[count-1]-(1/N)*P{N*xi[count-1]-sum_over_j(xj[count-1])}
    N:No. of neighbours
"""
while(count<500):
 
 if count==0:
     #initialize to the min-norm solution
       
        x1_array[count]=x1_init
        x2_array[count]=x2_init
        x3_array[count]=x3_init
        x4_array[count]=x4_init
        
 else:  
     #N*xi[count-1]
        f1=np.multiply(i,x1_array[count-1])
        f2=np.multiply(i,x2_array[count-1])
        f3=np.multiply(i,x3_array[count-1])
        f4=np.multiply(i,x4_array[count-1])
     #sum_over_j(xj[count-1])
        s=np.add(x1_array[count-1],x2_array[count-1])
        s1=np.add(s,x3_array[count-1])
        s11=np.add(s1,x4_array[count-1])
     #difference
        d1=np.subtract(f1,s11)
        d2=np.subtract(f2,s11)
        d3=np.subtract(f3,s11)
        d4=np.subtract(f4,s11)
    #multiplying with the projection matrices
        P11=np.matmul(P1,d1)
        P22=np.matmul(P2,d2)
        P33=np.matmul(P3,d3)
        P44=np.matmul(P4,d4)
    #scale by 1/N   
        d11=np.multiply(ir,P11)
        d22=np.multiply(ir,P22)
        d33=np.multiply(ir,P33)
        d44=np.multiply(ir,P44)
    #xi[count]=xi[count-1]-(1/N)*P{N*xi[count-1]-sum_over_j(xj[count-1])}   
        x1_array[count]=np.subtract(x1_array[count-1],d11)
        x2_array[count]=np.subtract(x2_array[count-1],d22)
        x3_array[count]=np.subtract(x3_array[count-1],d33)
        x4_array[count]=np.subtract(x4_array[count-1],d44)
     #How far is the estimate from the true solution?
        
        e1[:,count]=np.linalg.norm(np.subtract(np.matmul(A,x1_array[count-1]),b))
        e2[:,count]=np.linalg.norm(np.subtract(np.matmul(A,x2_array[count-1]),b))
        e3[:,count]=np.linalg.norm(np.subtract(np.matmul(A,x3_array[count-1]),b))
        e4[:,count]=np.linalg.norm(np.subtract(np.matmul(A,x4_array[count-1]),b))
 
 count=count+1        



plt.plot(count_array.T,e1.T,label='Agent 1')
plt.plot(count_array.T,e2.T,label='Agent 2')
plt.plot(count_array.T,e3.T,label='Agent 3')
plt.plot(count_array.T,e4.T,label='Agent 4')
plt.xlabel('Iteration count')
plt.ylabel('||Ax-b||2 for the agents 1-4')
plt.legend()
plt.grid()
plt.savefig('IterationCount500_1.pdf')
plt.show()
