# Consensus Algorithm to solve Linear ALgebraic Equations
Consider the system Ax=b 
where A is a 1000x1000 random matrix
4 agents, each operating on a 250 x 1000 submatrix and the update equation in the paper [A Distributed Algorithm for Solving a Linear Algebraic Equation](https://ieeexplore.ieee.org/document/7063919) is used.

At every iteration,I've plotted ||Axi[n] -b||2 where xi[n] is the estimate of agent 'i' for the n th iteration to check if the estimates actually converge to the true solution to Ax=b.

Although from the plot,it's evident that the estimates of the 4 agents are comparable and are quite close to each other, the deviation from the 'True Solution' is still substantial. 

I tried running for more iterations to check if a better approximate to the true solution can be obtained with increasing number of iterations. 

But there does seem to be a huge dependence on the choice of the system matrix A & vector b. Although most of the time the estimates seem to converge and the deviation from the true solution did reduce,there were certain cases when the estimates did not converge at all after 50 iterations but are very close to each other  upto 25 iterations.


However,the other plots (IterationCount50,100,200,400,500) do reveal that the estimates converge and the error from true solution decreases very gradually with increasing number of iterations; the maximum decrease in error is observed before 50 iterations in all cases.

