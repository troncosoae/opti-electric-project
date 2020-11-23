using JuMP
using SCS
using LinearAlgebra
using FFTW

N = 10
c = ones(N)
A = [
    zeros(1,N-2) 0;
    Matrix(I,N-2,N-2) zeros(N-2,1)
]
B = [1; zeros(N-2,1)]

model = Model(SCS.Optimizer)
set_silent(model)

@variable(model, S[1:N, 1:N], PSD)
@variable(model, P[1:N-1, 1:N-1], Symmetric)
@variable(model, r[1:N] >= 0)
# @variable(model, R_I[1:N])
@variable(model, R_R[1:N])
@variable(model, C[1:N-1])
@variable(model, D)
@variable(model, t >= 0)

# helper constraints
@constraint(model, [i = 0:N-2], r[i + 2] == r[N-i])  # r must be Symmetric

# oficial constraints
@constraint(model, r[2:N] .== C[1:N-1])
@constraint(model, r[1] == D)

@constraint(model, S[1:N-1, 1:N-1] .== P - transpose(A)*P*A)
@constraint(model, S[1:N-1, N] .== transpose(C) - transpose(A)*P*B)
@constraint(model, S[N, 1:N-1] .== C - transpose(B)*P*A)
@constraint(model, S[N,N] - 2*D + (transpose(B)*P*B)[1,1] == 0)

@constraint(model, [k = 1:N], sum(r[n]*cos(2*pi*(n-1)*(k-1)/N) for n in 1:N) == R_R[k])
# @constraint(model, [k = 1:N], -sum(r[n]*sin(2*pi*(n-1)*(k-1)/N) for n in 1:N) == R_I[k])

# @constraint(model, 1 - t <= R_R[1] <= 1 + t)
# @constraint(model, 1 - t <= R_R[2] <= 1 + t)
# @constraint(model, 1 - t <= R_R[N] <= 1 + t)

@constraint(model, R_R[1] + t >= 1)
@constraint(model, R_R[1] - t <= 1)
@constraint(model, R_R[2] + t >= 1)
@constraint(model, R_R[2] - t <= 1)
@constraint(model, R_R[N] + t >= 1)
@constraint(model, R_R[N] - t <= 1)

@constraint(model, R_R[5] + t >= 0)
@constraint(model, R_R[5] - t <= 0)
@constraint(model, R_R[6] + t >= 0)
@constraint(model, R_R[6] - t <= 0)
@constraint(model, R_R[7] + t >= 0)
@constraint(model, R_R[7] - t <= 0)



@objective(model, Min, t)
JuMP.optimize!(model)


print(model)


r = JuMP.value.(r)
R = fft(r)

obj_value = JuMP.objective_value(model)
println("Objective value: ", obj_value)
println("r: ")
show(stdout, MIME("text/plain"), r)
println()
println("R_R: ")
show(stdout, MIME("text/plain"), JuMP.value.(R_R))
# println()
# println("R_I: ")
# show(stdout, MIME("text/plain"), JuMP.value.(R_I))
println()
println()
println()

# println("A: ")
# show(stdout, MIME("text/plain"), A)
# println()
# println("B: ")
# show(stdout, MIME("text/plain"), B)
# println()
# println("C: ")
# show(stdout, MIME("text/plain"), JuMP.value.(C))
# println()
# println("D: ")
# show(stdout, MIME("text/plain"), JuMP.value(D))
# println()
# println()
# println("S: ")
# show(stdout, MIME("text/plain"), JuMP.value.(S))
# println()
# println("P: ")
# show(stdout, MIME("text/plain"), JuMP.value.(P))
