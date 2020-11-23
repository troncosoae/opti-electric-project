using JuMP
using SCS
using LinearAlgebra

###### Constants
length = 5

e = MathConstants.e
pi = MathConstants.pi

function get_A(n)
    A = zeros(Int32, n, n)
    println(size(A))
    for i in 1:1:size(A)[1] - 1
        A[i+1, i] = 1
    end
    return A
end

function get_B(n)
    B = zeros(Int32, n, 1)
    B[1, 1] = 1
    return B    
end

function get_AB(n)
    A = get_A(n)
    B = get_B(n)
    return A, B
end

A, B = get_AB(length - 1)
show(stdout, MIME("text/plain"), A)
show(stdout, MIME("text/plain"), B)

###### Model
model = Model(SCS.Optimizer)
set_silent(model)

###### Variables
# @variable(model, P[1:length - 1, 1:length - 1], Symmetric)
@variable(model, X[1:length, 1:length], PSD)
@variable(model, r[1:length]) 
@variable(model, R[1:length])  # Fourier transoform of r

###### Constraints
for i in 1:1:size(R)[1]
    # @constraint(model, 1 <= r[i] <= 2)
    euler_fact = zeros(ComplexF64, 5, 1)
    for j in 1:1:size(euler_fact)[1]
        euler_fact[j] = e^(-im*2*pi/length*(j-1)*(i-1))
    end
    show(stdout, MIME("text/plain"), euler_fact)
    @constraint(model, R[i] == dot(r, euler_fact))
end

# # Up left
# @constraint(model, X[1:length-1, 1: length-1] .== P - transpose(A) * P * A)
# # Up right
# @constraint(model, X[1:length-1, length] .== transpose(r[1:length-1]) - transpose(A) * P * B)
# # Down left
# @constraint(model, X[length, 1:length-1] .== r[2:length] - transpose(B) * P * A)
# # Down right
# @constraint(model, X[length, length] == 2*r[1] - transpose(B) * P * B)

###### Objecitive Function
@objective(model, Min, sum(r) + tr(X))
JuMP.optimize!(model)


println("optimal value: ", objective_value(model))
# println("X: ")
show(stdout, MIME("text/plain"), JuMP.value.(X))
show(stdout, MIME("text/plain"), JuMP.value.(r))

# println(
#     "\nx: ", sqrt(value(X[1,1])), 
#     "\ny: ", sqrt(value(X[2,2])), 
#     "\nz: ", sqrt(value(X[3,3])))
