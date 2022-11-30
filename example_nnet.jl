using Printf

# Load X and y variable
using JLD
using Plots
data = load("basisData.jld")
(X, y) = (data["X"], data["y"])
(n, d) = size(X)

#add bias variable to beginning of NN
X_t = [ones(n, 1) X]
d += 1

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [8, 4]
nParams = NeuralNet_nParams(d, nHidden)
w = randn(nParams, 1)

max_iter = 10000
step_size = 1e-4
acc_limit = 1e-8

function NN_funObj(w, X, y, nHidden)
    (n, d) = size(X)
    batch_size = round(n * 0.8)

    f = 0
    g = zeros(nParams, 1)
    # The stochastic gradient update:
    for iter in 1:batch_size
        i = rand(1:n)
        (f_i, g_i) = NeuralNet_backprop(w, X[i, :], y[i], nHidden)
        g = g .+ g_i
        f = f + f_i
    end

    g = g ./ batch_size
    f = f ./ batch_size
    return (f, g)
end

include("findMin.jl")
funObj(w) = NN_funObj(w, X_t, y, nHidden)
w = findMin(funObj, w, maxIter=max_iter, epsilon=acc_limit, verbose=false)

xVals = -10:0.05:10
Xhat = zeros(length(xVals), 1)
Xhat[:] .= xVals

# Add bias to data before predicting
(n, d) = size(Xhat)
Xhat_t = [ones(n, 1) Xhat]

yhat = NeuralNet_predict(w, Xhat_t, nHidden)
scatter(X, y, legend=false, linestyle=:dot)
plot!(Xhat, yhat, legend=false)
gui()
sleep(0.1)
savefig("nn_regression_plot.png")
