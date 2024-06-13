import SPSAOptimizers
import SPSAOptimizers: serialize, deserialize

obj = SPSAOptimizers.PowerSeries()
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

obj = SPSAOptimizers.ConstantSeries(1.0)
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

obj = SPSAOptimizers.IntDictStream()
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

obj = SPSAOptimizers.BernoulliDistribution(L=4)
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

obj = SPSAOptimizers.SPSA1(4)
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

obj = SPSAOptimizers.TrajectoryHessian(4)
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

obj = SPSAOptimizers.SPSA2(4)
json = serialize(obj)
obj_ = deserialize(json)
@assert json == serialize(obj_)

#= TODO: Actually write the json strings and then load them.
    I anticipate some difficulty inferring types on empty data structures.
=#
