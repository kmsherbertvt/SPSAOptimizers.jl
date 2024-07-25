module Optimizers
    """

    Used for dispatch to different optimization algorithms.

    Algorithms are organized by first-order, second-order, and so on.
    (Well, just those two for now,
        but I feel like there could be a "quasi-second order" someday.)

    Constructors are documented with concrete types (e.g. `SPSA1`),
        but details on optimization options,
        record schematics, and `iterate!` keyword arguments
        are found in the intermediate order types, e.g. `FirstOrderOptimizer`.


    """
    abstract type OptimizerType{F} end

    """
        iterate!(optimizer, fn, x; kwargs...)

    Advance by one step in an optimization routine, updating `x` in place.

    In addition to updating `x`, this will update the state of `optimizer`.

    # Parameters
    - optimizer::OptimizerType{F} - defines the optimization algorithm
    - fn - the loss-function, such that fn(x) returns a number of type F
    - x::AbstractVector{F} - the best guess so far

    Keyword arguments depend on the type of `optimizer`.

    """
    function iterate! end

    """
        optimize!(optimizer, fn, x0; kwargs...)

    Run an optimization routine to convergence, updating `x` in place.

    In addition to updating `x`, this will update the state of `optimizer`.

    # Parameters
    - optimizer::OptimizerType{F} - defines the optimization algorithm
    - fn - the loss-function, such that fn(x) returns a number of type F
    - x::AbstractVector{F} - the best guess so far

    # Keyword Arguments
    - maxiter::Int - the max number of iterations to attempt, successful or not
    - callback - function called at each iteration, successful or not:

        callback(optimizer, iterate)

      where `iterate` is a record of the proposed step.
      The callback is called immediately prior
        to deciding whether to accept or reject the step,
        and the step is always rejected if the callback returns true.

    - record, average_last, trace, tracefields - see Output section below

    Additional keyword arguments may be available
        depending on the type of `optimizer`.

    # Output

    A `record` (a `NamedTuple` with schema specified by the `optimizer`)
        is returned at the end.
    If you provide an integer to `average_last`,
        that record is an average of the last so many (successful) iterations.

    If you provide an object to the `record` keyword argument,
        that object will be the one used for the final return output.
    More importantly, this same object is used to keep track of the
        last successful iteration,
        so it will have meaningful data if something (e.g. keyboard interrupt)
        disrupts the algorithm.
    Note however that if terminated prematurely,
        `record` will only represent the last successful iteration,
        rather than an average, even if `average_last` is provided.

    You can also provide a `trace`,
        which is a list of records of every single step.
    Provide a collection of field names (as symbols) to specify which fields
        should be saved in the trace.
    The default is the collection of all scalar attributes.
    Set it to an empty collection to trace every possible attribute.

    """
    function optimize! end




    """
        Record(optimizer, L)

    Initialize a mutable object giving the status of an `L`-dimensional optimization.

    The object is a `NamedTuple` of vectors and references,
        whose precise schema depends on the type of `optimizer`.

    """
    function Record end

    """
        copyrecord!(archive, record)

    Copy all values from the `record` Record to the `archive` Record.
    """
    function copyrecord!(archive, record)
        for field in keys(archive)
            if archive[field] isa Ref
                # NOTE: For some reason, copy!(::Ref, ::Ref) doesn't work.
                archive[field][] = record[field][]
                # TODO: Surely there is an existing better way to do this?
            else
                copy!(archive[field], record[field])
            end
        end
        return archive
    end

    """
        averagerecords!(archive, records)

    Like `copyrecord!`, but stores an average of many records in `archive`.
    """
    function averagerecords!(archive, records)
        n = length(records)
        for field in keys(archive)
            if archive[field] isa Ref
                archive[field][] = sum(record[field][] for record in records) / n
            else
                archive[field] .= sum(record[field] for record in records) ./ n
            end
        end
    end


    """
        Trace()

    Initialize a vector of Records, giving a trace of an optimization.

    Simply allows for the more intuitive trace initialization, `Trace()`.
    Asking people to type `NamedTuple[]` just sounds so...pretentious.
    """
    function Trace()
        return NamedTuple[]
    end

    """
        trace!(trace, record, fields...)

    Copies a record and adds the copy to the trace.

    Optionally, list specific fields - all others will be omitted from the copy.

    """
    function trace!(trace, record, fields...)
        isempty(fields) && (fields = keys(record))
        recorded = NamedTuple(field => deepcopy(record[field]) for field in fields)
        push!(trace, recorded)
    end

    """
        trajectory(trace, field)

    Extract the values of a particular field from each record in a trace,
        and return as a vector (if `field` names a reference)
        or a matrix, where each column is the vector stored in one record.

    """
    function trajectory(trace, field)
        isempty(trace) && return []
        first(trace)[field] isa Ref && return [record[field][] for record in trace]
        return reduce(hcat, record[field] for record in trace)
    end



end