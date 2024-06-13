"""
"""
module Optimizers
    """
    """
    abstract type OptimizerType{F} end

    """
        iterate!(optimizer, fn, x; kwargs...)
    """
    function iterate! end

    """
        optimize!(optimizer, fn, x0; kwargs...)
    """
    function optimize! end




    """
        Record(optimizer, L)
    """
    function Record end

    """
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
    Simply allows for the more intuitive trace initialization, `Trace()`.
    Asking people to type `NamedTuple[]` just sounds so...pretentious.
    """
    function Trace()
        return NamedTuple[]
    end

    """
    Copies record and adds the copy to the trace.
    Optionally, list specific fields - all others will be omitted from the copy.
    """
    function trace!(trace, record, fields...)
        isempty(fields) && (fields = keys(record))
        recorded = NamedTuple(field => deepcopy(record[field]) for field in fields)
        push!(trace, recorded)
    end

    """
    TODO: Might we ever wish for a generator rather than a vector? Easy enough to do...
    """
    function trajectory(trace, field)
        isempty(trace) && return []
        first(trace)[field] isa Ref && return [record[field][] for record in trace]
        return reduce(hcat, record[field] for record in trace)
    end



end