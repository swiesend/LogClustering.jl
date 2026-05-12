"""
    Memory

Umbrella for the memory tier — long-term SQLite store today, with
short-term Redis warm state, pattern catalog, insights, and
reporting modules slotting in alongside it.
"""
module Memory

include("Schema.jl")
using .Schema

include("SQLite.jl")
using .SQLiteStore
const SQLite = SQLiteStore

export Schema, SQLiteStore, SQLite

end # module Memory
