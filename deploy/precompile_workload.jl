# Precompile workload — executed by PackageCompiler.create_sysimage to
# capture the hot paths an operator actually runs. The aim is to bake
# in:
#
#   - module load + __init__,
#   - rules JSON parsing + the seven evaluator dispatch arms,
#   - the streaming `_infer` -> evaluate -> JSON-encode pipeline,
#   - the StructuredLog JSON encoder.
#
# Anything that pulls Python (CondaPkg / UMAP / HDBSCAN) or that needs
# the Rust crate is intentionally left out so the workload runs in
# a stripped container.

using LogClustering
using LogClustering.CLI
using LogClustering.Rules
using LogClustering.Stream
using LogClustering.StructuredLog
using LogClustering.SinksWebhook
using LogClustering.Persistence
using Dates: now, UTC

# Silence stderr (in case redirect from CLI subcommands).
StructuredLog.set_level!(:error)

# --- Rules ------------------------------------------------------------

rs = Rules.default_rules(warmup_lines = 0, warmup_seconds = 0.0)

# Synthetic InferResult that touches every default rule kind.
for i in 1:5
    ir = Dict{String, Any}(
        "line"    => "FATAL: oom-killer invoked for pid 1234",
        "line_id" => i,
        "ts"      => string(now(UTC)),
        "drain"   => Dict("cluster_id" => i, "template" => "FATAL: <*> invoked for pid <*>"),
        "transformer_decoder" => Dict("nll" => Float64(i) * 0.5),
        "novelty" => Dict("value_novelty" => Float64(i) * 0.1),
    )
    Rules.evaluate(rs, ir)
end

# --- StructuredLog encoder --------------------------------------------

io = IOBuffer()
StructuredLog.set_format!(:json; stream = io)
StructuredLog.set_level!(:debug)
StructuredLog.info("precompile";  models = 1, lines = 5)
StructuredLog.warn("precompile";  reason = "oversize", bytes = 1)
StructuredLog.error_event("precompile"; status = 503, attempt = 1)
take!(io)
StructuredLog.set_level!(:info)

# --- CLI help paths ---------------------------------------------------

# `main(["--help"])` and the subcommand help paths exercise top-level
# dispatch + argument parsing + JSON3 + Markdown rendering.
for cmd in ("--help", "train", "classify", "score", "stream", "rules",
            "rca", "mask", "select-model")
    try
        CLI.main(String[cmd, "--help"])
    catch
    end
end

# --- A tiny end-to-end stream run ------------------------------------

# Use a temp file (no stdin redirection here — that's awkward inside the
# build workload). Driving the rule engine through evaluate() above
# already covered most of cmd_stream's hot path.

rules_path, rio = mktemp()
try
    write(rio,
          """{"version":1,"defaults":{"warmup_required":false,"cooldown_s":0},""" *
          """"rules":[{"id":"f","kind":"keyword","field":"line",""" *
          """"keywords":["FATAL"],"severity":"crit"}]}""")
    close(rio)
    data_path, dio = mktemp()
    try
        write(dio, "INFO ok\nFATAL boom\nINFO done\n")
        close(dio)
        CLI.main(String["stream",
                        "--rules", rules_path,
                        "--data",  data_path,
                        "--warmup-lines", "0",
                        "--warmup-seconds", "0",
                        "--status-interval", "0",
                        "--quiet",
                        "--log-level", "error"])
    finally
        rm(data_path; force = true)
    end
finally
    rm(rules_path; force = true)
end
