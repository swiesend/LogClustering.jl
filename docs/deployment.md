# Deployment guide

This page covers running `logcluster` as a production service. The
batch subcommands (`train`, `classify`, `score`, `rca`) are
out-of-band one-shots; `stream` is the long-running daemon that
matters here.

The wire schema for everything emitted on stdout / stderr is pinned
in [`json-schema.md`](json-schema.md).

---

## 1. Train a model

`stream` works without any model (rule-only mode), but rules that
read model signals — `score_threshold` against
`transformer_decoder.nll`, `novel_cluster` against `drain` — need a
saved bundle.

```sh
# Drain templates over a representative slice of historical logs.
logcluster train --kind drain \
                 --data /var/log/archive/syslog.last7d \
                 --out  /etc/logcluster/models/drain.jld2

# Transformer decoder for perplexity-style scoring. Tune --d-model
# and --seqlen to your traffic; the defaults match the planning doc.
logcluster train --kind transformer_decoder \
                 --data /var/log/archive/syslog.last7d \
                 --out  /etc/logcluster/models/transformer-decoder.jld2 \
                 --seqlen 32 --d-model 128 --n-layers 4 \
                 --epochs 20 --batch 64
```

Both bundles carry a `corpus_fingerprint` in their metadata; the
`--reuse` flag on `train` skips retraining when a compatible bundle
already lives in `--registry` (see `logcluster select-model --help`).

---

## 2. Author a rules file

Start from the bundled defaults:

```sh
logcluster rules --print-defaults > /etc/logcluster/rules.json
```

Edit the file, then validate before reloading:

```sh
logcluster rules --validate /etc/logcluster/rules.json   # exits 2 on error
logcluster rules --explain  /etc/logcluster/rules.json   # one line per rule
```

Dry-run against a captured corpus to tune thresholds:

```sh
logcluster rules --dry-run \
                 --rules /etc/logcluster/rules.json \
                 --data  /var/log/archive/syslog.yesterday \
                 --model /etc/logcluster/models/transformer-decoder.jld2
# -> { "lines": 81433, "by_rule": { "fatal_keywords": 11, "score_p99": 824 } }
```

A `score_p99` count near 1% of `lines` is the expected steady state
for `value: "auto:p99"`; aim for `error_rate_spike` and
`fatal_keywords` to fire only on genuine incidents in the replay.

See [`examples/rules-production.json`](../examples/rules-production.json)
for a fully-routed bundle with alertmanager + slack webhook sinks.

---

## 3. Run under Docker

Build the image once (multi-stage; produces a baked sysimage so
cold-start `logcluster --help` runs in <300 ms):

```sh
docker build -t ghcr.io/swiesend/logclustering:0.2.0 \
             -f deploy/Dockerfile .
```

Smoke test:

```sh
echo "FATAL panic" | docker run --rm -i \
  ghcr.io/swiesend/logclustering:0.2.0 \
  stream --warmup-lines 0 --warmup-seconds 0 --quiet \
         --log-level error
# -> {"event":"trigger","rule_id":"fatal_keywords",...}
```

Operator deployment — see [`deploy/docker-compose.yml`](../deploy/docker-compose.yml):

```sh
cd deploy && \
ALERTMANAGER_TOKEN=... SLACK_WEBHOOK_URL=... docker compose up -d
```

The container's stdout carries newline-delimited JSON trigger
records. Configure your log driver / collector (`vector`,
`fluent-bit`, `promtail`) to consume them. The example compose file
uses the `json-file` driver with rotation.

---

## 4. Run under systemd

Install the project, build a sysimage, drop the units in place:

```sh
sudo mkdir -p /opt/logclustering /etc/logcluster/models /var/log/logcluster
sudo rsync -av --delete <repo>/ /opt/logclustering/
sudo julia --project=/opt/logclustering /opt/logclustering/deploy/build_sysimage.jl \
           /opt/logclustering/LogClustering.so

sudo install -o root -g root -m 0644 \
     /opt/logclustering/deploy/systemd/logcluster.service \
     /etc/systemd/system/logcluster.service
sudo install -o root -g root -m 0644 \
     /opt/logclustering/deploy/systemd/logcluster.tmpfiles \
     /etc/tmpfiles.d/logcluster.conf
sudo systemd-tmpfiles --create /etc/tmpfiles.d/logcluster.conf

# Verify the unit parses cleanly.
sudo systemd-analyze verify /etc/systemd/system/logcluster.service

# Drop secrets into /etc/logcluster/env (mode 0640, owned by root).
sudo install -o root -g logcluster -m 0640 /dev/null /etc/logcluster/env
sudo tee -a /etc/logcluster/env <<'EOF'
ALERTMANAGER_TOKEN=...
SLACK_WEBHOOK_URL=...
EOF

sudo useradd -r -s /sbin/nologin -d /var/lib/logcluster logcluster || true
sudo systemctl daemon-reload
sudo systemctl enable --now logcluster.service
```

Watch the service:

```sh
sudo journalctl -u logcluster -f       # stderr (operator logs)
sudo tail -f /var/log/logcluster/triggers.jsonl   # stdout (trigger JSON)
```

The unit is hardened (`ProtectSystem=strict`, `ProtectHome=true`,
`PrivateTmp=true`, `NoNewPrivileges=true`, read-only paths under
`/etc/logcluster` and `/opt/logclustering`). Memory is capped at
2 GiB and 64 tasks — tune to your traffic.

---

## 5. Shipping the JSON output downstream

`logcluster` is shaped so the output is a primary, durable event
stream rather than a notification side-channel. A few common
sinks:

### Loki (via Promtail or Vector)

```yaml
# vector.toml
[sources.logcluster]
type = "file"
include = ["/var/log/logcluster/triggers.jsonl"]

[transforms.parse_json]
type = "remap"
inputs = ["logcluster"]
source = '. = parse_json!(.message)'

[sinks.loki]
type = "loki"
inputs = ["parse_json"]
endpoint = "http://loki:3100"
labels.severity = "{{ severity }}"
labels.rule_id  = "{{ rule_id }}"
labels.app      = "logcluster"
encoding.codec  = "json"
```

### SQLite (analyst-friendly local store)

```sh
tail -F /var/log/logcluster/triggers.jsonl \
  | jq -c 'select(.event=="trigger") |
           { ts, rule_id, severity, line_id, line }' \
  | sqlite-utils insert /var/lib/logcluster/triggers.sqlite triggers - \
        --nl --pk=ts --alter
```

### Alertmanager (built-in)

Use `examples/rules-production.json` as a starting point — it
already routes `crit` events to a `webhook:alertmanager` sink with
the Prometheus Alertmanager v2 body format.

---

## 6. Tuning your rules

### Warmup

`--warmup-lines N` (default 500) and `--warmup-seconds S` (default 30)
buffer the first N lines (whichever gate trips first). During this
window:

- `novel_cluster` populates its baseline set;
- `rate_spike` / `volume_anomaly` populate their per-window counters;
- `score_threshold` populates the reservoir for `auto:*` thresholds.

Rules with `warmup_required: true` (the default) wait for the gate to
clear before firing. Rules with `warmup_required: false` fire from
line 1 — appropriate for unambiguous signals like `fatal_keywords`
or `regex_oom_kill`.

### `auto:*` thresholds

| Value         | Meaning                                                                |
|---------------|------------------------------------------------------------------------|
| `auto:p99`    | 99th percentile of a 1024-element reservoir, refreshed on each line.   |
| `auto:p995`   | 99.5th percentile. Use when p99 is too noisy on a high-cardinality metric. |
| `auto:zscore:K` | `mean + K * std` over the reservoir; a Gaussian-shaped tail.          |

The reservoir adapts slowly, so a sustained shift in the underlying
distribution will eventually re-quiet the rule. For a stricter "alert
on regressions" semantic, swap to a fixed threshold (`value: 4.5`).

### Cooldowns

`cooldown_s` suppresses repeat firings of the same rule for the same
event class. Pick a value bigger than the typical incident duration
so you get one alert per outage, not one per matching log line. A
follow-up `crit` from a different rule still fires.

### Drop-on-full webhooks

Webhook delivery is best-effort. If the rate of `crit` triggers
exceeds the webhook endpoint's tolerance for any reason, the producer
**drops** the delivery (with a stderr warning) so the inference loop
keeps up. stdout JSON is the durable record — the dropped delivery
shows up there.
