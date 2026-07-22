# JSON-on-the-wire schema

`logcluster` emits two stable JSON surfaces:

- **`stream` stdout** — newline-delimited JSON records with an `event`
  discriminator. The primary, durable sink for trigger events.
- **stderr** — structured operator logs from `StructuredLog`
  (`level` / `ts` / `msg` + arbitrary kwargs).

The batch subcommands `classify --json`, `score --json`, and `rca --json`
emit one JSON object per record on stdout (no `event` discriminator,
since the stream context doesn't apply).

Pin to version 1.

---

## `stream` events on stdout

Every record is a single JSON object, one per line. Use `jq -c` to filter:

```sh
logcluster stream --tail /var/log/syslog \
  | jq -c 'select(.event=="trigger")'
```

### `event: "trigger"` — a rule fired (always emitted)

```jsonc
{ "event":     "trigger",
  "ts":        "2026-04-27T14:02:11.842Z",
  "rule_id":   "fatal_keywords",
  "rule_kind": "keyword",
  "severity":  "crit",
  "line_id":   12345,
  "line":      "kernel: Out of memory: Killed process 2031 (sshd)",
  "fields":    { "matched": ["OOM"] },
  "sinks":     ["stdout", "webhook:default"],
  "frame":     { "source": "syslog_5424", "host": "srv-1", "app": "kernel" },
  "drain":     { "cluster_id": 17, "template": "kernel: Out of memory: Killed process <*>" },
  "transformer_decoder": { "nll": 4.91 } }
```

Per-kind `fields`:

| `rule_kind`       | `fields`                                                |
|-------------------|---------------------------------------------------------|
| `keyword`         | `{ "matched": [<keyword>, ...] }`                       |
| `regex`           | `{ "match": "<substring>" }`                            |
| `score_threshold` | `{ "metric": "...", "value": F, "threshold": F }`       |
| `novel_cluster`   | `{ "model": "drain", "cluster_id": N }`                 |
| `novel_token`     | `{ "novel_tokens": [<string>, ...] }`                   |
| `rate_spike`      | `{ "count_in_window": N, "window_s": F, "baseline": F, "min_count": N }` |
| `volume_anomaly`  | `{ "count_in_window": N, "window_s": F, "baseline": F }` |

### `event: "line"` — full per-line record (only with `--emit-all`)

```jsonc
{ "event":     "line",
  "ts":        "2026-04-27T14:02:11.842Z",
  "line_id":   12345,
  "line":      "kernel: Out of memory: Killed process 2031 (sshd)",
  "frame":     { ... },
  "drain":     { ... },
  "transformer_decoder": { "nll": 4.91 },
  "novelty":   { "value_novelty": 0.12 },
  "triggered": ["fatal_keywords"] }
```

`triggered` is the list of rule ids that fired this line (possibly
empty when `--emit-all` is set without any matching rule).

### `event: "status"` — heartbeat

```jsonc
{ "event":     "status",
  "ts":        "...",
  "uptime_s":  312.4,
  "lines":     { "total": 31200, "rate_per_s": 99.8, "dropped_oversize": 2 },
  "triggers":  { "total": 14, "by_rule": { "fatal_keywords": 9, "high_nll": 5 } },
  "rotations": 1,
  "queue":     { "ingest": 12 } }
```

Emitted every `--status-interval` seconds (default 10; set 0 to disable).

### `event: "shutdown"` — final line of the run

`reason` is `"eof"` (input ended), `"signal"` (SIGINT — a graceful
`docker stop` / `systemctl stop`, exit 130), or `"max_events"`
(the `--max-events` limit was reached).

```jsonc
{ "event":         "shutdown",
  "ts":            "...",
  "reason":        "eof",
  "triggers_total": 14 }
```

---

## Stderr — operator logs

One JSON object per call (`--log-format json`, default):

```jsonc
{ "level": "info",  "ts": "...", "msg": "stream started",
  "model": "/etc/logcluster/models/tx-dec.jld2", "rules": "/etc/logcluster/rules.json" }
{ "level": "warn",  "ts": "...", "msg": "line dropped",
  "reason": "oversize", "bytes": 1048577, "limit": 65536 }
{ "level": "error", "ts": "...", "msg": "webhook failed",
  "url": "https://hooks.slack.com/…", "status": 503, "attempt": 5 }
```

Sensitive kwargs (`authorization`, `token`, `secret`, `api_key`,
`password`, `cookie`) are redacted to `***` before write. Webhook
URLs are logged as `scheme://host/…` only — path, query, and
userinfo are stripped, since Slack incoming-webhook URLs and
`?token=` query params are themselves secrets.

Switch to human text for development:

```sh
logcluster stream --log-format text --log-level debug ...
```

---

## Rule-bundle schema (`--rules FILE`)

```jsonc
{
  "version": 1,
  "defaults": {
    "severity":         "warn",   // info | warn | crit
    "cooldown_s":       30,
    "warmup_required":  true
  },
  "rules": [
    { "id":        "<unique>",
      "kind":      "score_threshold",
      "metric":    "transformer_decoder.nll",  // dotted path
      "comparison": ">",                       // > >= < <= ==
      "value":     "auto:p99",                 // fixed | auto:p99 | auto:p995 | auto:zscore:K
      "severity":  "warn",
      "cooldown_s": 30 },

    { "id": "novel_template", "kind": "novel_cluster",
      "model": "drain" },

    { "id": "rate", "kind": "rate_spike",
      "match": { "kind": "regex", "field": "line", "pattern": "..." },
      "window_s": 60, "min_count": 10, "baseline_multiplier": 3.0 },

    { "id": "vol", "kind": "volume_anomaly",
      "window_s": 60, "baseline_multiplier": 5.0 },

    { "id": "kw", "kind": "keyword",
      "field": "line", "keywords": ["FATAL", "OOM"],
      "case_sensitive": false },

    { "id": "rgx", "kind": "regex",
      "field": "line", "pattern": "..." }
  ],
  "routes": [
    { "match": { "severity": ["crit"] },
      "sinks": ["webhook:default"] }
  ],
  "sinks": {
    "webhook:default": {
      "url":     "https://alertmanager.example.com/api/v2/alerts",
      "method":  "POST",
      "headers": { "Content-Type": "application/json",
                   "Authorization": "Bearer ${env:ALERTMANAGER_TOKEN}" },
      "format":  "alertmanager"            // raw | slack | alertmanager
    }
  }
}
```

`${env:VAR}` placeholders in `url` and `headers` are substituted from
the process environment at delivery time.

The `routes` block is additive — `stdout` is always implied for every
firing rule.

---

## Exit codes

| Exit | Meaning                                                 |
|------|---------------------------------------------------------|
| 0    | Success (clean shutdown signal for `stream`).           |
| 1    | At least one rule fired (only when `--exit-on-trigger`). |
| 2    | Argument / config / schema error (no work done).        |
| 3    | I/O error: a model / detector / rules / patterns / data file was missing at boot, or the input source failed mid-run. |
| 130  | `stream` received SIGINT and shut down gracefully (drained channels, finalized the session, emitted the `shutdown` event). Deployments should stop the service with SIGINT — the systemd unit sets `KillSignal=SIGINT` and the container image `STOPSIGNAL SIGINT`. |
