using Test
using LogClustering
using LogClustering.Framing
using LogClustering.Framing: foreach_sd_element, foreach_sd_param,
                             SOURCE_SYSLOG_5424, SOURCE_K8S_CRI,
                             SOURCE_DOCKER_JSONL, SOURCE_RAW

@testset "Framing" begin
    @testset "RFC 5424 — NILVALUE SD, no message" begin
        # Minimal, well-formed 5424 with NILVALUE structured-data.
        line = "<34>1 2003-10-11T22:14:15.003Z mymachine.example.com su - ID47 -"
        f = parse_frame(line)
        @test f.source == SOURCE_SYSLOG_5424
        @test f.priority == 34
        @test f.version == 1
        @test f.timestamp == "2003-10-11T22:14:15.003Z"
        @test f.host == "mymachine.example.com"
        @test f.app == "su"
        @test f.procid == "-"
        @test f.msgid == "ID47"
        @test f.sd == "-"
        @test f.message == ""
    end

    @testset "RFC 5424 — single SD-ELEMENT, BOM-prefixed message" begin
        line = "<165>1 2003-10-11T22:14:15.003Z mymachine.example.com evntslog - ID47 " *
               "[exampleSDID@32473 iut=\"3\" eventSource=\"Application\" eventID=\"1011\"] " *
               "\ufeffAn application event log entry..."
        f = parse_frame(line)
        @test f.source == SOURCE_SYSLOG_5424
        @test f.priority == 165
        @test f.version == 1
        @test f.sd ==
              "[exampleSDID@32473 iut=\"3\" eventSource=\"Application\" eventID=\"1011\"]"
        @test f.message == "An application event log entry..."
    end

    @testset "RFC 5424 — two nested SD-ELEMENTs" begin
        line = "<165>1 2003-10-11T22:14:15.003Z mymachine.example.com evntslog - ID47 " *
               "[exampleSDID@32473 iut=\"3\" eventSource=\"Application\" eventID=\"1011\"]" *
               "[examplePriority@32473 class=\"high\"]"
        f = parse_frame(line)
        @test f.source == SOURCE_SYSLOG_5424
        ids = String[]
        foreach_sd_element(f.sd) do id, params
            push!(ids, String(id))
            if id == "exampleSDID@32473"
                pairs = Tuple{String, String}[]
                foreach_sd_param(params) do n, v
                    push!(pairs, (String(n), String(v)))
                end
                @test pairs == [("iut", "3"),
                                ("eventSource", "Application"),
                                ("eventID", "1011")]
            end
        end
        @test ids == ["exampleSDID@32473", "examplePriority@32473"]
    end

    @testset "RFC 5424 — escaped quote in PARAM-VALUE" begin
        line = "<13>1 2024-01-01T00:00:00Z host app - - " *
               "[tag@1 q=\"he said \\\"hi\\\"\" n=\"ok\"]"
        f = parse_frame(line)
        @test f.source == SOURCE_SYSLOG_5424
        found = Dict{String, String}()
        foreach_sd_element(f.sd) do _id, params
            foreach_sd_param(params) do n, v
                found[String(n)] = String(v)
            end
        end
        @test found["q"] == "he said \\\"hi\\\""
        @test found["n"] == "ok"
    end

    @testset "RFC 5424 — malformed falls back to RAW" begin
        # version is 0 → invalid per RFC (NONZERO-DIGIT)
        line = "<34>0 2003-10-11T22:14:15Z host app 1 - -"
        f = parse_frame(line)
        @test f.source == SOURCE_RAW
        @test f.message == line

        # Missing closing bracket in SD
        line2 = "<34>1 2003-10-11T22:14:15Z host app 1 - [bad@1 k=\"v\""
        f2 = parse_frame(line2)
        @test f2.source == SOURCE_RAW
    end

    @testset "Kubernetes CRI" begin
        line = "2016-10-06T00:17:09.669794202Z stdout F The content of the log entry."
        f = parse_frame(line)
        @test f.source == SOURCE_K8S_CRI
        @test f.timestamp == "2016-10-06T00:17:09.669794202Z"
        @test f.stream == "stdout"
        @test f.tag == "F"
        @test f.message == "The content of the log entry."

        # Partial tag
        line2 = "2016-10-06T00:17:09.669794202Z stderr P partial"
        f2 = parse_frame(line2)
        @test f2.source == SOURCE_K8S_CRI
        @test f2.stream == "stderr"
        @test f2.tag == "P"
    end

    @testset "Kubernetes CRI — reject bad stream/tag" begin
        bad = "2016-10-06T00:17:09.669794202Z stdin F message"
        @test parse_frame(bad).source == SOURCE_RAW
        bad2 = "2016-10-06T00:17:09.669794202Z stdout X message"
        @test parse_frame(bad2).source == SOURCE_RAW
    end

    @testset "Docker JSON-lines" begin
        line = "{\"log\":\"Hello world\\n\",\"stream\":\"stdout\",\"time\":\"2024-01-01T00:00:00.000000000Z\"}"
        f = parse_frame(line)
        @test f.source == SOURCE_DOCKER_JSONL
        @test f.message == "Hello world\\n"         # raw JSON value; un-escape downstream
        @test f.stream == "stdout"
        @test f.timestamp == "2024-01-01T00:00:00.000000000Z"
    end

    @testset "Docker JSON-lines — key reordering & extra key" begin
        line = "{\"time\":\"T1\",\"extra\":\"ignored\",\"stream\":\"stderr\",\"log\":\"oops\"}"
        f = parse_frame(line)
        @test f.source == SOURCE_DOCKER_JSONL
        @test f.timestamp == "T1"
        @test f.stream == "stderr"
        @test f.message == "oops"
    end

    @testset "Raw fallback" begin
        line = "this is just a plain line"
        f = parse_frame(line)
        @test f.source == SOURCE_RAW
        @test f.message == line
    end

    @testset "Zero-copy: fields are views of input" begin
        line = "<13>1 2024-01-01T00:00:00Z host app 1 - - hello"
        f = parse_frame(line)
        @test f.message == "hello"
        # `.string` is the backing String; all non-empty fields share it.
        @test f.timestamp.string === f.host.string === f.message.string
    end

    @testset "Hot path: allocations bounded and input-length-independent" begin
        # The only heap traffic per call is the returned `Frame` itself
        # (no intermediate allocations for SubStrings, digit parsing, SD
        # descent). Regression bound: ≤ 512 bytes per call, constant in
        # the size of the input.
        short = "<13>1 2024-01-01T00:00:00Z h a 1 - -"
        long  = "<165>1 2003-10-11T22:14:15.003Z mymachine.example.com evntslog - ID47 " *
                "[exampleSDID@32473 iut=\"3\" eventSource=\"Application\" eventID=\"1011\"]" *
                "[examplePriority@32473 class=\"high\"] " *
                "\ufeff" * repeat("payload ", 200)
        parse_frame(short); parse_frame(long)
        a_short = @allocated parse_frame(short)
        a_long  = @allocated parse_frame(long)
        @test a_short <= 512
        @test a_long  <= 512
        @test a_short == a_long                          # input-length-independent
    end
end
