`timescale 1ns/1ps
// Accelerator testbench: loads A and B into the on-chip buffers, starts the
// controller, waits for done, and checks every element of C against a
// reference matrix multiply. Runs two layers:
//   layer 1: K = 8  (K >= 2N-1: tiles stream with no padding)
//   layer 2: K = 5  (K <  2N-1: the controller pads each tile to 2N-1 cycles)
module accel_tb;
    localparam N = 4, DW = 8, ACCW = 32, AW = 16, CDEPTH = 256;
    localparam MT = 2, NT = 3, KMAX = 8;          // C is (MT*N) x (NT*N) = 8 x 12

    reg clk = 0, rst = 1, start = 0;
    reg  [AW-1:0] k_len = 0;
    wire busy, done;

    reg signed [DW-1:0] A [0:MT*N*KMAX-1];        // A[r][k] at r*KMAX + k
    reg signed [DW-1:0] B [0:KMAX*NT*N-1];        // B[k][c] at k*(NT*N) + c
    reg [N*DW-1:0]      word;
    reg signed [ACCW-1:0] got;
    integer r, c, k, i, mi, nj, t, expected, errors = 0, seed = 3, cycles, layer, K;

    accel_top #(.N(N), .DW(DW), .ACCW(ACCW), .AW(AW), .CDEPTH(CDEPTH)) dut (
        .clk(clk), .rst(rst), .start(start),
        .m_tiles(MT[AW-1:0]), .n_tiles(NT[AW-1:0]), .k_len(k_len),
        .busy(busy), .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        @(posedge clk); #1 rst = 0;

        for (layer = 0; layer < 2; layer = layer + 1) begin
            K = (layer == 0) ? 8 : 5;

            // Random matrices; the first layer also gets the INT8 extremes.
            for (r = 0; r < MT*N*KMAX; r = r + 1) A[r] = $random(seed) % 128;
            for (r = 0; r < KMAX*NT*N; r = r + 1) B[r] = $random(seed) % 128;
            if (layer == 0) begin A[0] = -128; B[0] = -128; A[1] = 127; B[NT*N] = 127; end

            // Load the buffers in the layouts accel_top expects.
            for (mi = 0; mi < MT; mi = mi + 1)
                for (k = 0; k < K; k = k + 1) begin
                    for (i = 0; i < N; i = i + 1) word[i*DW +: DW] = A[(mi*N+i)*KMAX + k];
                    dut.amem[mi*K + k] = word;
                end
            for (nj = 0; nj < NT; nj = nj + 1)
                for (k = 0; k < K; k = k + 1) begin
                    for (i = 0; i < N; i = i + 1) word[i*DW +: DW] = B[k*(NT*N) + nj*N + i];
                    dut.bmem[nj*K + k] = word;
                end

            // Run the layer.
            k_len = K;
            start = 1; @(posedge clk); #1 start = 0;
            cycles = 0;
            while (!done && cycles < 10000) begin
                @(posedge clk); #1;
                cycles = cycles + 1;
            end
            if (!done) begin
                $display("FAIL layer %0d: timed out", layer);
                errors = errors + 1;
            end

            // Check every element of C.
            for (r = 0; r < MT*N; r = r + 1)
                for (c = 0; c < NT*N; c = c + 1) begin
                    expected = 0;
                    for (k = 0; k < K; k = k + 1)
                        expected = expected + A[r*KMAX + k] * B[k*(NT*N) + c];
                    t   = (r / N) * NT + (c / N);
                    got = dut.cmem[(c % N)*CDEPTH + t*N + (r % N)];
                    if (got !== expected) begin
                        $display("FAIL layer %0d C[%0d][%0d]: got %0d expected %0d", layer, r, c, got, expected);
                        errors = errors + 1;
                    end
                end

            $display("layer %0d: K=%0d, %0d tiles, %0d cycles start-to-done", layer, K, MT*NT, cycles);
        end

        if (errors == 0) $display("PASS: both layers correct (%0d x %0d outputs each)", MT*N, NT*N);
        else             $display("%0d error(s)", errors);
        $finish;
    end
endmodule
