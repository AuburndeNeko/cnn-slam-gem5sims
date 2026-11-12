`timescale 1ns/1ps
// Testbench for systolic_os_top: feeds UNSKEWED data (one column of A and one
// row of B per cycle, all lanes together) and checks every C[i][j] against a
// reference matrix multiply, for two tiles in a row.
module systolic_top_tb;
    localparam N = 4, K = 6, DW = 8, ACCW = 32;

    reg clk = 0, rst = 1, clear = 0;
    reg  [N*DW-1:0]     a_col = 0, b_row = 0;
    wire [N*N*ACCW-1:0] c_flat;

    reg signed [DW-1:0]   A [0:N*K-1];   // A[i][k] stored at A[i*K+k]
    reg signed [DW-1:0]   B [0:K*N-1];   // B[k][j] stored at B[k*N+j]
    reg signed [ACCW-1:0] got;
    integer i, j, k, t, trial, expected, errors = 0, seed = 11;

    systolic_os_top #(.N(N), .DW(DW), .ACCW(ACCW)) dut (
        .clk(clk), .rst(rst), .clear(clear),
        .a_col(a_col), .b_row(b_row), .c_flat(c_flat)
    );

    always #5 clk = ~clk;

    // Feed one tile: at cycle t < K, all of column t of A and row t of B;
    // then zeros while the data finishes moving through the array.
    task run_tile;
        begin
            for (t = 0; t < K + 2*N - 1; t = t + 1) begin
                for (i = 0; i < N; i = i + 1)
                    a_col[i*DW +: DW] = (t < K) ? A[i*K+t] : 0;
                for (j = 0; j < N; j = j + 1)
                    b_row[j*DW +: DW] = (t < K) ? B[t*N+j] : 0;
                @(posedge clk); #1;
            end
            a_col = 0; b_row = 0;
        end
    endtask

    task check_tile;
        begin
            for (i = 0; i < N; i = i + 1)
                for (j = 0; j < N; j = j + 1) begin
                    expected = 0;
                    for (k = 0; k < K; k = k + 1)
                        expected = expected + A[i*K+k] * B[k*N+j];
                    got = c_flat[(i*N+j)*ACCW +: ACCW];
                    if (got !== expected) begin
                        $display("FAIL tile %0d C[%0d][%0d]: got %0d expected %0d",
                                 trial, i, j, got, expected);
                        errors = errors + 1;
                    end
                end
        end
    endtask

    initial begin
        @(posedge clk); #1 rst = 0;

        for (trial = 0; trial < 2; trial = trial + 1) begin
            for (k = 0; k < N*K; k = k + 1) begin
                A[k] = $random(seed) % 128;
                B[k] = $random(seed) % 128;
            end
            if (trial == 0) begin
                A[0] = -128; B[0] = -128;
                A[1] = 127;  B[N] = 127;
            end

            clear = 1; @(posedge clk); #1 clear = 0;
            run_tile;
            check_tile;
        end

        if (errors == 0) $display("PASS: %0dx%0d array with skew, K=%0d, 2 tiles, all outputs correct", N, N, K);
        else             $display("%0d error(s)", errors);
        $finish;
    end
endmodule
