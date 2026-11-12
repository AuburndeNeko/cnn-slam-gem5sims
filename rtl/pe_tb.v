`timescale 1ns/1ps
// Testbench: feeds four (a, b) pairs into one PE, checks the running sum,
// the pass-through outputs, and the clear signal.
module pe_tb;
    reg clk = 0, rst = 1, clear = 0;
    reg  signed [7:0]  a_in = 0, b_in = 0;
    wire signed [7:0]  a_out, b_out;
    wire signed [31:0] acc;

    integer k, errors = 0, expected = 0;
    reg signed [7:0] A [0:3];
    reg signed [7:0] B [0:3];

    pe #(.DW(8), .ACCW(32)) dut (
        .clk(clk), .rst(rst), .clear(clear),
        .a_in(a_in), .b_in(b_in),
        .a_out(a_out), .b_out(b_out), .acc(acc)
    );

    always #5 clk = ~clk;   // 10 ns clock period

    initial begin
        // Includes negatives and the INT8 extremes to test signed handling.
        A[0] = 3;   A[1] = -4; A[2] = 127; A[3] = -128;
        B[0] = 2;   B[1] = 5;  B[2] = 127; B[3] = -128;

        @(posedge clk); #1 rst = 0;            // release reset

        // Pipelined PE: the product of step k reaches acc one edge later,
        // so after edge k, acc holds the sum of steps 0..k-1.
        for (k = 0; k < 4; k = k + 1) begin
            a_in = A[k]; b_in = B[k];
            @(posedge clk); #1;
            if (acc !== expected) begin
                $display("FAIL step %0d: acc=%0d expected=%0d", k, acc, expected);
                errors = errors + 1;
            end
            if (a_out !== A[k] || b_out !== B[k]) begin
                $display("FAIL step %0d: pass-through a_out=%0d b_out=%0d", k, a_out, b_out);
                errors = errors + 1;
            end
            expected = expected + A[k] * B[k];
        end

        // One more edge with zero inputs lets the last product land.
        a_in = 0; b_in = 0;
        @(posedge clk); #1;
        if (acc !== expected) begin
            $display("FAIL final: acc=%0d expected=%0d", acc, expected);
            errors = errors + 1;
        end

        // clear now loads the product registered alongside it. With zero
        // inputs that product is 0, so acc should be 0 two edges later.
        clear = 1;
        @(posedge clk); #1 clear = 0;
        @(posedge clk); #1;
        if (acc !== 0) begin
            $display("FAIL clear: acc=%0d", acc);
            errors = errors + 1;
        end

        if (errors == 0) $display("PASS: sum was %0d (expected 32499), clear works", expected);
        else             $display("%0d error(s)", errors);
        $finish;
    end
endmodule
