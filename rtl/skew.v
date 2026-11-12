// Skew stage for the systolic array.
// Takes N values that arrive in the same cycle (one column of A, or one row
// of B, as read from SRAM) and delays lane i by i clock cycles, producing the
// staircase the array needs. Lane 0 passes straight through.
module skew #(
    parameter N  = 4,
    parameter DW = 8
) (
    input  wire            clk,
    input  wire            rst,
    input  wire [N*DW-1:0] in_vec,    // lane i = in_vec[i*DW +: DW]
    output wire [N*DW-1:0] out_vec    // lane i, delayed by i cycles
);
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : g_lane
            if (i == 0) begin : g_pass
                assign out_vec[0 +: DW] = in_vec[0 +: DW];
            end else begin : g_delay
                reg [DW-1:0] sr [0:i-1];   // i-stage shift register
                integer s;
                always @(posedge clk) begin
                    if (rst) begin
                        for (s = 0; s < i; s = s + 1) sr[s] <= 0;
                    end else begin
                        sr[0] <= in_vec[i*DW +: DW];
                        for (s = 1; s < i; s = s + 1) sr[s] <= sr[s-1];
                    end
                end
                assign out_vec[i*DW +: DW] = sr[i-1];
            end
        end
    endgenerate
endmodule
