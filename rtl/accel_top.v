// Accelerator top level: on-chip buffers + controller + systolic array.
// Computes C = A x B for A of size (m_tiles*N) x K and B of size K x (n_tiles*N).
// Matrix sizes must be multiples of N (pad with zeros in software otherwise).
//
// Buffer layouts (behavioural models of on-chip SRAMs, for simulation only;
// in a real chip these would be SRAM macros, so do not synthesize this file):
//   amem[mi*K + k]       = column k of row-tile mi of A   (lane i = A[mi*N+i][k])
//   bmem[nj*K + k]       = row k of column-tile nj of B   (lane j = B[k][nj*N+j])
//   cmem[j*CDEPTH + t*N + r] = bank j (array column j): C[mi*N+r][nj*N+j],
//                              where t = mi*n_tiles + nj is the tile number
module accel_top #(
    parameter N      = 4,
    parameter DW     = 8,
    parameter ACCW   = 32,
    parameter AW     = 16,
    parameter ADEPTH = 256,
    parameter BDEPTH = 256,
    parameter CDEPTH = 256
) (
    input  wire          clk,
    input  wire          rst,
    input  wire          start,
    input  wire [AW-1:0] m_tiles,
    input  wire [AW-1:0] n_tiles,
    input  wire [AW-1:0] k_len,
    output wire          busy,
    output wire          done
);
    // On-chip buffers.
    reg [N*DW-1:0] amem [0:ADEPTH-1];
    reg [N*DW-1:0] bmem [0:BDEPTH-1];
    reg [ACCW-1:0] cmem [0:N*CDEPTH-1];

    // Cycles per tile: K, but at least 2N-1 so the column readout never collides.
    wire [AW-1:0] tile_len = (k_len < 2*N - 1) ? (2*N - 1) : k_len;

    wire          rd_en, first, pad;
    wire [AW-1:0] a_addr, b_addr;
    reg           drained;

    os_controller #(.AW(AW)) u_ctrl (
        .clk(clk), .rst(rst), .start(start),
        .m_tiles(m_tiles), .n_tiles(n_tiles), .k_len(k_len), .tile_len(tile_len),
        .drained(drained), .busy(busy), .done(done),
        .rd_en(rd_en), .first(first), .pad(pad),
        .a_addr(a_addr), .b_addr(b_addr)
    );

    // Buffer reads take one cycle, so first/pad are delayed one cycle to match.
    reg [N*DW-1:0] a_q, b_q;
    reg            first_q, pad_q;
    always @(posedge clk) begin
        if (rst) begin
            a_q     <= 0;
            b_q     <= 0;
            first_q <= 1'b0;
            pad_q   <= 1'b1;
        end else begin
            if (rd_en) begin
                a_q <= amem[a_addr];
                b_q <= bmem[b_addr];
            end
            first_q <= first;
            pad_q   <= pad;
        end
    end

    wire [N*DW-1:0]   a_col = pad_q ? {N*DW{1'b0}} : a_q;
    wire [N*DW-1:0]   b_row = pad_q ? {N*DW{1'b0}} : b_q;
    wire [N*ACCW-1:0] col_result;
    wire [N-1:0]      col_valid;

    // The array is cleared at the start of every layer. The previous layer has
    // fully drained by then, and this stops the empty tile left behind by its
    // flush cycle from being reported as a result of the new layer.
    wire array_rst = rst | start;

    systolic_os_top #(.N(N), .DW(DW), .ACCW(ACCW)) u_array (
        .clk(clk), .rst(array_rst), .first(first_q),
        .a_col(a_col), .b_row(b_row),
        .col_result(col_result), .col_valid(col_valid)
    );

    // Output collectors: each column writes its results, in order, into its bank.
    reg [AW-1:0] cnt [0:N-1];
    integer j;
    always @(posedge clk) begin
        if (rst || start) begin
            for (j = 0; j < N; j = j + 1) cnt[j] <= 0;
        end else begin
            for (j = 0; j < N; j = j + 1)
                if (col_valid[j]) begin
                    cmem[j*CDEPTH + cnt[j]] <= col_result[j*ACCW +: ACCW];
                    cnt[j] <= cnt[j] + 1;
                end
        end
    end

    // Drained once every column has delivered m_tiles * n_tiles * N results.
    integer jj;
    always @* begin
        drained = 1'b1;
        for (jj = 0; jj < N; jj = jj + 1)
            if (cnt[jj] != m_tiles * n_tiles * N) drained = 1'b0;
    end
endmodule
