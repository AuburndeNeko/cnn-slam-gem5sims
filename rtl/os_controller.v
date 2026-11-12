// Controller for the output-stationary systolic array.
// Walks over all N x N output tiles of C = A x B (row-tile mi outer, column-tile
// nj inner) and, for each tile, issues K buffer reads (k = 0..K-1) with
// `first` on k = 0. If K < 2N-1 the tile is padded to 2N-1 cycles with zero
// data, so the array's column readout never collides. After the last tile it
// sends one flush cycle, waits until all results are written, then pulses done.
//
// Outputs are combinational from the state, for the CURRENT cycle; the buffers
// return data one cycle later, so the top level delays first/pad by one cycle.
module os_controller #(
    parameter AW = 16                    // width of counters and addresses
) (
    input  wire          clk,
    input  wire          rst,
    input  wire          start,          // pulse to begin a layer (when idle)
    input  wire [AW-1:0] m_tiles,        // number of row-tiles of C   (M / N)
    input  wire [AW-1:0] n_tiles,        // number of column-tiles of C (cols / N)
    input  wire [AW-1:0] k_len,          // shared dimension K
    input  wire [AW-1:0] tile_len,       // cycles per tile: max(K, 2N-1), set by top
    input  wire          drained,        // all results have been written to C
    output wire          busy,
    output reg           done,           // stays high until the next start
    output reg           rd_en,          // read the A and B buffers this cycle
    output reg           first,          // this cycle carries k = 0 of a tile
    output reg           pad,            // this cycle carries zero data
    output reg  [AW-1:0] a_addr,
    output reg  [AW-1:0] b_addr
);
    localparam IDLE = 2'd0, RUN = 2'd1, FLUSH = 2'd2, DRAIN = 2'd3;

    reg [1:0]    state;
    reg [AW-1:0] mi, nj, k;
    reg [AW-1:0] a_base, b_base;   // start address of the current A / B tile

    assign busy = (state != IDLE);

    // What to do this cycle.
    always @* begin
        rd_en  = 1'b0;
        first  = 1'b0;
        pad    = 1'b1;
        a_addr = a_base + k;
        b_addr = b_base + k;
        case (state)
            RUN: begin
                first = (k == 0);
                pad   = (k >= k_len);      // padding cycles at the end of a short tile
                rd_en = !pad;
            end
            FLUSH: begin
                first = 1'b1;              // hands over the last tile's results
            end
            default: ;
        endcase
    end

    // Advance through the tiles.
    always @(posedge clk) begin
        if (rst) begin
            state  <= IDLE;
            done   <= 1'b0;
            mi     <= 0;
            nj     <= 0;
            k      <= 0;
            a_base <= 0;
            b_base <= 0;
        end else begin
            case (state)
                IDLE: if (start) begin
                    state  <= RUN;
                    done   <= 1'b0;
                    mi     <= 0;
                    nj     <= 0;
                    k      <= 0;
                    a_base <= 0;
                    b_base <= 0;
                end
                RUN: begin
                    if (k == tile_len - 1) begin
                        k <= 0;
                        if (nj == n_tiles - 1) begin
                            nj     <= 0;
                            b_base <= 0;
                            if (mi == m_tiles - 1) begin
                                state <= FLUSH;
                            end else begin
                                mi     <= mi + 1;
                                a_base <= a_base + k_len;
                            end
                        end else begin
                            nj     <= nj + 1;
                            b_base <= b_base + k_len;
                        end
                    end else begin
                        k <= k + 1;
                    end
                end
                FLUSH: state <= DRAIN;
                DRAIN: if (drained) begin
                    state <= IDLE;
                    done  <= 1'b1;
                end
            endcase
        end
    end
endmodule
