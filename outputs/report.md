# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $13,736  
**Projected savings:** $13,397  (**49%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,930 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cleanest region: europe-north1
- Lowest energy-price region: us-east-wa
- Carbon-weighted balanced region: europe-north1

_Figures are June-2026 as-of snapshots; re-baseline before acting._

---

## Phân tích chi tiết (FinOps Engineer — NimbusAI)

### 1. Vì sao "GPU-Util 98%" là một lời nói dối tốn kém

`nvidia-smi` báo gpu-h100-4, gpu-a10g-1 đạt GPU-Util rất cao nhưng MFU thấp — bạn trả đủ tiền cho
cả giờ GPU nhưng chỉ nhận được một phần nhỏ FLOPs mà con chip có thể sinh ra.
Cơ chế đằng sau:

- **GPU-Util % chỉ đo "clock có đang bận"** (time-active), không đo hiệu quả tính toán.
  GPU có thể "bận" vì đang chờ dữ liệu từ HBM (memory stall), chờ kernel launch,
  hoặc chạy các kernel nhỏ không tận dụng được tensor cores.
- **MFU = achieved FLOPs / peak FLOPs** mới phản ánh đúng phần trăm sức mạnh tính toán
  thực sự được dùng. Với workload memory-bound (LLM decode ~1–2 FLOP/byte, dưới ridge
  point 295 FLOP/byte của H100), GPU "bận" nhưng băng thông HBM là nút thắt — FLOPs
  không thể đạt đỉnh.
- **Tác động tài chính:** các GPU bị "lie" (gpu-h100-4 (H100, util 98.2%, MFU 0.194), gpu-a10g-1 (A10G, util 96.9%, MFU 0.268)) đang bị over-provisioned.
  Right-size xuống tier thấp hơn tiết kiệm **$655/tháng**;
  cộng với việc tắt GPU idle (utilization < 10%) tiết kiệm thêm
  **$600/tháng**.

### 2. Phân tích từng đòn bẩy tiết kiệm

| Đòn bẩy | Tiết kiệm/tháng | % tổng savings | Ghi chú |
|---|---|---|---|
| Purchasing (spot/reserved) | $10,930 | 82% | Lever lớn nhất — chọn đúng tier theo duty cycle + interruption rate |
| Inference (cascade/cache/batch) | $1,212 | 9% | $/1M-token giảm từ $6.488 → $1.126 (−82.6%) |
| Right-size util-lies | $655 | 5% | Hạ cấp GPU "nói dối" xuống tier thấp hơn |
| Kill idle GPUs | $600 | 4% | Tắt GPU chạy không, utilization < 10% |

**Vì sao Purchasing là lever lớn nhất?** Vì hóa đơn GPU chủ yếu đến từ việc *thuê
instance chạy 24/7* với giá on-demand đắt nhất. Chuyển 24/7 inference sang reserved
(−45%) và job training có thể gián đoạn sang spot tác động trực tiếp lên số giờ GPU
lớn nhất. Đây là lever "một lần quyết định, tiết kiệm hàng tháng" — khác với
inference lever cần thay đổi hạ tầng routing.

### 3. Đề xuất hành động theo thứ tự ROI

1. **Purchasing (tuần 1):** Ký reserved 3yr cho workload ổn định (duty ≥ 55%) và
   job training duty cao mà spot effective không còn rẻ hơn reserved; chuyển job
   interruptible còn lại sang spot. Policy mở rộng tăng savings M3 từ **39.6% lên
   42.6%**. → **$10,930/tháng**.
2. **Inference (tuần 2–3):** Bật cascade (80% request chỉ cần model nhỏ), prompt
   caching (chat/rag có system prompt tĩnh, cache-hit cao), batch API cho eval.
   → **$1,212/tháng** + giảm $/1M-token 82.6%.
3. **Right-size + kill idle (tuần 2):** Hạ cấp GPU "nói dối", tắt GPU idle ngoài giờ.
   → **$1,255/tháng**.
4. **Reasoning budget (tuần 4):** Dataset đã dưới cap 10%; chỉ dùng reasoning khi
   complexity score ≥ 0.8 hoặc tác vụ cần multi-step/tool use, rồi áp strict cap 5%.
   Đây là extension scenario tách riêng, không cộng vào waterfall M5 bốn lever.
   → **$17/tháng** + giảm 12,004 Wh/ngày (gross avoidance; cần
   kiểm tra chất lượng route thay thế).

### 4. Tính bền vững — carbon gắn liền với chi phí

- **Energy per query:** 0.24 Wh; **carbon per query:** 0.091 gCO2e (tại us-east-1).
- **Reasoning là "quả bom năng lượng":** 8.4% traffic nhưng tiêu
  29,788 Wh/ngày — gấp ~15.8× năng lượng của phần
  traffic còn lại (1,888 Wh/ngày), vì reasoning tiêu thụ ~80× năng
  lượng một query thường.
- **Vùng triển khai:** `europe-north1` (Na Uy, thủy điện) có carbon 30 gCO2/kWh —
  sạch hơn 22× so với `europe-central2` (Ba Lan, 660 gCO2/kWh). `us-east-wa` là
  vùng có điện rẻ nhất; lựa chọn cân bằng trọng số carbon 75% / giá điện 25% là
  `europe-north1`. Chuyển job interruptible từ us-east-1 sang
  europe-north1 cắt **626,150 gCO2e (−92.1%)**.
- **Trade-off:** vùng sạch nhất có thể xa users nhất (latency tăng). Với job training
  không real-time, đánh đổi này hoàn toàn chấp nhận được — training không cần
  low-latency, nên ưu tiên carbon + chi phí điện.

### 5. Kết luận

Tổng tiết kiệm **$13,397/tháng (49%)** — nằm trong band
40–95% mục tiêu. Đo bằng `$/1M-token`, chi phí inference giảm từ
**$6.488 → $1.126** (−82.6%).
Ba hành động đầu tiên nếu là FinOps lead: (1) ký reserved + chuyển spot ngay,
(2) bật cascade/cache/batch, (3) right-size GPU "nói dối" và tắt GPU idle.
