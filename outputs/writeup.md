# Bài viết nộp — Lab 25: GPU FinOps Optimization

**Học viên:** Nguyễn Ngọc Gia Bảo · Mã: 2A202601234 · Track 2 (Infrastructure) · Day 25
**Kèm theo:** `outputs/report.md`, `outputs/savings.png`, `outputs/focus_export.csv`
**Trạng thái kiểm tra:** `python verify.py` → 11/11 · `pytest -q` → 15 passed

---

## 1. Baseline vs. Optimized

| Chỉ số | Baseline | Optimized | Thay đổi |
|---|---:|---:|---:|
| Tổng chi phí GPU / tháng | $18,005 | $9,434 | **−47.6%** |
| Inference `$/1M-token` | $6.488 | $1.126 | **−83%** |
| `$/1M-token` (đã tính phí ghi cache 1.25×) | $6.488 | $1.204 | −81% |
| Chi phí mua GPU / tháng | $16,539 (all on-demand) | $10,176 | −38.5% |

Con số tôi mang đi báo cáo là **$1.204/1M-token**, không phải $1.126. Phiên bản $1.126 bỏ qua phí **ghi** cache — nó đẹp hơn 5% nhưng không phải hóa đơn thật. Đây là nguyên tắc tôi rút ra: FinOps chỉ có giá trị khi con số chịu được kiểm toán ngược từ invoice.

## 2. Đòn bẩy nào đóng góp nhiều nhất?

| Lever | $/tháng | % tổng tiết kiệm |
|---|---:|---:|
| Purchasing (spot + reserved) | $6,363 | **74%** |
| Inference (cascade + cache + batch) | $1,212 | 14% |
| Tắt GPU idle | $600 | 7% |
| Right-size GPU nói dối util | $396 | 5% |

**Purchasing thắng áp đảo vì cơ cấu chi tiêu, không phải vì nó "hay" hơn.** Hạ tầng training/serving chiếm $16,539/tháng còn toàn bộ token inference chỉ $1,466/tháng — 92% tiền nằm ở GPU-giờ. Cùng một tỷ lệ giảm 40% thì áp lên chỗ nhiều tiền hơn sẽ ra nhiều tiền hơn.

Nhưng xét theo **tỷ lệ**, inference mới là đòn bẩy mạnh nhất: −83% so với −38.5% của purchasing. Lý do là chiết khấu inference **nhân** với nhau chứ không cộng: cascade (large→small: đơn giá input −93%, output −97%) × cache (0.1×) × batch (0.5×). Purchasing thì bị chặn trần bởi mức chiết khấu nhà cung cấp công bố (spot ~−40%, reserved 3yr ~−44%).

Hệ quả thực tế: khi NimbusAI tăng trưởng traffic, purchasing sẽ bão hòa còn inference tiếp tục co giãn — nên phải làm cả hai, theo thứ tự ở mục 5.

## 3. GPU-Util Lie

`gpu-h100-4` báo **98.2% GPU-Util** nhưng **MFU chỉ 0.194**, trong khi `gpu-h100-3` cùng loại H100 đạt MFU 0.427.

**Cơ chế:** `nvidia-smi` GPU-Util trả lời câu hỏi *"trong cửa sổ lấy mẫu vừa rồi có ít nhất một kernel đang chạy không?"* — đó là bộ đếm **thời gian bận**, không phải bộ đếm hiệu quả. Một kernel đứng chờ đọc HBM (memory stall), hoặc một chuỗi kernel nhỏ mà phần lớn thời gian là launch overhead và bubble giữa các stage, vẫn ghim con số ở ~100% trong khi tensor core rỗng. Roofline xác nhận: cường độ tính toán đo được là **245 FLOP/byte** so với ridge point **296 FLOP/byte** của H100 → job memory-bound, đói băng thông chứ không đói FLOPs.

**Tác động tài chính:** GPU đó tốn $2.50/h × 24 × 30 = **$1,800/tháng** và trả về ~19% FLOPs lý thuyết — tức khoảng 45% năng suất của một H100 khỏe trong cùng fleet. Nếu chỉ nhìn dashboard util 98%, đội platform sẽ kết luận "fleet đang full tải, cần mua thêm H100" — quyết định sai đắt nhất trong lab này. Hành động đúng đã đo được: chuyển sang MI300X (5.3 TB/s, $1.95/h) → tiết kiệm **$396/tháng** và job chạy nhanh hơn vì được đúng thứ nó thiếu là băng thông.

`gpu-a10g-1` cũng bị flag (96.9% util / MFU 0.268) nhưng phân tích ext-2 nói **giữ nguyên**: nó chỉ cần 0.21 TB/s, mà GPU rẻ hơn (L4, $0.80/h) chỉ có 0.30 TB/s ở MBU thực tế 60% → phải mua 2 chiếc = $1.60/h, đắt hơn A10G $1.00/h. Ở đây đòn bẩy là **tăng batch size**, không phải đổi phần cứng.

## 4. Năm extension đã thực hiện (đều có số đo)

| # | Extension | Kết quả đo được | Insight |
|---|---|---|---|
| 1 | `recommend_tier_v2()` — chính sách mua có định giá rủi ro (`finops/pricing.py`, dùng trong M3) | Policy cũ khai $9,849 (40.5% tiết kiệm); policy mới: $10,176 (38.5%) → **cũ thổi phồng $327/tháng** | 3 lỗi của policy cũ: (a) tính reservation theo giờ **dùng** trong khi commitment bill 24×7; (b) gán mọi spot pool cùng tỷ lệ thu hồi 5%/h — thực tế L4 15%/h vs H100 5%/h; (c) không so 1yr vs 3yr. Chiết khấu H100 3yr là −44% so với giá hôm nay nhưng chỉ **−35%** so với giá thị trường trung bình 3 năm (giá GPU giảm ~15%/năm). |
| 2 | Right-sizing theo MBU + roofline (`m1.rightsize_candidate`) | 1 lần đổi máy = **$396/tháng**; **5 GPU cố ý giữ nguyên** | Sizing theo **p95 băng thông + đỉnh VRAM + p95 FLOPs**, và **đếm số chiếc** GPU rẻ cần dùng. Giá theo băng thông đảo ngược hoàn toàn giá theo giờ: MI300X $0.37/TB-s-h, A100 $0.90, A10G $1.67, **L4 $2.67** — GPU "rẻ nhất" theo $/h lại đắt nhất theo thứ mà inference thực sự mua. |
| 3 | `cache_is_worth_it()` + `cached_cost_with_write()` | Điểm hòa vốn **0.28 read/write**; đo thực tế trong TTL 5 phút: assistant 1.87, search 1.42, rag 1.29, **eval 0.90** | Cache write 1.25× hoàn vốn nhờ read 0.1× → chỉ cần **1 lần tái sử dụng** trong TTL là có lời. Cả 4 team đều qua ngưỡng, nhưng eval chỉ hơn 3× — một tenant traffic mỏng hơn (hoặc TTL ngắn hơn) sẽ **lỗ** vì cache. Tính đúng phí ghi: $8.48 → $9.07/ngày. |
| 4 | Ngân sách reasoning (M2 + M5) | **8.4%** request = **16%** chi phí = **94%** năng lượng. Cap xuống 5% traffic: **$9/tháng + 358 kWh/tháng** | Một request reasoning tốn **148 Wh** so với **0.86 Wh** của request thường (~173×) vì hệ số năng lượng ~80× **nhân** với ~6× output token. Đây là đòn bẩy **năng lượng**, không phải đòn bẩy tiền — nếu chỉ nhìn hóa đơn USD sẽ không bao giờ thấy nó. Routing rule: chọn reasoning theo **độ phức tạp task** (đo bằng eval), không theo lựa chọn của người dùng. |
| 5 | Carbon-aware scheduling (`missions/m6_carbon_aware.py`) | Chuyển 5 job checkpointed (2,004 kWh/tháng) sang europe-north1: **−701 kg CO2e/tháng (−92%)**; us-east-wa rẻ nhất: **−$130/tháng tiền điện** | "Vùng tối ưu" có **ba đáp án khác nhau**: rẻ nhất us-east-wa ($110/mo), sạch nhất europe-north1 (0.06 tCO2e), cân bằng khi định giá carbon $100/tấn → us-east-wa ($128 blended). Chỉ job interruptible được chuyển: grid sạch nhất cách user ~95ms — miễn phí với training checkpointed, không chấp nhận được với chat. |

## 5. Nếu tôi là FinOps lead — 3 hành động đầu tiên

1. **Tắt GPU idle bằng auto-shutdown (tuần 1, $600/tháng).** `gpu-h100-5` idle 8 giờ mỗi đêm sau khi job kết thúc. Không cần thương lượng với ai, không rủi ro sản phẩm, một hook trong scheduler. Tiền miễn phí thì lấy trước — và nó tạo uy tín để xin duyệt hai việc sau.

2. **Tách flotilla: job checkpointed → spot, dịch vụ 24×7 → reserved 3yr (tháng 1, $6,363/tháng).** Đây là 74% tổng tiết kiệm. Điều kiện tiên quyết là **checkpointing phải thật** — tôi sẽ đo tỷ lệ thu hồi và thời gian rework thực tế 2 tuần trước khi ký commitment, vì mô hình đang giả định 5%/h cho H100. Chỉ commit đúng phần **baseload** đo được; phần đỉnh để on-demand. Không bao giờ commit dựa trên duty cycle của một tháng đẹp trời.

3. **Bật đo lường trước khi tối ưu inference (tháng 1–2, $1,212/tháng + mở đường cho tương lai).** Trước khi cascade 80% traffic sang model nhỏ, phải có **eval gate** — nếu không, ta cắt chi phí bằng cách âm thầm hạ chất lượng, và đó là món nợ đắt hơn tiền tiết kiệm. Song song: bật `$/1M-token` theo team trên dashboard (tag coverage đang 92% → cổng chargeback đã mở), rồi mới chuyển từ showback sang chargeback.

**Việc tôi sẽ không làm ngay:** mua thêm GPU. Fleet nhìn như đang full (util 93–98%) nhưng MFU trung bình chỉ ~0.31 — công suất đang nằm sẵn trong máy đã thuê, chỉ là chưa lấy ra được.

---

### Ghi chú về độ tin cậy

Toàn bộ số liệu sinh từ dữ liệu tổng hợp seed=25, giá là snapshot minh họa tháng 6/2026. Ba giả định nhạy nhất, xếp theo mức ảnh hưởng: (1) tỷ lệ thu hồi spot theo pool — sai 1 điểm % làm lệch bucket lớn nhất; (2) mức giảm giá thị trường 15%/năm dùng để định giá lock-in 3 năm; (3) hệ số năng lượng reasoning ~80×, quyết định toàn bộ kết luận về carbon. Cả ba cần re-baseline bằng số thật trước khi áp dụng.
