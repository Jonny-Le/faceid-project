import os, time, platform
import torch
import torch.nn as nn
from torch import amp

# ============================================================
# verify_env.py – Environment and GPU benchmark check
# Kiểm tra toàn bộ môi trường AI và hiệu năng GPU
# ============================================================

def print_header(title_en, title_vi):
    print("\n" + "="*100)
    print(f"{title_en}\n{title_vi}")
    print("="*100)

def main():
    # ------------------------------------------------------------
    # Clear screen before running (for readability)
    # Xóa toàn bộ cửa sổ trước khi in kết quả
    # ------------------------------------------------------------
    os.system('clear')
    print("🧠 PyTorch Environment Verification Started")
    print("🔍 Kiểm tra môi trường PyTorch và GPU bắt đầu...\n")

    # ------------------------------------------------------------
    # Step 1 – Configure TF32 (TensorFloat32) precision
    # Bật TF32 để tăng tốc FP32 trên Tensor Cores (nếu GPU hỗ trợ)
    # ------------------------------------------------------------
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✅ TF32 mode enabled for both cuBLAS (GEMM) and cuDNN (Conv).")
        print("   Đã bật chế độ TF32 cho cả GEMM và Convolution.\n")
    except Exception as e:
        print("⚠️  Could not fully enable TF32:", e, "\n")

    # ------------------------------------------------------------
    # Step 2 – Print environment summary
    # In thông tin tóm tắt môi trường
    # ------------------------------------------------------------
    print_header("ENVIRONMENT INFO", "THÔNG TIN MÔI TRƯỜNG")
    print("Python Version     :", platform.python_version())
    print("PyTorch Version    :", torch.__version__)
    print("CUDA Toolkit       :", torch.version.cuda)
    print("CUDA Available     :", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("❌ No CUDA device found! / Không phát hiện GPU CUDA.")
        return

    print("GPU Device         :", torch.cuda.get_device_name(0))
    print("Compute Capability :", torch.cuda.get_device_capability(0))
    print("CUDA Arch List     :", os.environ.get("TORCH_CUDA_ARCH_LIST", "(not set)"))
    print("cuDNN Version      :", torch.backends.cudnn.version())
    print("NCCL Available     :", torch.distributed.is_nccl_available())

    # ------------------------------------------------------------
    # Step 3 – TF32 status check
    # Kiểm tra trạng thái TF32 hiện tại
    # ------------------------------------------------------------
    print_header("TF32 SETTINGS", "CÀI ĐẶT TF32")
    print("cuDNN Conv FP32 Precision :", torch.backends.cudnn.conv.fp32_precision)
    print("cuBLAS Matmul allow_tf32  :", torch.backends.cuda.matmul.allow_tf32)

    # ------------------------------------------------------------
    # Step 4 – Check channels_last memory format
    # Kiểm tra định dạng bộ nhớ channels_last
    # ------------------------------------------------------------
    print_header("CHANNELS_LAST FORMAT CHECK", "KIỂM TRA ĐỊNH DẠNG BỘ NHỚ channels_last")
    x = torch.randn(2, 3, 224, 224)
    print("Default stride (NCHW):", x.stride())
    x_cl = x.to(memory_format=torch.channels_last)
    print("Channels_last stride:", x_cl.stride())

    # ------------------------------------------------------------
    # Step 5 – GEMM benchmark (matrix multiply)
    # Kiểm tra tốc độ nhân ma trận (đơn vị: TFLOP/s)
    # ------------------------------------------------------------
    print_header("GEMM BENCHMARK", "KIỂM TRA HIỆU NĂNG GEMM (NHÂN MA TRẬN)")
    device = "cuda"
    sizes = [(4096,4096,4096)]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]

    for (M,N,K) in sizes:
        for dt in dtypes:
            try:
                a = torch.randn(M,K, device=device, dtype=dt)
                b = torch.randn(K,N, device=device, dtype=dt)
                for _ in range(3):  # warmup
                    with amp.autocast('cuda', dtype=dt):
                        (a @ b).relu_()
                    torch.cuda.synchronize()
                torch.cuda.synchronize(); t0 = time.time()
                with amp.autocast('cuda', dtype=dt):
                    c = a @ b
                torch.cuda.synchronize(); dt_ms = (time.time() - t0) * 1000
                tflops = (2*M*N*K) / ((dt_ms/1000)*1e12)
                print(f"{str(dt).split('.')[-1]:>9}  {M}x{K} @ {K}x{N}  time={dt_ms:7.2f} ms  ~{tflops:6.2f} TFLOP/s")
            except Exception as e:
                print(f"❌ {str(dt).split('.')[-1]} ERROR:", e)

    # ------------------------------------------------------------
    # Step 6 – Convolution benchmark (like CNN layer)
    # Kiểm tra tốc độ convolution 3x3 (như trong mạng CNN)
    # ------------------------------------------------------------
    print_header("CONV2D BENCHMARK", "KIỂM TRA HIỆU NĂNG CONVOLUTION 2D")
    for dt in [torch.float16, torch.bfloat16, torch.float32]:
        try:
            conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device)
            x = torch.randn(32, 64, 112, 112, device=device).to(memory_format=torch.channels_last)
            for _ in range(5):  # warmup
                with amp.autocast('cuda', dtype=dt):
                    y = conv(x)
                torch.cuda.synchronize()
            torch.cuda.synchronize(); t0 = time.time()
            with amp.autocast('cuda', dtype=dt):
                y = conv(x)
            torch.cuda.synchronize(); dt_ms = (time.time() - t0) * 1000
            print(f"{str(dt).split('.')[-1]:>9}  Conv(64→128, 3x3)  time={dt_ms:6.2f} ms")
        except Exception as e:
            print(f"❌ {str(dt).split('.')[-1]} ERROR:", e)

    # ------------------------------------------------------------
    # Step 7 – torch.compile functional test
    # Kiểm tra tính năng biên dịch động của PyTorch
    # ------------------------------------------------------------
    print_header("torch.compile TEST", "KIỂM TRA torch.compile (Biên dịch động)")
    try:
        m = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        ).cuda()
        m_compiled = torch.compile(m)
        x = torch.randn(8,3,224,224, device="cuda").to(memory_format=torch.channels_last)
        with amp.autocast('cuda', dtype=torch.float16):
            y = m_compiled(x)
        torch.cuda.synchronize()
        print("✅ torch.compile works correctly (OK)")
        print("   torch.compile hoạt động bình thường.")
    except Exception as e:
        print("⚠️ torch.compile skipped or failed:", e)

    # ------------------------------------------------------------
    # Step 8 – Summary
    # ------------------------------------------------------------
    print_header("SUMMARY", "TỔNG KẾT")
    print("✅ If all sections show OK or reasonable speeds -> Environment is READY.")
    print("   Nếu các phần kiểm tra đều OK hoặc tốc độ hợp lý → Môi trường đã sẵn sàng cho AI training.\n")
    print("🏁 Verification completed successfully.")
    print("   Quá trình kiểm tra đã hoàn tất thành công.")

if __name__ == "__main__":
    main()
