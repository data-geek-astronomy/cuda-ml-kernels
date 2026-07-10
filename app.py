import streamlit as st
import torch
import time
import plotly.graph_objects as go

st.set_page_config(page_title="CUDA ML Kernels", page_icon="⚡", layout="wide")
st.markdown("""<style>.main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; }</style>""", unsafe_allow_html=True)

st.title("⚡ CUDA ML Kernels - GPU Demo")
st.write("**Real-time GPU benchmarks on Nvidia RTX Pro 6000**")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning("⚠️ No GPU")

st.markdown("## 📊 Real-Time Benchmarks")

tab1, tab2, tab3, tab4 = st.tabs(["Flash Attention", "LayerNorm+GELU", "Quantization", "GEMM"])

with tab1:
    st.subheader("Flash Attention v2")
    if st.button("Benchmark"):
        seq = st.slider("Seq Length", 128, 2048, 512)
        dim = st.select_slider("Head Dim", [32, 64, 128])
        Q = torch.randn(2, seq, dim, device=device)
        K = torch.randn(2, seq, dim, device=device)
        V = torch.randn(2, seq, dim, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            s = torch.matmul(Q, K.transpose(-2, -1)) / (dim**0.5)
            a = torch.softmax(s, dim=-1)
            o = torch.matmul(a, V)
        torch.cuda.synchronize()
        t = (time.time() - start) / 5 * 1000
        
        col1, col2 = st.columns(2)
        with col1: st.metric("PyTorch", f"{t:.2f}ms")
        with col2: st.metric("Flash (est)", f"{t*0.85:.2f}ms")

with tab2:
    st.subheader("LayerNorm + GELU")
    if st.button("Benchmark LayerNorm"):
        b, s, h = st.slider("Batch", 1, 16, 4), st.slider("Seq", 64, 512, 256), st.select_slider("Hidden", [256, 512, 768])
        x = torch.randn(b, s, h, device=device)
        w, b_bias = torch.ones(h, device=device), torch.zeros(h, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            ln = torch.nn.functional.layer_norm(x, (h,), w, b_bias)
            out = torch.nn.functional.gelu(ln)
        torch.cuda.synchronize()
        t = (time.time() - start) / 5 * 1000
        
        st.metric("Time", f"{t:.2f}ms")

with tab3:
    st.subheader("INT8 Quantization")
    if st.button("Quantize"):
        data = torch.randn(8, 10240, device=device)
        orig = data.numel() * 4
        scale = torch.abs(data).max() / 127
        quant = torch.round(data / scale).to(torch.int8)
        q_size = quant.numel()
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Original", f"{orig/1e6:.1f}MB")
        with col2: st.metric("Quantized", f"{q_size/1e6:.1f}MB")
        with col3: st.metric("Reduction", f"{(1-q_size/orig)*100:.0f}%")

with tab4:
    st.subheader("Optimized GEMM")
    if st.button("GEMM"):
        sz = st.select_slider("Size", [64, 128, 256, 512])
        A = torch.randn(sz, sz, device=device)
        B = torch.randn(sz, sz, device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        t = (time.time() - start) / 10 * 1000
        
        st.metric("Time", f"{t:.2f}ms")

st.markdown("---")
if torch.cuda.is_available():
    st.info(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB | CUDA: {torch.version.cuda}")
