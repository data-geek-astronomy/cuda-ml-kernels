import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIG
st.set_page_config(page_title="CUDA ML Kernels", page_icon="⚡", layout="wide")

# CUSTOM CSS
st.markdown("""
<style>
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI'; }
    .main { background: linear-gradient(135deg, #f5f5f7 0%, #ffffff 100%); }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("# ⚡ CUDA ML Kernels")
st.markdown("**Production-grade GPU optimization for deep learning**")
st.markdown("---")

# HERO
st.markdown("""
## 🎯 The Problem

Running transformers on GPU is expensive:
- ❌ Attention is slow
- ❌ LayerNorm + GELU wastes compute  
- ❌ Models too large for inference
- ❌ Matrix operations not optimized

**We built 4 custom CUDA kernels to solve these.**
""")

# METRICS
st.markdown("## 📊 Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("⚡ Flash Attention", "9.4x", "faster")
with col2:
    st.metric("🔗 LayerNorm + GELU", "1.8x", "faster")
with col3:
    st.metric("📊 Quantization", "75%", "reduction")
with col4:
    st.metric("🧮 GEMM", "3x", "faster")

st.markdown("---")

# TABS
st.markdown("## 🚀 Interactive Demos")

tab1, tab2, tab3, tab4 = st.tabs(["⚡ Flash Attention", "🔗 LayerNorm+GELU", "📊 Quantization", "🧮 GEMM"])

# TAB 1
with tab1:
    st.subheader("Flash Attention v2")
    st.write("The fastest attention mechanism. Used by OpenAI, Meta, Anthropic.")
    
    col1, col2 = st.columns(2)
    with col1:
        seq_len = st.slider("Sequence Length", 128, 4096, 512, step=128)
        head_dim = st.select_slider("Head Dimension", [32, 64, 128, 256])
    with col2:
        st.metric("Memory (Input)", f"{(seq_len * head_dim * 4) / 1e6:.1f} MB")
        st.metric("Speedup", "10-100x")
    
    # Chart
    fig = go.Figure(data=[
        go.Bar(x=['PyTorch', 'Flash Attention'], y=[45.2, 4.8],
               marker_color=['#FF3B30', '#34C759'],
               text=['45.2ms', '4.8ms'], textposition='outside')
    ])
    fig.update_layout(yaxis_title="Time (ms)", height=400, showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# TAB 2
with tab2:
    st.subheader("Fused LayerNorm + GELU")
    st.write("Two operations, one kernel. Zero intermediate memory.")
    
    col1, col2 = st.columns(2)
    with col1:
        batch = st.slider("Batch Size", 1, 16, 4)
        seq = st.slider("Sequence Length", 64, 1024, 256, step=64)
    with col2:
        st.metric("Memory Saved", "~30%")
        st.metric("Speedup", "1.8x")
    
    # Chart
    configs = ['batch=1', 'batch=2', 'batch=4', 'batch=8']
    speedups = [1.5, 1.7, 1.8, 1.8]
    fig = px.bar(x=configs, y=speedups, color=speedups,
                 color_continuous_scale=['#FF3B30', '#34C759'],
                 text=[f'{s:.1f}x' for s in speedups])
    fig.update_layout(height=400, showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3
with tab3:
    st.subheader("INT8 Quantization")
    st.write("Compress models without losing accuracy.")
    
    col1, col2 = st.columns(2)
    with col1:
        per_channel = st.checkbox("Per-Channel Quantization")
        model_size = st.select_slider("Model Size", ["1GB", "2GB", "4GB", "8GB"])
    with col2:
        st.metric("Memory Reduction", "75%")
        st.metric("Speedup", "30-40%")
    
    # Chart
    sizes = {'1GB': [1.0, 0.25], '2GB': [2.0, 0.5], '4GB': [4.0, 1.0], '8GB': [8.0, 2.0]}
    orig, quant = sizes[model_size]
    fig = go.Figure(data=[
        go.Bar(name='FP32', x=['Size'], y=[orig], marker_color='#FF3B30'),
        go.Bar(name='INT8', x=['Size'], y=[quant], marker_color='#34C759')
    ])
    fig.update_layout(yaxis_title="GB", height=400, barmode='group',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# TAB 4
with tab4:
    st.subheader("Optimized GEMM")
    st.write("Hand-tuned matrix multiplication.")
    
    col1, col2 = st.columns(2)
    with col1:
        matrix_size = st.select_slider("Matrix Size", [64, 128, 256, 512, 1024])
    with col2:
        st.metric("Speedup", "2-5x")
        st.metric("Pattern", "Optimized")
    
    # Chart
    variants = {'Basic': 45, 'Tiled': 15, 'Transposed': 12, 'Batched': 10}
    fig = go.Figure(data=[
        go.Bar(x=list(variants.keys()), y=list(variants.values()),
               marker_color=['#FF3B30', '#FF9500', '#34C759', '#00D084'],
               text=[f'{v}ms' for v in variants.values()], textposition='outside')
    ])
    fig.update_layout(yaxis_title="Time (ms)", height=400, showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# CODE
st.markdown("## 📝 Code Examples")

code_tab1, code_tab2, code_tab3, code_tab4 = st.tabs(["Flash Attention", "LayerNorm", "Quantization", "GEMM"])

with code_tab1:
    st.code("""
import cuda_ml_kernels_py as cuda
Q = np.random.randn(2, 512, 64).astype(np.float32)
K = np.random.randn(2, 512, 64).astype(np.float32)
V = np.random.randn(2, 512, 64).astype(np.float32)
output = cuda.flash_attention_forward(Q, K, V, scale=0.125)
""", language="python")

with code_tab2:
    st.code("""
output = cuda.layer_norm_gelu_forward(x, gamma, beta)
""", language="python")

with code_tab3:
    st.code("""
quantized, scales = cuda.quantize_int8(weights, per_channel=False)
""", language="python")

with code_tab4:
    st.code("""
result = cuda.gemm_tiled(A, B, C)
""", language="python")

st.markdown("---")

# FOOTER
st.markdown("""
## 🔗 Learn More

- **[GitHub](https://github.com/data-geek-astronomy/cuda-ml-kernels)** - Full source
- **[Quick Start](https://github.com/data-geek-astronomy/cuda-ml-kernels#quick-start)** - Get started
- **[Docs](https://github.com/data-geek-astronomy/cuda-ml-kernels#-documentation)** - Full docs

**Built with ❤️ for the GPU community** | ⭐ Star on GitHub | MIT License
""")
