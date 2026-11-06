#!/usr/bin/env python3
"""
Universal Model FLOPs Calculator
Supports: dense, MoE, multimodal (VL) models
Reads config.json from local model directory, no weights needed.

Usage:
    python calc_model_flops.py /path/to/model               # default batch=1, seq=128
    python calc_model_flops.py /path/to/model -b 1 -s 128   # custom
    python calc_model_flops.py /path/to/model --all          # scan all models in parent dir
"""

import argparse
import json
import os
import sys


# ============================================================================
# Config Normalization
# ============================================================================

def load_config(model_path):
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(cfg_path) as f:
        return json.load(f)


def normalize_config(cfg):
    """Extract text_config and vision_config from various config layouts."""
    # Standard nested: text_config at top level (qwen3_5_moe, internvl, etc.)
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        text_cfg = cfg["text_config"]
        vision_cfg = cfg.get("vision_config", None)
    # Qwen3-Omni: thinker_config.text_config + thinker_config.vision_config
    elif "thinker_config" in cfg and isinstance(cfg["thinker_config"], dict):
        thinker = cfg["thinker_config"]
        text_cfg = thinker.get("text_config", thinker)
        vision_cfg = thinker.get("vision_config", cfg.get("vision_config", None))
    else:
        text_cfg = cfg
        vision_cfg = cfg.get("vision_config", None)

    return text_cfg, vision_cfg


def get_arch_type(text_cfg):
    """Determine architecture type: dense, moe, or linear_moe."""
    has_moe = (
        "num_experts" in text_cfg
        or "n_routed_experts" in text_cfg
        or "num_local_experts" in text_cfg
    )
    has_linear_attn = "layer_types" in text_cfg

    if has_moe and has_linear_attn:
        return "linear_moe"
    elif has_moe:
        return "moe"
    elif has_linear_attn:
        return "linear_dense"
    else:
        return "dense"


# ============================================================================
# FLOPs Calculation Primitives
# ============================================================================

def linear_flops(in_features, out_features, tokens):
    """FLOPs for a linear layer: 2 * in * out * tokens."""
    return 2 * in_features * out_features * tokens


def mlp_flops(hidden_size, intermediate_size, tokens):
    """Standard SwiGLU MLP: gate_proj + up_proj + down_proj."""
    return linear_flops(hidden_size, intermediate_size, tokens) * 3


def attention_flops(hidden_size, num_heads, num_kv_heads, head_dim, seq_len, tokens):
    """Standard attention: QKV projection + attention scores + output projection."""
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # QKV projections
    qkv = linear_flops(hidden_size, q_dim, tokens) + 2 * linear_flops(hidden_size, kv_dim, tokens)

    # Attention core: QK^T + softmax@V
    # QK^T: [B, nh, S, hd] @ [B, nh, hd, S] -> [B, nh, S, S], macs = B*nh*S*hd*S
    # SV:   [B, nh, S, S] @ [B, nh, S, hd] -> [B, nh, S, hd], macs = B*nh*S*S*hd
    attn_core = 4 * num_heads * head_dim * seq_len * tokens

    # Output projection
    o_proj = linear_flops(q_dim, hidden_size, tokens)

    return qkv + attn_core + o_proj


def is_mla_attention(text_cfg):
    """Check if model uses MLA (Multi-head Latent Attention) like DeepSeek-V2/V3."""
    return "kv_lora_rank" in text_cfg and "qk_nope_head_dim" in text_cfg


def mla_attention_flops(text_cfg, seq_len, tokens):
    """MLA (Multi-head Latent Attention) FLOPs for DeepSeek-V2/V3/R1.

    MLA compresses KV into a low-rank latent space:
      - Query: optional low-rank (q_a_proj + q_b_proj) or direct q_proj
      - KV: kv_a_proj_with_mqa (down) + kv_b_proj (up)
      - Attention core uses q_head_dim and v_head_dim
    """
    H = text_cfg["hidden_size"]
    num_heads = text_cfg["num_attention_heads"]
    q_lora_rank = text_cfg.get("q_lora_rank", None)
    kv_lora_rank = text_cfg["kv_lora_rank"]
    qk_nope_head_dim = text_cfg["qk_nope_head_dim"]
    qk_rope_head_dim = text_cfg["qk_rope_head_dim"]
    v_head_dim = text_cfg["v_head_dim"]
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Query projection
    if q_lora_rank is None:
        # Direct: H -> num_heads * q_head_dim
        q_proj = linear_flops(H, num_heads * q_head_dim, tokens)
    else:
        # Low-rank: q_a_proj (H -> q_lora_rank) + q_b_proj (q_lora_rank -> num_heads * q_head_dim)
        q_proj = linear_flops(H, q_lora_rank, tokens) + linear_flops(q_lora_rank, num_heads * q_head_dim, tokens)

    # KV compression
    # kv_a_proj_with_mqa: H -> (kv_lora_rank + qk_rope_head_dim)
    kv_a = linear_flops(H, kv_lora_rank + qk_rope_head_dim, tokens)
    # kv_b_proj: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
    kv_b = linear_flops(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), tokens)
    kv_proj = kv_a + kv_b

    # Attention core: QK^T + softmax@V
    # QK^T: [B, nh, S, q_head_dim] @ [B, nh, q_head_dim, S] -> macs = B*nh*S*q_head_dim*S
    qk = 2 * num_heads * q_head_dim * seq_len * tokens
    # SV: [B, nh, S, S] @ [B, nh, S, v_head_dim] -> macs = B*nh*S*S*v_head_dim
    sv = 2 * num_heads * v_head_dim * seq_len * tokens
    attn_core = qk + sv

    # Output projection: num_heads * v_head_dim -> H
    o_proj = linear_flops(num_heads * v_head_dim, H, tokens)

    return q_proj + kv_proj + attn_core + o_proj


def linear_attention_flops(text_cfg, seq_len, tokens):
    """GatedDeltaNet-style linear attention (Qwen3.5+)."""
    H = text_cfg["hidden_size"]
    lin_key_dim = text_cfg.get("linear_key_head_dim", 128) * text_cfg.get("linear_num_key_heads", 16)
    lin_val_dim = text_cfg.get("linear_value_head_dim", 128) * text_cfg.get("linear_num_value_heads", 32)

    # in_proj_qkv: H -> (2*key_dim + val_dim)
    # in_proj_z: H -> val_dim
    # in_proj_b: H -> num_v_heads
    # in_proj_a: H -> num_v_heads
    num_v_heads = text_cfg.get("linear_num_value_heads", 32)
    proj_flops = (
        linear_flops(H, 2 * lin_key_dim + lin_val_dim, tokens)  # in_proj_qkv
        + linear_flops(H, lin_val_dim, tokens)                   # in_proj_z
        + linear_flops(H, num_v_heads, tokens)                   # in_proj_b
        + linear_flops(H, num_v_heads, tokens)                   # in_proj_a
        + linear_flops(lin_val_dim, H, tokens)                   # out_proj
    )

    # Core delta rule: O(n * d_k * d_v), NOT O(n^2 * d_k * d_v)
    # Each token does d_k * d_v matrix update, total = d_k * d_v * n
    core_flops = 2 * lin_key_dim * lin_val_dim * tokens

    # Conv1d: kernel_size * conv_dim per token
    conv_kernel = text_cfg.get("linear_conv_kernel_dim", 4)
    conv_dim = 2 * lin_key_dim + lin_val_dim
    conv_flops = conv_kernel * conv_dim * tokens

    return proj_flops + core_flops + conv_flops


# ============================================================================
# Layer-level FLOPs
# ============================================================================

def dense_layer_flops(text_cfg, seq_len, tokens, layer_type="full_attention"):
    """FLOPs for a single dense (non-MoE) decoder layer."""
    H = text_cfg["hidden_size"]
    I = text_cfg.get("intermediate_size", 4 * H)
    num_heads = text_cfg.get("num_attention_heads", 32)
    num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
    head_dim = text_cfg.get("head_dim", H // num_heads)

    # Attention: MLA, linear attention, or standard
    if layer_type == "linear_attention":
        attn = linear_attention_flops(text_cfg, seq_len, tokens)
    elif is_mla_attention(text_cfg):
        attn = mla_attention_flops(text_cfg, seq_len, tokens)
    else:
        attn = attention_flops(H, num_heads, num_kv_heads, head_dim, seq_len, tokens)

    # MLP
    mlp = mlp_flops(H, I, tokens)

    # RMSNorm (2 * H * tokens, negligible but included)
    norm = 4 * H * tokens  # input_layernorm + post_attention_layernorm

    return attn + mlp + norm


def moe_layer_flops(text_cfg, seq_len, tokens, layer_type="full_attention"):
    """FLOPs for a single MoE decoder layer."""
    H = text_cfg["hidden_size"]
    num_heads = text_cfg.get("num_attention_heads", 32)
    num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
    head_dim = text_cfg.get("head_dim", H // num_heads)

    # Attention: MLA, linear attention, or standard
    if layer_type == "linear_attention":
        attn = linear_attention_flops(text_cfg, seq_len, tokens)
    elif is_mla_attention(text_cfg):
        attn = mla_attention_flops(text_cfg, seq_len, tokens)
    else:
        attn = attention_flops(H, num_heads, num_kv_heads, head_dim, seq_len, tokens)

    # MoE block
    num_experts = text_cfg.get("num_experts", text_cfg.get("n_routed_experts", text_cfg.get("num_local_experts", 0)))
    num_experts_per_tok = text_cfg.get("num_experts_per_tok", 0)
    moe_inter = text_cfg.get("moe_intermediate_size", text_cfg.get("intermediate_size", 4 * H))

    # Shared expert: check shared_expert_intermediate_size (Qwen3) or n_shared_experts (DeepSeek)
    shared_inter = text_cfg.get("shared_expert_intermediate_size", None)
    n_shared = text_cfg.get("n_shared_experts", 0)

    # Router: H -> num_experts
    router = linear_flops(H, num_experts, tokens)

    # Selected experts: K experts, each is gate_proj + up_proj + down_proj
    expert = num_experts_per_tok * mlp_flops(H, moe_inter, tokens)

    # Shared expert(s)
    shared = 0
    if shared_inter is not None and shared_inter > 0:
        # Qwen3-style: single shared expert with its own intermediate size
        shared = mlp_flops(H, shared_inter, tokens)
        shared += linear_flops(H, 1, tokens)  # shared_expert_gate
    elif n_shared > 0:
        # DeepSeek-style: n_shared experts, each using moe_intermediate_size
        shared = n_shared * mlp_flops(H, moe_inter, tokens)

    # RMSNorm
    norm = 4 * H * tokens

    return attn + router + expert + shared + norm


# ============================================================================
# Model-level FLOPs
# ============================================================================

def calculate_text_flops(text_cfg, batch_size, seq_len):
    """Calculate FLOPs for the text/decoder part of the model."""
    L = text_cfg.get("num_hidden_layers", 0)
    H = text_cfg["hidden_size"]
    V = text_cfg.get("vocab_size", 0)
    tokens = batch_size * seq_len

    arch_type = get_arch_type(text_cfg)

    # Determine layer types
    layer_types = text_cfg.get("layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * L

    # DeepSeek-style: first_k_dense_replace layers are dense, rest are MoE
    first_k_dense = text_cfg.get("first_k_dense_replace", 0)
    moe_layer_freq = text_cfg.get("moe_layer_freq", 1)

    total_flops = 0
    breakdown = {"attention": 0, "mlp": 0, "moe_router": 0, "moe_experts": 0, "moe_shared": 0, "norm": 0}

    for i in range(L):
        lt = layer_types[i] if i < len(layer_types) else "full_attention"

        is_moe_layer = False
        if arch_type in ("moe", "linear_moe"):
            if i < first_k_dense:
                is_moe_layer = False
            elif moe_layer_freq > 1 and (i - first_k_dense) % moe_layer_freq != 0:
                is_moe_layer = False
            else:
                is_moe_layer = True

        if is_moe_layer:
            layer_flops = moe_layer_flops(text_cfg, seq_len, tokens, lt)
        else:
            layer_flops = dense_layer_flops(text_cfg, seq_len, tokens, lt)

        total_flops += layer_flops

    # LM Head
    lm_head = linear_flops(H, V, tokens) if V > 0 else 0
    total_flops += lm_head

    return total_flops, lm_head


def calculate_vision_flops(vision_cfg, batch_size, num_image_tokens=256):
    """Calculate FLOPs for vision encoder (ViT-based)."""
    if vision_cfg is None:
        return 0

    L = vision_cfg.get("num_hidden_layers", vision_cfg.get("depth", 0))
    H = vision_cfg.get("hidden_size", 1024)
    I = vision_cfg.get("intermediate_size", 4 * H)
    num_heads = vision_cfg.get("num_attention_heads", 16)
    head_dim = H // num_heads if num_heads > 0 else 64

    tokens = num_image_tokens  # vision tokens, not text seq_len

    total = 0
    for _ in range(L):
        attn = attention_flops(H, num_heads, num_heads, head_dim, tokens, tokens)
        mlp = mlp_flops(H, I, tokens)
        total += attn + mlp

    # Patch embedding (conv): patch_size^2 * 3 * H * num_patches
    patch_size = vision_cfg.get("patch_size", 14)
    if isinstance(patch_size, list):
        patch_size = patch_size[0]
    patch_embed = 2 * patch_size * patch_size * 3 * H * tokens
    total += patch_embed

    return total


# ============================================================================
# Params Calculation
# ============================================================================

def calculate_params(text_cfg, vision_cfg=None):
    """Estimate total parameters from config."""
    H = text_cfg["hidden_size"]
    L = text_cfg.get("num_hidden_layers", 0)
    V = text_cfg.get("vocab_size", 0)
    num_heads = text_cfg.get("num_attention_heads", 32)
    num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
    head_dim = text_cfg.get("head_dim", H // num_heads)
    I = text_cfg.get("intermediate_size", 4 * H)

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    # Attention params per layer (MLA vs standard)
    if is_mla_attention(text_cfg):
        q_lora_rank = text_cfg.get("q_lora_rank", None)
        kv_lora_rank = text_cfg["kv_lora_rank"]
        qk_nope = text_cfg["qk_nope_head_dim"]
        qk_rope = text_cfg["qk_rope_head_dim"]
        v_hdim = text_cfg["v_head_dim"]
        q_hdim = qk_nope + qk_rope
        if q_lora_rank is None:
            q_p = H * num_heads * q_hdim
        else:
            q_p = H * q_lora_rank + q_lora_rank * num_heads * q_hdim
        kv_p = H * (kv_lora_rank + qk_rope) + kv_lora_rank * num_heads * (qk_nope + v_hdim)
        o_p = num_heads * v_hdim * H
        attn_params = q_p + kv_p + o_p
    else:
        attn_params = H * q_dim + 2 * H * kv_dim + q_dim * H

    # MLP params
    mlp_params = 3 * H * I  # gate + up + down

    arch_type = get_arch_type(text_cfg)
    first_k_dense = text_cfg.get("first_k_dense_replace", 0)

    total = 0
    for i in range(L):
        is_moe = False
        if arch_type in ("moe", "linear_moe"):
            if i >= first_k_dense:
                is_moe = True

        if is_moe:
            num_experts = text_cfg.get("num_experts", text_cfg.get("n_routed_experts", text_cfg.get("num_local_experts", 0)))
            moe_inter = text_cfg.get("moe_intermediate_size", I)
            moe_mlp = 3 * H * moe_inter
            expert_params = num_experts * moe_mlp
            router_params = H * num_experts
            # Shared experts: Qwen3-style or DeepSeek-style
            shared_inter = text_cfg.get("shared_expert_intermediate_size", 0)
            n_shared = text_cfg.get("n_shared_experts", 0)
            if shared_inter:
                shared_params = 3 * H * shared_inter + H
            elif n_shared:
                shared_params = n_shared * 3 * H * moe_inter
            else:
                shared_params = 0
            total += attn_params + expert_params + router_params + shared_params
        else:
            total += attn_params + mlp_params

    # Embedding
    total += V * H

    # Vision
    if vision_cfg:
        vL = vision_cfg.get("num_hidden_layers", vision_cfg.get("depth", 0))
        vH = vision_cfg.get("hidden_size", 1024)
        vI = vision_cfg.get("intermediate_size", 4 * vH)
        v_heads = vision_cfg.get("num_attention_heads", 16)
        total += vL * (vH * vH + 2 * vH * vH + vH * vH + 3 * vH * vI)

    return total


# ============================================================================
# Main
# ============================================================================

def fmt_num(n):
    if n >= 1e12:
        return "{:.4e} ({:.2f} TFLOPs)".format(n, n / 1e12)
    elif n >= 1e9:
        return "{:.4e} ({:.2f} GFLOPs)".format(n, n / 1e9)
    elif n >= 1e6:
        return "{:.4e} ({:.2f} MFLOPs)".format(n, n / 1e6)
    else:
        return "{:.4e}".format(n)


def fmt_params(n):
    if n >= 1e9:
        return "{:.2f}B".format(n / 1e9)
    elif n >= 1e6:
        return "{:.2f}M".format(n / 1e6)
    else:
        return str(n)


def calculate_model_flops(model_path, batch_size=1, seq_len=128, num_image_tokens=256, verbose=True):
    cfg = load_config(model_path)
    text_cfg, vision_cfg = normalize_config(cfg)

    model_name = os.path.basename(os.path.normpath(model_path))
    arch_type = get_arch_type(text_cfg)
    has_vision = vision_cfg is not None

    text_flops, lm_head_flops = calculate_text_flops(text_cfg, batch_size, seq_len)
    vision_flops = calculate_vision_flops(vision_cfg, batch_size, num_image_tokens) if has_vision else 0
    total_flops = text_flops + vision_flops

    total_params = calculate_params(text_cfg, vision_cfg)

    if verbose:
        print("=" * 70)
        print("Model: {}".format(model_name))
        print("Path:  {}".format(model_path))
        print("Type:  {}{}".format(arch_type, " + vision" if has_vision else ""))
        print("Input: batch={}, seq_len={}, image_tokens={}".format(batch_size, seq_len, num_image_tokens if has_vision else 0))
        print("-" * 70)
        print("Text FLOPs:    {}".format(fmt_num(text_flops)))
        if has_vision:
            print("Vision FLOPs:  {}".format(fmt_num(vision_flops)))
        print("  (LM Head):    {}".format(fmt_num(lm_head_flops)))
        print("-" * 70)
        print("Total FLOPs:   {}".format(fmt_num(total_flops)))
        print("Total Params:  {} ({})".format(fmt_params(total_params), "{:.4e}".format(total_params)))
        print("=" * 70)

    return total_flops, total_params


def main():
    parser = argparse.ArgumentParser(description="Universal Model FLOPs Calculator")
    parser.add_argument("model_path", help="Path to model directory containing config.json")
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("-s", "--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--image-tokens", type=int, default=256, help="Number of image tokens for VL models (default: 256)")
    parser.add_argument("--all", action="store_true", help="Scan all models in parent directory")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if args.all:
        model_path_norm = os.path.normpath(args.model_path)
        if os.path.isfile(os.path.join(model_path_norm, "config.json")):
            parent_dir = os.path.dirname(model_path_norm)
        else:
            parent_dir = model_path_norm
        models = []
        for name in sorted(os.listdir(parent_dir)):
            mp = os.path.join(parent_dir, name)
            if os.path.isdir(mp) and os.path.isfile(os.path.join(mp, "config.json")):
                models.append(mp)

        if not models:
            print("No models with config.json found in {}".format(parent_dir))
            sys.exit(1)

        print("\nScanning {} models in {}\n".format(len(models), parent_dir))

        if args.quiet:
            print("{:<45} {:<12} {:<12} {:<15} {:<15}".format(
                "Model", "Type", "Params", "GFLOPs", "TFLOPs"))
            print("-" * 100)

        results = []
        for mp in models:
            try:
                flops, params = calculate_model_flops(mp, args.batch, args.seq_len, args.image_tokens, verbose=not args.quiet)
                results.append((os.path.basename(mp), flops, params))
                if args.quiet:
                    cfg = load_config(mp)
                    tc, _ = normalize_config(cfg)
                    at = get_arch_type(tc)
                    has_v = "V" if cfg.get("vision_config") else " "
                    print("{:<45} {:<11}V {:<12} {:<15.2f} {:<15.4f}".format(
                        os.path.basename(mp), at, fmt_params(params), flops/1e9, flops/1e12))
            except Exception as e:
                if args.quiet:
                    print("{:<45} {:<24}".format(os.path.basename(mp), "ERROR: {}".format(str(e)[:23])))
                else:
                    print("ERROR [{}]: {}".format(os.path.basename(mp), e))

        if args.quiet:
            print("-" * 100)
            print("Total: {} models".format(len(results)))
        else:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print("{:<40} {:>15} {:>15}".format("Model", "GFLOPs", "Params"))
            print("-" * 70)
            for name, flops, params in results:
                print("{:<40} {:>15.2f} {:>15}".format(name, flops/1e9, fmt_params(params)))
            print("-" * 70)
    else:
        calculate_model_flops(args.model_path, args.batch, args.seq_len, args.image_tokens)


if __name__ == "__main__":
    main()
