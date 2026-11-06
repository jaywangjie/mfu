#!/usr/bin/env python
# encoding: utf-8
"""
从训练日志解析参数，用公式计算模型FLOPs并计算训练吞吐。
FLOPs用TFLOPS，参数用B表示。
"""
import argparse
import glob
import re
import sys
import os
import json
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calc_model_flops import calculate_text_flops, calculate_vision_flops, calculate_params, load_config, normalize_config


def parse():
    parser = argparse.ArgumentParser(description="从训练日志解析参数并计算训练FLOPS")
    parser.add_argument("--log_path", type=str, nargs='+',
                        help="训练日志文件路径，支持多个(多机场景: 可同时传入"
                             "各节点日志，参数与迭代时间会分别从对应日志提取)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径，如未指定则从日志中解析model_name_or_path")
    parser.add_argument("--seq_length", type=int, default=None,
                        help="序列长度，如未指定则从日志中解析cutoff_len")
    parser.add_argument("--world_size", type=int, default=None,
                        help="并行规模，如未指定则从日志中提取")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="FLOPs计算时使用的batch size (默认1)")
    parser.add_argument("--num_image_tokens", type=int, default=256,
                        help="VL模型图像token数(默认256)")
    parser.add_argument("--warmup_iters", type=int, default=1,
                        help="预热迭代数，这些迭代将被排除(默认1)")
    parser.add_argument("--bp_factor", type=float, default=3.0,
                        help="反向传播系数，前向1+反向2=3 (默认3.0)")
    parser.add_argument("--chip_power", type=float, default=None,
                        help="芯片算力(TFLOPS)，如未指定则用ascend-dmi自动查询")
    return parser.parse_args()


def parse_log(log_paths, warmup_iters):
    """解析训练日志，返回迭代记录列表

    支持多机场景：log_paths 可为多个日志文件路径。会合并所有日志中的
    迭代记录并按 iteration 去重(首次出现为准)，适用于参数和迭代时间
    分布在不同节点日志中的情况(如 node0 打印参数、last node 打印迭代时间)。
    """
    if isinstance(log_paths, str):
        log_paths = [log_paths]
    records = []
    seen_iters = set()
    pattern = re.compile(
        r'iteration\s+(\d+)/\s*\d+\s*\|.*?'
        r'elapsed time per iteration \(ms\):\s*([\d.]+).*?'
        r'global batch size:\s*(\d+)'
    )
    for file_path in log_paths:
        if not os.path.isfile(file_path):
            print("警告: 日志文件不存在，跳过: {}".format(file_path), file=sys.stderr)
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    iteration = int(match.group(1))
                    if iteration in seen_iters:
                        continue
                    seen_iters.add(iteration)
                    records.append({
                        'iteration': iteration,
                        'elapsed_time_ms': float(match.group(2)),
                        'global_batch_size': int(match.group(3)),
                    })
    if not records:
        print("错误: 未从任何日志中解析到迭代记录", file=sys.stderr)
        sys.exit(1)
    records.sort(key=lambda r: r['iteration'])
    valid = [r for r in records if r['iteration'] > warmup_iters]
    if len(valid) <= 1:
        print("错误: 有效迭代数不足2步，无法去掉首尾", file=sys.stderr)
        sys.exit(1)
    # 去掉最后一步
    valid = valid[:-1]
    return valid


def extract_from_logs(log_paths, patterns):
    """在多个日志文件中依次搜索，返回 (首个匹配值, 来源文件路径)

    patterns 可为单个正则字符串或正则列表；按列表顺序依次尝试，返回
    首个命中结果。兼容冒号分隔(`param: value`)与点号对齐
    (`param ...... value`, 如 MindSpeed-LLM 参数块)两种日志格式。
    参数名用负向回顾 `(?<!\\w)` 锚定，避免误匹配形如
    `log_world_size_to_tensorboard`、`decoder_seq_length` 的子串。
    """
    if isinstance(log_paths, str):
        log_paths = [log_paths]
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        compiled = re.compile(pattern)
        for file_path in log_paths:
            if not os.path.isfile(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = compiled.search(line)
                    if match:
                        return match.group(1), file_path
    return None, None


def extract_model_path(log_paths):
    # 优先 MindSpeed-MM WARNING 行中的真实模型路径(参数块中
    # tokenizer_name_or_path 常为 None), 其次 tokenizer_name_or_path
    # (HF模型目录, 含config.json), 再次 torchrun 命令行的
    # --tokenizer-name-or-path, 最后兜底旧的 model_name_or_path
    # (冒号或点号对齐)。过滤值为 "None" 的占位。
    patterns = [
        r'model_name_or_path\]\s*(\S+)',
        r'(?<!\w)tokenizer_name_or_path\s+\.+\s*(\S+)',
        r'tokenizer-name-or-path\s+(\S+)',
        r'(?<!\w)model_name_or_path\s+\.+\s*(\S+)',
        r'(?<!\w)model_name_or_path:\s*(\S+)',
    ]
    val, src = extract_from_logs(log_paths, patterns)
    if val and val != "None":
        return val.rstrip("/'\""), src
    return None, None


def extract_seq_length(log_paths):
    # MindSpeed-LLM 参数块用 seq_length(点号对齐), 兼容旧的
    # cutoff_len(冒号或点号)与 torchrun 命令行的 --seq-length。
    patterns = [
        r'(?<!\w)seq_length\s+\.+\s*(\d+)',
        r'(?<!\w)cutoff_len\s+\.+\s*(\d+)',
        r'(?<!\w)cutoff_len:\s*(\d+)',
        r'--seq-length\s+(\d+)',
    ]
    val, src = extract_from_logs(log_paths, patterns)
    return (int(val) if val else None), src


def extract_world_size(log_paths):
    # 点号对齐(MindSpeed-LLM)优先, 兼容冒号格式; `(?<!\w)` 避免命中
    # `log_world_size_to_tensorboard`。
    patterns = [
        r'(?<!\w)world_size\s+\.+\s*(\d+)',
        r'(?<!\w)world_size:\s*(\d+)',
    ]
    val, src = extract_from_logs(log_paths, patterns)
    return (int(val) if val else None), src


def parse_instance_name():
    """从环境变量 MA_CURRENT_INSTANCE_NAME 解析当前节点信息

    实例名格式: ma-job-{uuid}-vj-{node_rank}-worker-{worker_rank}
    返回 (node_rank, instance_name)；未设置或无法解析时返回 (None, None)
    """
    name = os.environ.get('MA_CURRENT_INSTANCE_NAME', '')
    if not name:
        return None, None
    m = re.search(r'-vj-(\d+)-worker-', name)
    if m:
        return int(m.group(1)), name
    return None, None


def discover_node_logs(log_path):
    """根据当前日志路径自动发现所有节点的日志

    依次尝试两种发现策略:
    1. ModelArts 训练环境: 解析环境变量 MA_CURRENT_INSTANCE_NAME 得到
       node_rank, 将路径中 vj-N 替换为 vj-* 进行 glob 搜索。
    2. 通用兜底: 将路径中 worker-N 段替换为 worker-* 进行 glob 搜索,
       适用于日志文件名含 worker-N 的场景(如 MindSpeed-LLM 多机,
       node0 日志只含参数、迭代时间打印在 last node 日志)。
    返回按 worker_rank 排序的 [(worker_rank, log_path), ...] 列表。
    无法发现多节点日志时回退为单条记录。
    """
    node_rank, _ = parse_instance_name()

    def _collect(pattern):
        matched = sorted(glob.glob(pattern))
        if not matched:
            return []
        node_logs = []
        for path in matched:
            m = re.search(r'worker-(\d+)', path)
            rank = int(m.group(1)) if m else 0
            node_logs.append((rank, path))
        node_logs.sort(key=lambda x: x[0])
        return node_logs

    # 策略1: ModelArts 环境, 路径含 vj-N
    if node_rank is not None and re.search(r'vj-\d+', log_path):
        result = _collect(re.sub(r'vj-\d+', 'vj-*', log_path))
        if result:
            return result

    # 策略2: 路径含 worker-N, 替换为 worker-* 发现兄弟日志
    if re.search(r'worker-\d+', log_path):
        result = _collect(re.sub(r'worker-\d+', 'worker-*', log_path))
        if len(result) > 1:
            return result

    if os.path.isfile(log_path):
        return [(0, log_path)]
    return []


def get_chip_compute_power():
    """使用 ascend-dmi 查询芯片bf16算力

    返回单卡算力(TFLOPS)。如果 device_id 包含 "/"（如"0/1"），
    表示结果为多卡合计，除以卡数得到单卡算力。
    """
    try:
        cmd = ["ascend-dmi", "-f", "-t", "bf16", "-d", "0", "--fmt", "json", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print("警告: ascend-dmi查询失败: {}".format(result.stderr.strip()), file=sys.stderr)
            return None
        data = json.loads(result.stdout)
        cp = data.get("computing_power", {})
        device_id = cp.get("device_id", "")
        tflops = cp.get("tflops_bf16", None)
        if tflops is None:
            print("警告: 无法从ascend-dmi输出中解析算力", file=sys.stderr)
            return None
        num_devices = len(device_id.split("/"))
        if num_devices > 1:
            tflops = tflops / num_devices
        return tflops
    except FileNotFoundError:
        print("警告: ascend-dmi命令不可用，无法自动查询芯片算力", file=sys.stderr)
        return None
    except Exception as e:
        print("警告: 查询芯片算力失败: {}".format(e), file=sys.stderr)
        return None


def calc_model_flops_formula(model_path, batch_size, seq_len, num_image_tokens=256):
    """用公式计算模型前向FLOPs

    对多模态模型同时计入视觉编码器FLOPs。返回 (flops, params)
    """
    cfg = load_config(model_path)
    text_cfg, vision_cfg = normalize_config(cfg)
    text_flops, lm_head = calculate_text_flops(text_cfg, batch_size, seq_len)
    if vision_cfg:
        vision_flops = calculate_vision_flops(vision_cfg, batch_size, num_image_tokens)
        print("  含视觉编码器FLOPs: text {:.2f} + vision {:.2f} = {:.2f} TFLOPs".format(
            text_flops / 1e12, vision_flops / 1e12, (text_flops + vision_flops) / 1e12))
        total_flops = text_flops + vision_flops
    else:
        total_flops = text_flops
    total_params = calculate_params(text_cfg, vision_cfg)
    return total_flops, total_params


def calculate_training_flops(records, model_flops, params,
                             seq_length, world_size, batch_size, bp_factor,
                             chip_power=None, is_multi_node=False):
    """计算训练FLOPS和吞吐"""
    times = [r['elapsed_time_ms'] for r in records]
    gbs = records[0]['global_batch_size']

    avg_step_time_ms = np.mean(times)
    avg_step_time_s = avg_step_time_ms / 1000.0

    flops_per_sample = model_flops / batch_size
    train_flops_per_iter = flops_per_sample * bp_factor * gbs
    per_chip_train_flops = train_flops_per_iter / world_size

    total_tflops = train_flops_per_iter / avg_step_time_s / 1e12
    per_chip_tflops = per_chip_train_flops / avg_step_time_s / 1e12

    tokens_per_iter = gbs * seq_length
    tps_per_chip = tokens_per_iter / avg_step_time_s / world_size

    params_b = params / 1e9
    flops_tflops = model_flops / 1e12

    mfu = None
    if chip_power is not None and chip_power > 0:
        mfu = per_chip_tflops / chip_power * 100

    scenario = "多机" if is_multi_node else "单机"
    print("=" * 70)
    print("训练FLOPS计算结果 (基于公式计算)")
    print("=" * 70)
    print("训练场景:              {}".format(scenario))
    print("模型参数量:            {:.2f} B".format(params_b))
    print("前向FLOPs/sample:      {:.2f} TFLOPs".format(flops_tflops / batch_size))
    print("序列长度(seq_length):  {}".format(seq_length))
    print("全局批大小(GBS):       {}".format(gbs))
    print("并行规模(world_size):  {}".format(world_size))
    print("反向传播系数(bp_factor): {}".format(bp_factor))
    print("有效迭代数:            {} (已去掉第一步和最后一步)".format(len(records)))
    print("平均迭代时间:          {:.2f} ms ({:.4f} s)".format(avg_step_time_ms, avg_step_time_s))
    print("每迭代tokens数:        {}".format(format(tokens_per_iter, ',')))
    print("-" * 70)
    print("每迭代训练FLOPs:       {:.2f} TFLOPs".format(train_flops_per_iter / 1e12))
    print("tokens/sec/chip:       {:.2f}".format(tps_per_chip))
    print("集群总:                {:.2f} TFLOPS/s".format(total_tflops))
    print("单卡:                  {:.2f} TFLOPS/s".format(per_chip_tflops))
    if chip_power is not None and chip_power > 0:
        print("芯片算力(bf16):        {:.2f} TFLOPS".format(chip_power))
        print("MFU:                   {:.2f}%".format(mfu))
    print("=" * 70)


if __name__ == '__main__':
    args = parse()

    log_paths = args.log_path
    is_multi_node = False

    # ===== 单机/多机场景自动检测 =====
    if len(log_paths) == 1:
        node_logs = discover_node_logs(log_paths[0])
        if len(node_logs) > 1:
            is_multi_node = True
            log_paths = [path for _, path in node_logs]
            print("=" * 70)
            print("检测到多机场景，共 {} 个节点".format(len(node_logs)))
            for rank, path in node_logs:
                print("  worker{}: {}".format(rank, path))
            print("  -> 参数(model_path/cutoff_len/world_size)从所有日志中搜索")
            print("  -> 迭代时间从所有日志中合并(按iteration去重)")
            print("=" * 70)
        else:
            print("=" * 70)
            print("检测到单机场景")
            print("  日志: {}".format(log_paths[0]))
            print("=" * 70)
    else:
        is_multi_node = True
        print("=" * 70)
        print("多机场景，已传入 {} 个日志文件".format(len(log_paths)))
        for p in log_paths:
            print("  {}".format(p))
        print("=" * 70)

    # ===== 参数提取(多机时优先从node0日志中找到) =====
    model_path = args.model_path
    if model_path is None:
        model_path, src = extract_model_path(log_paths)
        if model_path is None:
            print("错误: 无法从日志中解析model_name_or_path，请通过--model_path指定", file=sys.stderr)
            sys.exit(1)
        print("  -> model_name_or_path 来源于: {}".format(src))
    if not model_path.endswith('/'):
        model_path = model_path + '/'
    print("模型路径:              {}".format(model_path))

    seq_length = args.seq_length
    if seq_length is None:
        seq_length, src = extract_seq_length(log_paths)
        if seq_length is None:
            print("错误: 无法从日志中解析seq_length，请通过--seq_length指定", file=sys.stderr)
            sys.exit(1)
        print("  -> seq_length 来源于: {}".format(src))
    print("序列长度(seq_length):  {}".format(seq_length))

    world_size = args.world_size
    if world_size is None:
        world_size, src = extract_world_size(log_paths)
        if world_size is None:
            print("错误: 无法从日志中提取world_size，请通过--world_size指定", file=sys.stderr)
            sys.exit(1)
        print("  -> world_size 来源于: {}".format(src))
    print("并行规模(world_size):  {}".format(world_size))

    # ===== 迭代记录提取(多机时从所有节点日志合并) =====
    records = parse_log(log_paths, args.warmup_iters)
    gbs = records[0]['global_batch_size']

    print("\n正在用公式计算模型FLOPs (gbs={}, seq_length={}) ...".format(gbs, seq_length))
    model_flops, params = calc_model_flops_formula(model_path, args.batch_size, seq_length, args.num_image_tokens)

    chip_power = args.chip_power
    if chip_power is None:
        chip_power = get_chip_compute_power()

    print("\n")
    calculate_training_flops(records, model_flops, params,
                             seq_length, world_size, args.batch_size, args.bp_factor,
                             chip_power=chip_power, is_multi_node=is_multi_node)
