#!/usr/bin/env python3
"""训练监控 Web 服务器 - 实时显示 CPU、GPU、训练进度等信息。

启动方式:
    conda run -n internnav python scripts/train/monitor_server.py

浏览器访问: http://localhost:5000
"""
import os
import json
import time
import re
import glob
import psutil
import subprocess
from pathlib import Path
from flask import Flask, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

# 配置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints"


def get_gpu_info():
    """获取 GPU 使用信息"""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,'
             'memory.used,memory.total,power.draw,power.limit,fan.speed',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 9:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'temp': _safe_float(parts[2]),
                    'util': _safe_float(parts[3]),
                    'mem_util': _safe_float(parts[4]),
                    'mem_used': _safe_float(parts[5]),
                    'mem_total': _safe_float(parts[6]),
                    'power': _safe_float(parts[7]),
                    'power_limit': _safe_float(parts[8]),
                    'fan': _safe_float(parts[9]) if len(parts) > 9 else 0,
                })
        return gpus
    except Exception as e:
        return [{'error': str(e)}]


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def get_gpu_processes():
    """获取 GPU 上运行的进程"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return []
        procs = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                procs.append({
                    'pid': int(parts[0]),
                    'gpu_mem': _safe_float(parts[1]),
                    'name': parts[2]
                })
        return procs
    except Exception:
        return []


def get_cpu_info():
    """获取 CPU 使用信息"""
    return {
        'percent': psutil.cpu_percent(interval=0.5),
        'count': psutil.cpu_count(),
        'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
    }


def get_memory_info():
    """获取内存使用信息"""
    mem = psutil.virtual_memory()
    return {
        'total': round(mem.total / (1024**3), 2),
        'used': round(mem.used / (1024**3), 2),
        'percent': mem.percent,
    }


def get_disk_info():
    """获取磁盘使用信息"""
    try:
        usage = psutil.disk_usage(str(PROJECT_ROOT))
        return {
            'total': round(usage.total / (1024**3), 1),
            'used': round(usage.used / (1024**3), 1),
            'percent': usage.percent,
        }
    except:
        return {'total': 0, 'used': 0, 'percent': 0}


def find_training_processes():
    """查找正在运行的训练进程（仅匹配 python 训练进程）"""
    processes = []
    my_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if proc.info['pid'] == my_pid:
                continue
            cmdline = proc.info.get('cmdline', [])
            if not cmdline:
                continue
            cmdline_str = ' '.join(cmdline)
            # 只匹配 python 执行 train.py 的进程，排除 grep/curl/sh 等
            is_python = proc.info['name'] in ('python', 'python3') or 'python' in cmdline[0]
            has_train = 'train.py' in cmdline_str
            if is_python and has_train:
                uptime = time.time() - proc.info.get('create_time', time.time())
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cpu': proc.info['cpu_percent'],
                    'mem': round(proc.info['memory_info'].rss / (1024**3), 2) if proc.info['memory_info'] else 0,
                    'uptime': _fmt_duration(uptime),
                    'cmdline': ' '.join(cmdline[-3:]) if len(cmdline) > 3 else ' '.join(cmdline),
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes


def _fmt_duration(seconds):
    """将秒数格式化为人可读的时间"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def get_training_status(exp_name='bridgedp_train'):
    """获取训练状态"""
    exp_dir = CHECKPOINT_BASE / exp_name
    if not exp_dir.exists():
        return {'status': 'not_started', 'message': f'实验目录不存在: {exp_dir}'}

    # 读取训练日志
    log_file = exp_dir / 'logs' / 'train.log'
    log_lines = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_lines = f.readlines()[-50:]
        except Exception as e:
            log_lines = [f'读取日志失败: {e}']

    # 读取 trainer_state.json（搜索所有可能的位置）
    state_files = list((exp_dir / 'ckpts').glob('checkpoint-*/trainer_state.json'))
    if not state_files:
        state_files = list(exp_dir.glob('**/trainer_state.json'))

    latest_state = None
    if state_files:
        state_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        try:
            with open(state_files[0], 'r') as f:
                latest_state = json.load(f)
        except:
            pass

    # 统计 checkpoint 数量
    ckpt_dirs = sorted((exp_dir / 'ckpts').glob('checkpoint-*'))

    # 解析 TensorBoard 事件文件获取 loss 历史
    tb_dir = exp_dir / 'tensorboard'
    loss_history = _parse_tensorboard_summary(tb_dir)

    # 读取详细训练状态（由 DetailedProgressCallback 写入）
    detailed_status = None
    status_file = exp_dir / 'logs' / 'training_status.json'
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                detailed_status = json.load(f)
        except Exception:
            pass

    return {
        'status': 'running' if find_training_processes() else 'stopped',
        'exp_dir': str(exp_dir),
        'log_lines': log_lines,
        'checkpoints': len(ckpt_dirs),
        'latest_checkpoint': ckpt_dirs[-1].name if ckpt_dirs else None,
        'trainer_state': latest_state,
        'loss_history': loss_history,
        'detailed': detailed_status,
    }


def _parse_tensorboard_summary(tb_dir):
    """尝试解析 TensorBoard 事件文件"""
    if not tb_dir.exists():
        return []

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        event_files = list(tb_dir.glob('events.out.tfevents.*'))
        if not event_files:
            return []

        ea = EventAccumulator(str(tb_dir))
        ea.Reload()

        loss_data = []
        tags = ea.Tags().get('scalars', [])
        loss_tag = None
        for t in tags:
            if 'loss' in t.lower():
                loss_tag = t
                break

        if loss_tag:
            for e in ea.Scalars(loss_tag):
                loss_data.append({
                    'step': e.step,
                    'value': e.value,
                    'time': e.wall_time,
                })
        return loss_data[-200:]  # 最后200个数据点
    except Exception:
        return []


def _parse_config_from_source(config_path: Path) -> dict:
    """用 ast 解析配置源文件，提取 IlCfg(...) 中的关键字参数，无需导入任何依赖。"""
    import ast

    result = {}
    try:
        source = config_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
    except Exception:
        return result

    for node in ast.walk(tree):
        # 查找 IlCfg(...) 调用
        if isinstance(node, ast.Call):
            func_name = ''
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name != 'IlCfg':
                continue
            # 提取所有关键字参数
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                try:
                    value = ast.literal_eval(kw.value)
                    result[kw.arg] = value
                except (ValueError, TypeError):
                    pass  # 跳过无法静态求值的参数（如变量引用）

        # 同时从 ExpCfg(...) 提取 model_name
        if isinstance(node, ast.Call):
            func_name = ''
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == 'ExpCfg':
                for kw in node.keywords:
                    if kw.arg == 'model_name':
                        try:
                            result['model_name'] = ast.literal_eval(kw.value)
                        except (ValueError, TypeError):
                            pass

    return result


def get_training_config(exp_name='bridgedp_train'):
    """动态读取训练配置参数（通过 AST 解析配置源文件，无需导入依赖）"""
    config_file = PROJECT_ROOT / 'scripts' / 'train' / 'base_train' / 'configs' / 'bridgedp.py'

    # 默认值（作为 fallback）
    defaults = {
        'model_name': 'bridgedp',
        'epochs': 1000,
        'batch_size': 16,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'warmup_ratio': 0.05,
        'optimizer': 'adamw_torch',
        'lr_scheduler': 'cosine',
        'image_size': 224,
        'memory_size': 8,
        'predict_size': 24,
        'sigma_base': 1.0,
        'sigma_goal': 0.1,
        'n_prior_tokens': 4,
        'report_to': 'tensorboard',
        'num_workers': 4,
    }

    if config_file.exists():
        parsed = _parse_config_from_source(config_file)
        # 用解析到的真实值覆盖默认值
        for key in defaults:
            if key in parsed:
                defaults[key] = parsed[key]

    return defaults


@app.route('/')
def index():
    """主页面"""
    return render_template('monitor.html')


@app.route('/api/status')
def api_status():
    """API: 获取所有状态信息"""
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'gpu': get_gpu_info(),
        'gpu_procs': get_gpu_processes(),
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'disk': get_disk_info(),
        'processes': find_training_processes(),
        'training': get_training_status(),
        'config': get_training_config(),
    })


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Bridge-DP 训练监控服务器")
    print("=" * 50)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_BASE}")
    print(f"访问地址:   http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
