"""
SYSTEM PERFORMANCE CONFIGURATION
================================
Optimizations for Windows and Linux systems.

This module provides:
1. CPU/RAM detection and configuration
2. Process affinity and priority settings
3. Memory optimization settings
4. Recommended system tuning parameters
"""

import os
import sys
import platform
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import subprocess

# ============== SYSTEM DETECTION ==============

@dataclass
class SystemInfo:
    """System hardware information"""
    os_name: str
    cpu_count: int
    physical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    cpu_model: str
    has_avx2: bool
    has_gpu: bool
    gpu_name: Optional[str]


def detect_system() -> SystemInfo:
    """Detect system hardware configuration"""
    import multiprocessing as mp
    
    os_name = platform.system()
    cpu_count = mp.cpu_count()
    
    # Estimate physical cores (without hyperthreading)
    physical_cores = cpu_count // 2 if cpu_count > 1 else 1
    
    # RAM detection
    total_ram_gb = 16.0  # Default
    available_ram_gb = 8.0
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024**3)
        available_ram_gb = mem.available / (1024**3)
    except ImportError:
        pass
    
    # CPU model
    cpu_model = platform.processor() or "Unknown"
    
    # AVX2 support (for NumPy/Numba optimization)
    has_avx2 = False
    if os_name == "Linux":
        try:
            result = subprocess.run(['cat', '/proc/cpuinfo'], capture_output=True, text=True)
            has_avx2 = 'avx2' in result.stdout.lower()
        except:
            pass
    elif os_name == "Windows":
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            has_avx2 = 'avx2' in info.get('flags', [])
        except:
            pass
    
    # GPU detection
    has_gpu = False
    gpu_name = None
    
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    return SystemInfo(
        os_name=os_name,
        cpu_count=cpu_count,
        physical_cores=physical_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_model=cpu_model,
        has_avx2=has_avx2,
        has_gpu=has_gpu,
        gpu_name=gpu_name
    )


# ============== OPTIMAL CONFIGURATION ==============

def get_optimal_config(system: SystemInfo = None) -> Dict:
    """
    Get optimal configuration based on detected hardware.
    
    Returns configuration dict with recommended settings.
    """
    if system is None:
        system = detect_system()
    
    config = {
        # Parallelism
        'n_data_workers': max(1, system.physical_cores - 1),
        'n_feature_workers': max(1, system.physical_cores),
        'n_cv_workers': min(5, max(1, system.physical_cores // 2)),
        'lgbm_threads': system.cpu_count,
        
        # Memory
        'max_memory_gb': system.available_ram_gb * 0.8,
        'chunk_size': min(500_000, int(system.available_ram_gb * 50_000)),
        'use_float32': True,
        
        # LightGBM
        'lgbm_histogram_pool_size': min(2048, int(system.available_ram_gb * 128)),
        'lgbm_max_bin': 255,
        
        # GPU
        'use_gpu': system.has_gpu,
        'gpu_name': system.gpu_name,
        
        # Batch sizes
        'optimal_batch_size': _calculate_batch_size(system),
    }
    
    return config


def _calculate_batch_size(system: SystemInfo) -> int:
    """Calculate optimal batch size based on available RAM"""
    # Assume ~100 bytes per sample per feature
    # With 50 features, ~5KB per sample
    bytes_per_sample = 5000
    
    # Use 50% of available RAM for data
    available_bytes = system.available_ram_gb * 0.5 * (1024**3)
    
    batch_size = int(available_bytes / bytes_per_sample)
    
    # Cap at reasonable limits
    return min(max(10_000, batch_size), 10_000_000)


# ============== PROCESS OPTIMIZATION ==============

def set_process_priority(priority: str = 'high') -> bool:
    """
    Set current process priority.
    
    Args:
        priority: 'low', 'normal', 'high', 'realtime'
    """
    try:
        import psutil
        
        p = psutil.Process(os.getpid())
        
        if platform.system() == 'Windows':
            priority_map = {
                'low': psutil.IDLE_PRIORITY_CLASS,
                'normal': psutil.NORMAL_PRIORITY_CLASS,
                'high': psutil.HIGH_PRIORITY_CLASS,
                'realtime': psutil.REALTIME_PRIORITY_CLASS
            }
        else:  # Linux/Unix
            priority_map = {
                'low': 19,
                'normal': 0,
                'high': -10,
                'realtime': -20
            }
        
        if priority in priority_map:
            if platform.system() == 'Windows':
                p.nice(priority_map[priority])
            else:
                os.nice(priority_map[priority])
            return True
        
    except Exception as e:
        print(f"Could not set priority: {e}")
    
    return False


def set_cpu_affinity(cores: list = None) -> bool:
    """
    Set CPU affinity for current process.
    
    Args:
        cores: List of CPU cores to use, or None for all cores
    """
    try:
        import psutil
        
        p = psutil.Process(os.getpid())
        
        if cores is None:
            cores = list(range(psutil.cpu_count()))
        
        p.cpu_affinity(cores)
        return True
        
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")
    
    return False


# ============== ENVIRONMENT OPTIMIZATION ==============

def optimize_environment() -> Dict[str, str]:
    """
    Set environment variables for optimal performance.
    
    Returns dict of variables that were set.
    """
    env_vars = {}
    
    system = detect_system()
    
    # OpenMP settings (for NumPy, LightGBM)
    omp_threads = str(system.cpu_count)
    os.environ['OMP_NUM_THREADS'] = omp_threads
    os.environ['OMP_SCHEDULE'] = 'dynamic'
    os.environ['OMP_PROC_BIND'] = 'spread'
    env_vars['OMP_NUM_THREADS'] = omp_threads
    
    # MKL settings (Intel Math Kernel Library)
    os.environ['MKL_NUM_THREADS'] = omp_threads
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    env_vars['MKL_NUM_THREADS'] = omp_threads
    
    # OpenBLAS settings
    os.environ['OPENBLAS_NUM_THREADS'] = omp_threads
    env_vars['OPENBLAS_NUM_THREADS'] = omp_threads
    
    # Numba settings
    os.environ['NUMBA_NUM_THREADS'] = omp_threads
    os.environ['NUMBA_THREADING_LAYER'] = 'omp'
    env_vars['NUMBA_NUM_THREADS'] = omp_threads
    
    # Disable thread pinning that can cause slowdowns
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    # LightGBM settings
    os.environ['LGB_NUM_THREADS'] = omp_threads
    
    return env_vars


# ============== LINUX TUNING RECOMMENDATIONS ==============

LINUX_TUNING_GUIDE = """
# ============================================
# LINUX SYSTEM TUNING FOR AI TRAINING
# ============================================

# Run as root or with sudo

# 1. Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 2. Reduce swappiness (keep data in RAM)
echo "vm.swappiness=10" >> /etc/sysctl.conf
sysctl -p

# 3. Increase shared memory limits
echo "kernel.shmmax=68719476736" >> /etc/sysctl.conf
echo "kernel.shmall=4294967296" >> /etc/sysctl.conf
sysctl -p

# 4. I/O scheduler for SSDs
# Check current: cat /sys/block/sda/queue/scheduler
echo "none" > /sys/block/sda/queue/scheduler  # For NVMe
# Or
echo "mq-deadline" > /sys/block/sda/queue/scheduler  # For SATA SSD

# 5. Transparent Huge Pages (can help or hurt - test)
echo "always" > /sys/kernel/mm/transparent_hugepage/enabled

# 6. CPU governor for performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > $cpu
done

# 7. Disable CPU frequency scaling during training
cpupower frequency-set -g performance

# 8. Increase dirty page settings for better I/O
echo "vm.dirty_ratio=40" >> /etc/sysctl.conf
echo "vm.dirty_background_ratio=10" >> /etc/sysctl.conf
sysctl -p

# 9. Set process nice value for training script
# Run your training with: nice -n -19 python train_optimized.py

# 10. Use numactl for NUMA systems
# numactl --interleave=all python train_optimized.py
"""

WINDOWS_TUNING_GUIDE = """
# ============================================
# WINDOWS SYSTEM TUNING FOR AI TRAINING
# ============================================

# Run PowerShell as Administrator

# 1. Set power plan to High Performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 2. Disable Windows Search indexing on data drives
# Control Panel > Indexing Options > Modify > Uncheck data drives

# 3. Disable Windows Defender real-time scanning for data folder
# Add exclusion: Settings > Windows Security > Virus & threat protection
# > Manage settings > Add or remove exclusions

# 4. Disable Superfetch/SysMain for training
sc stop "SysMain"
sc config "SysMain" start=disabled

# 5. Increase process priority via Task Manager or:
# Start-Process python -ArgumentList "train_optimized.py" -Priority High

# 6. Set processor scheduling for background services
# Computer > Properties > Advanced > Performance Settings > Advanced
# > Processor scheduling: Background services

# 7. Disable unnecessary startup programs
# Task Manager > Startup tab

# 8. Virtual memory: Set fixed size (1.5x RAM) on SSD
# System Properties > Advanced > Performance > Virtual Memory

# 9. Disable USB selective suspend
# Power Options > Change plan settings > Change advanced power settings
# > USB settings > USB selective suspend setting: Disabled

# 10. Run training script with high priority:
# Start-Process -FilePath "python" -ArgumentList "train_optimized.py" -Priority High -NoNewWindow
"""


def print_tuning_guide():
    """Print system tuning guide based on OS"""
    system = detect_system()
    
    print("\n" + "="*60)
    print("SYSTEM TUNING RECOMMENDATIONS")
    print("="*60)
    
    print(f"\nDetected System:")
    print(f"  OS: {system.os_name}")
    print(f"  CPU: {system.cpu_model}")
    print(f"  Cores: {system.cpu_count} (physical: {system.physical_cores})")
    print(f"  RAM: {system.total_ram_gb:.1f}GB (available: {system.available_ram_gb:.1f}GB)")
    print(f"  AVX2: {system.has_avx2}")
    print(f"  GPU: {system.gpu_name if system.has_gpu else 'None'}")
    
    if system.os_name == "Linux":
        print(LINUX_TUNING_GUIDE)
    elif system.os_name == "Windows":
        print(WINDOWS_TUNING_GUIDE)
    
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATION")
    print("="*60)
    
    config = get_optimal_config(system)
    for key, value in config.items():
        print(f"  {key}: {value}")


# ============== REQUIREMENTS CHECK ==============

def check_dependencies() -> Dict[str, bool]:
    """Check if performance dependencies are installed"""
    deps = {}
    
    # Core
    deps['numpy'] = _check_import('numpy')
    deps['pandas'] = _check_import('pandas')
    deps['lightgbm'] = _check_import('lightgbm')
    
    # Performance
    deps['numba'] = _check_import('numba')
    deps['psutil'] = _check_import('psutil')
    deps['joblib'] = _check_import('joblib')
    
    # Optional
    deps['torch'] = _check_import('torch')
    deps['optuna'] = _check_import('optuna')
    
    print("\n" + "="*60)
    print("DEPENDENCY CHECK")
    print("="*60)
    
    for dep, installed in deps.items():
        status = "✓" if installed else "✗"
        print(f"  {status} {dep}")
    
    missing = [d for d, i in deps.items() if not i]
    if missing:
        print(f"\nMissing optional dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
    
    return deps


def _check_import(module: str) -> bool:
    """Check if a module can be imported"""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ============== MAIN ==============

if __name__ == "__main__":
    # Run full diagnostics
    check_dependencies()
    print_tuning_guide()
    
    # Apply optimizations
    print("\n" + "="*60)
    print("APPLYING OPTIMIZATIONS")
    print("="*60)
    
    env_vars = optimize_environment()
    for var, value in env_vars.items():
        print(f"  {var}={value}")
    
    set_process_priority('high')
    print("  Process priority: HIGH")
