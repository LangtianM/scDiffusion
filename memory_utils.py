#!/usr/bin/env python3
"""
内存监控和优化工具
用于监控和优化scDiffusion训练过程中的内存使用
"""

import gc
import psutil
import torch
import time
from typing import Optional
import logging

class MemoryMonitor:
    """内存监控类，用于跟踪和优化内存使用"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.max_memory_used = 0
        self.logger = logging.getLogger(__name__)
        
    def log_memory_usage(self, step: Optional[int] = None, force: bool = False):
        """记录当前内存使用情况"""
        if step is not None:
            self.step_count = step
        
        if self.step_count % self.log_interval == 0 or force:
            # CPU内存使用
            process = psutil.Process()
            cpu_memory_mb = process.memory_info().rss / 1024 / 1024
            
            # GPU内存使用
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
                
                self.logger.info(f"Step {self.step_count}: "
                               f"CPU: {cpu_memory_mb:.1f}MB, "
                               f"GPU: {gpu_memory_mb:.1f}MB, "
                               f"GPU Cached: {gpu_cached_mb:.1f}MB")
                
                # 跟踪最大内存使用
                current_total = cpu_memory_mb + gpu_memory_mb
                if current_total > self.max_memory_used:
                    self.max_memory_used = current_total
                    
            else:
                self.logger.info(f"Step {self.step_count}: CPU: {cpu_memory_mb:.1f}MB")
                if cpu_memory_mb > self.max_memory_used:
                    self.max_memory_used = cpu_memory_mb
    
    def cleanup_memory(self, aggressive: bool = False):
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if aggressive:
            # 更积极的内存清理
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    obj.detach_()
        
        gc.collect()
        
        self.logger.info("Memory cleanup completed")
    
    def get_memory_summary(self):
        """获取内存使用摘要"""
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        summary = {
            'current_cpu_mb': cpu_memory_mb,
            'max_memory_used_mb': self.max_memory_used,
            'step_count': self.step_count
        }
        
        if torch.cuda.is_available():
            summary.update({
                'current_gpu_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            })
            
        return summary

def optimize_pytorch_memory():
    """优化PyTorch内存设置"""
    # 设置PyTorch内存优化选项
    if torch.cuda.is_available():
        # 减少内存碎片
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # 设置内存分配策略
        torch.cuda.empty_cache()
        
    # 设置垃圾回收阈值
    gc.set_threshold(700, 10, 10)
    
def memory_efficient_training_wrapper(training_function):
    """训练函数的内存优化装饰器"""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor(log_interval=100)
        optimize_pytorch_memory()
        
        try:
            result = training_function(*args, **kwargs)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                monitor.logger.error(f"内存不足错误: {e}")
                monitor.logger.info("尝试清理内存...")
                monitor.cleanup_memory(aggressive=True)
                
                # 提供内存使用建议
                summary = monitor.get_memory_summary()
                monitor.logger.info(f"内存使用摘要: {summary}")
                monitor.logger.info("建议:")
                monitor.logger.info("1. 减少batch_size")
                monitor.logger.info("2. 增加microbatch参数")
                monitor.logger.info("3. 使用更小的模型")
                raise
        finally:
            final_summary = monitor.get_memory_summary()
            monitor.logger.info(f"训练完成，最终内存摘要: {final_summary}")
            
    return wrapper

# 使用示例:
# from memory_utils import memory_efficient_training_wrapper, MemoryMonitor
# 
# @memory_efficient_training_wrapper  
# def your_training_function():
#     # 你的训练代码
#     pass