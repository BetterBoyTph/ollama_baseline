#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目一键运行脚本

该脚本按顺序执行以下操作：
1. 下载数据
2. 处理数据
3. 微调模型
4. 转换模型格式
5. 评估模型
6. 部署模型
7. 启动Web界面
8. 启动MCP服务器

适用于Linux Ubuntu系统，RTX 3090显卡，AutoDL服务器实例

使用说明:
========
1. 基本用法:
   python main_run.py

2. 跳过某些步骤:
   python main_run.py --skip-steps 1,2,3  # 跳过步骤1,2,3

3. 只运行特定步骤:
   python main_run.py --only-step 4      # 只运行步骤4

4. 配置化参数使用:
   python main_run.py --config config.yaml  # 使用自定义配置文件

5. 查看帮助:
   python main_run.py --help

配置文件说明:
===========
可以通过YAML配置文件自定义各个模块的参数，示例配置如下:

data_download:
  source_url: "https://raw.githubusercontent.com/datawhalechina/self-llm/master/dataset"
  data_file: "huanhuan.json"

data_process:
  max_samples: null  # null表示处理全部数据
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

model_training:
  config_file: "training/huanhuan_config_fast.yaml"
  base_model: "qwen2.5:0.5b"
  output_dir: "training/models/huanhuan_fast"

model_deployment:
  base_model: "qwen2.5:0.5b"
  modelfile: "deployment/Modelfile.huanhuan"
  model_name: "huanhuan_fast"

web_interface:
  port: 8501

mcp_server:
  port: 3141
"""

import os
import sys
import subprocess
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
    
# 导入typing模块的List类型用于类型注解
from typing import List

# 默认配置
DEFAULT_CONFIG = {
    "data_download": {
        "source_url": "https://raw.githubusercontent.com/datawhalechina/self-llm/master/dataset",
        "data_file": "huanhuan.json"
    },
    "data_process": {
        "max_samples": 50,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1
    },
    "model_training": {
        "config_file": "training/huanhuan_config_fast.yaml",
        "base_model": "qwen2.5:0.5b",
        "output_dir": "training/models/huanhuan_fast"
    },
    "model_deployment": {
        "base_model": "qwen2.5:0.5b",
        "modelfile": "deployment/Modelfile.huanhuan",
        "model_name": "huanhuan_fast"
    },
    "web_interface": {
        "port": 8501
    },
    "mcp_server": {
        "port": 3141
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path (str, optional): 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
            # 合并用户配置和默认配置
            for key, value in user_config.items():
                if key in config:
                    config[key].update(value)
                else:
                    config[key] = value
    
    return config

def run_command(command, description, cwd=None, timeout=None):
    """
    运行命令并显示输出
    
    Args:
        command (str): 要执行的命令
        description (str): 命令描述
        cwd (str): 工作目录，默认为None（当前目录）
        timeout (int): 超时时间（秒），默认为None
    
    Returns:
        bool: 命令执行是否成功
    """
    print(f"\n{'='*60}")
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    print(f"工作目录: {cwd or '当前目录'}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True, 
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout
        )
        print(result.stdout)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败")
        print(f"错误信息: {e}")
        print(f"输出: {e.output}")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ {description} 超时")
        return False
    except Exception as e:
        print(f"❌ 执行 {description} 时发生未知错误: {e}")
        return False

def check_prerequisites():
    """
    检查运行环境和前提条件
    """
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本必须 >= 3.8")
        return False
    
    # 检查CUDA是否可用
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    except ImportError:
        print("❌ 未安装PyTorch")
        return False
    
    # 检查Docker是否安装
    try:
        result = subprocess.run(
            "docker --version", 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ Docker已安装: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ Docker未安装，请先安装Docker")
        print("   Ubuntu安装命令: sudo apt install docker.io")
        return False
    except FileNotFoundError:
        print("❌ Docker未安装，请先安装Docker")
        print("   Ubuntu安装命令: sudo apt install docker.io")
        return False
    
    print("✅ 环境检查通过")
    return True

def setup_environment():
    """
    设置环境变量
    """
    print("🔧 设置环境变量...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU
    print("✅ 环境变量设置完成")

def step1_download_data(config: Dict[str, Any]):
    """
    步骤1: 下载数据
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n📥 步骤1: 下载数据")
    command = "python dataScripts/download_data.py"
    return run_command(command, "下载数据", cwd=project_root)

def step2_process_data(config: Dict[str, Any]):
    """
    步骤2: 处理数据
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n🧹 步骤2: 处理数据")
    # 构建命令参数
    max_samples = config["data_process"]["max_samples"]
    command = "python dataScripts/huanhuan_data_prepare.py"
    if max_samples is not None:
        command += f" {max_samples}"
    
    return run_command(command, "处理数据", cwd=project_root)

def step3_finetune_model(config: Dict[str, Any]):
    """
    步骤3: 微调模型
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n🤖 步骤3: 微调模型")
    # 使用配置文件进行训练
    config_file = config["model_training"]["config_file"]
    # 确保使用相对于项目根目录的路径
    config_file_path = project_root / config_file
    command = f"python training/huanhuan_train.py {config_file_path}"
    return run_command(command, "微调模型", cwd=project_root)

def step4_convert_model(config: Dict[str, Any]):
    """
    步骤4: 转换模型格式
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n🔄 步骤4: 转换模型格式")
    # 先检查是否有训练好的模型
    model_dir = project_root / config["model_training"]["output_dir"]
    if not model_dir.exists():
        print("❌ 未找到训练好的模型，请先完成模型训练")
        return False
    
    # 检查是否有适配器权重
    adapter_path = model_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        print("❌ 未找到适配器权重文件")
        return False
    
    # 转换LoRA权重为GGUF格式
    command = f"python llama.cpp/convert_lora_to_gguf.py --outfile deployment/huanhuan_fast_lora.gguf {model_dir}"
    return run_command(command, "转换模型格式", cwd=project_root)

def step5_evaluate_model(config: Dict[str, Any]):
    """
    步骤5: 评估模型
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n📊 步骤5: 评估模型")
    command = "python evaluate/example_usage.py"
    return run_command(command, "评估模型", cwd=project_root)

def step6_deploy_model(config: Dict[str, Any]):
    """
    步骤6: 部署模型 (使用vLLM)
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n🚀 步骤6: 部署模型 (使用vLLM)")
    if not config["model_deployment"]["enabled"]:
        print("⏭️  跳过模型部署步骤")
        return True
    
    # 检查模型文件是否存在
    model_path = Path(config["model_deployment"]["model_path"])
    if not model_path.exists():
        print("❌ 未找到训练好的模型文件")
        return False
    
    # 1. 拉取vLLM Docker镜像
    print("📥 拉取vLLM Docker镜像...")
    pull_command = "docker pull vllm/vllm-openai:latest"
    if not run_command(pull_command, "拉取vLLM Docker镜像"):
        print("❌ 拉取vLLM Docker镜像失败")
        return False
    
    # 2. 启动vLLM服务
    print("🔄 启动vLLM服务...")
    try:
        # 检查是否已有vLLM容器在运行
        result = subprocess.run(
            "docker ps | grep huanhuan-vllm", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            print("✅ vLLM服务已在运行")
        else:
            # 启动vLLM容器
            start_command = f"""docker run --gpus all \\
                --name huanhuan-vllm \\
                -v {project_root}/training/training/models/huanhuan_fast:/models/huanhuan_fast \\
                -p 8000:8000 \\
                --env NCCL_IGNORE_DISABLED_P2P=1 \\
                vllm/vllm-openai:latest \\
                --model Qwen/Qwen2.5-0.5B-Instruct \\
                --adapter /models/huanhuan_fast \\
                --host 0.0.0.0 \\
                --port 8000 \\
                --enable-lora \\
                --max-lora-rank 64 \\
                --tensor-parallel-size 1 \\
                --gpu-memory-utilization 0.8 \\
                --max-model-len 4096 \\
                --enforce-eager"""
            
            subprocess.Popen(
                start_command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("⏳ 等待vLLM服务启动...")
            time.sleep(15)  # 等待服务启动
            
            # 验证服务是否启动成功
            verify_command = "curl -f http://localhost:8000/v1/models"
            if run_command(verify_command, "验证vLLM服务"):
                print("✅ vLLM服务启动成功")
            else:
                print("⚠️  vLLM服务启动可能失败，请手动检查")
    except Exception as e:
        print(f"⚠️  启动vLLM服务时遇到问题: {e}")
        print("   将尝试继续执行后续步骤...")
    
    return True

def step7_start_web_interface(config: Dict[str, Any]):
    """
    步骤7: 启动Web界面
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n🌐 步骤7: 启动Web界面")
    port = config["web_interface"]["port"]
    print(f"💡 Web界面将在后台运行，访问地址: http://localhost:{port}")
    command = f"streamlit run application/huanhuan_web.py --server.port {port}"
    
    # 在后台运行Web界面
    try:
        subprocess.Popen(
            command,
            shell=True,
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ Web界面已在后台启动")
        return True
    except Exception as e:
        print(f"❌ 启动Web界面失败: {e}")
        return False

def step8_start_mcp_server(config: Dict[str, Any]):
    """
    步骤8: 启动MCP服务器
    
    Args:
        config (Dict[str, Any]): 配置字典
    """
    print("\n📡 步骤8: 启动MCP服务器")
    port = config["mcp_server"]["port"]
    print(f"💡 MCP服务器将在后台运行，端口: {port}")
    command = "python mcp_server/server.py"
    
    # 在后台运行MCP服务器
    try:
        subprocess.Popen(
            command,
            shell=True,
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ MCP服务器已在后台启动")
        return True
    except Exception as e:
        print(f"❌ 启动MCP服务器失败: {e}")
        return False

def main():
    """
    主函数：按顺序执行所有步骤
    """
    print("🎭 甄嬛传角色对话系统 - 一键运行脚本")
    print("适用于Linux Ubuntu系统，RTX 3090显卡，AutoDL服务器实例")
    print(f"项目根目录: {project_root}")
    
    parser = argparse.ArgumentParser(
        description="甄嬛传角色对话系统一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config", 
        default="fast_run/config.yaml",
        help='配置文件路径 (默认: fast_run/config.yaml)'
    )
    parser.add_argument(
        "--skip-steps", 
        type=str, 
        help="跳过的步骤编号，用逗号分隔，例如: 1,2,3"
    )
    parser.add_argument(
        "--only-steps", 
        type=str, 
        help="只执行指定步骤，例如: 6,7"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 解析跳过的步骤
    skip_steps = set()
    if args.skip_steps:
        try:
            skip_steps = {int(x.strip()) for x in args.skip_steps.split(",")}
        except ValueError:
            print("❌ --skip-steps 参数格式错误，请使用数字并用逗号分隔")
            return False
    # 解析只执行的步骤
    only_steps = []
    if args.only_steps:
        try:
            only_steps = [int(x.strip()) for x in args.only_steps.split(",")]
        except ValueError:
            print("❌ --only-steps 参数格式错误，请使用数字并用逗号分隔")
            return False
    
    # 解析只执行的步骤
    only_steps = []
    if args.only_steps:
        try:
            only_steps = [int(x) for x in args.only_steps.split(",")]
        except ValueError:
            print("❌ --only-steps 参数格式错误，请使用数字并用逗号分隔")
            return False
    
    # 检查前提条件
    if not check_prerequisites():
        print("❌ 环境检查失败，请检查依赖安装")
        return False
    
    # 设置环境
    setup_environment()
    
    # 定义所有步骤
    steps = [
        (1, "下载数据", step1_download_data),
        (2, "处理数据", step2_process_data),
        (3, "微调模型", step3_finetune_model),
        (4, "转换模型格式", step4_convert_model),
        (5, "评估模型", step5_evaluate_model),
        (6, "部署模型", step6_deploy_model),
        (7, "启动Web界面", step7_start_web_interface),
        (8, "启动MCP服务器", step8_start_mcp_server)
    ]
    
    # 确定要执行的步骤
    steps_to_run = []
    if only_steps:
        steps_to_run = [step for step in steps if step[0] in only_steps]
    else:
        steps_to_run = [step for step in steps if step[0] not in skip_steps]
    
    # 执行步骤
    start_time = time.time()
    success_count = 0
    
    for step_num, step_name, step_func in steps:
        # 判断是否应该执行此步骤
        if only_steps and step_num not in only_steps:
            print(f"\n⏭️  跳过步骤 {step_num}: {step_name}")
            success_count += 1
            continue
            
        if step_num in skip_steps:
            print(f"\n⏭️  跳过步骤 {step_num}: {step_name}")
            success_count += 1
            continue
        
        # 执行步骤
        if step_func(config):
            success_count += 1
        else:
            print(f"\n❌ 步骤 {step_num}: {step_name} 失败，停止执行后续步骤")
            break
    
    # 总结
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("📋 执行总结:")
    print(f"   成功步骤: {success_count}/{len(steps_to_run)}")
    print(f"   总耗时: {duration:.2f} 秒")
    
    if success_count == len(steps_to_run):
        web_port = config["web_interface"]["port"]
        mcp_port = config["mcp_server"]["port"]
        model_name = config["model_deployment"]["model_name"]
        
        print("🎉 所有步骤执行完成！")
        print("\n💡 使用说明:")
        print(f"   - Web界面地址: http://localhost:{web_port}")
        print(f"   - MCP服务器地址: http://localhost:{mcp_port}")
        print(f"   - vLLM服务地址: http://localhost:8000")
        print(f"   - 模型名称: {model_name}")
    else:
        print("⚠️  部分步骤执行失败，请检查日志")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()