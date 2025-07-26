#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
甄嬛传数据集下载脚本

从GitHub下载训练数据

使用方法:
    python scripts/download_data.py
"""

import requests
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys

class HuanHuanDataDownloader:
    """
    甄嬛传数据下载器
    """
    
    def __init__(self, data_dir: str = None):
        # 如果未指定数据目录，则使用项目根目录下的data目录
        if data_dir is None:
            # 获取当前脚本所在目录的父级目录作为项目根目录
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.raw_dir = self.data_dir / "raw"
        
        # 创建目录
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据源URL
        self.base_url = "https://raw.githubusercontent.com/datawhalechina/self-llm/master/dataset"
        
        # 数据文件
        self.data_file = "huanhuan.json"
    
    def download_file(self, url: str, save_path: Path, description: str = "") -> bool:
        """
        下载文件
        """
        try:
            logger.info(f"开始下载: {description or url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(save_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"下载完成: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"下载失败 {url}: {e}")
            return False
    
    def download_data(self) -> bool:
        """
        下载训练数据
        """
        logger.info("开始下载甄嬛传数据集...")
        
        url = f"{self.base_url}/{self.data_file}"
        save_path = self.raw_dir / self.data_file
        
        return self.download_file(url, save_path, "甄嬛传训练数据")

    def run(self) -> bool:
        """
        执行数据下载
        """
        if self.download_data():
            logger.info("🎉 数据下载完成！")
            logger.info(f"📁 数据保存在: {self.data_dir}")
            logger.info("📝 接下来可以运行: python training/huanhuan_data_prepare.py")
            return True
        else:
            logger.error("❌ 数据下载失败")
            return False
    
def main():
    """主函数"""
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="甄嬛传数据下载脚本")
    parser.add_argument(
        "--data-dir",
        help="数据存储目录 (默认: 项目根目录下的data目录)"
    )
    
    args = parser.parse_args()
    
    downloader = HuanHuanDataDownloader(data_dir=args.data_dir)
    
    if not downloader.run():
        sys.exit(1)

if __name__ == "__main__":
    main()