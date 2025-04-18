# run.py

import os
import sys
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    """主函数，处理命令行参数并启动应用"""
    parser = argparse.ArgumentParser(description="SmartNote AI - 私人知识问答库")
    
    parser.add_argument("--port", type=int, default=8501, 
                      help="运行应用的端口号")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                      help="主机地址")
    parser.add_argument("--debug", action="store_true", 
                      help="启用调试模式")
    
    args = parser.parse_args()
    
    # 启动Streamlit应用
    os.system(f"streamlit run app/app.py --server.port {args.port} --server.address {args.host}")

if __name__ == "__main__":
    main()