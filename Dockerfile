# 使用轻量级且兼容性好的 Python 3.10 镜像
FROM python:3.10-slim

# 标明维护者信息（可选）
LABEL maintainer="VASP-AI-Diagnoser"

# 设置工作目录
WORKDIR /app

# 安装必要的系统依赖 
# gcc 和 g++ 是安装 pymatgen 底层依赖(如 spglib)时必需的编译工具
# curl 用于后续容器的健康检查
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 将依赖文件复制到容器中
COPY requirements.txt .

# 升级 pip 并安装 Python 依赖（使用清华源加速国内下载，若在海外部署可删去 -i 参数）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 将我们的核心应用代码复制进容器
COPY app.py .

# 暴露 Streamlit 的默认端口
EXPOSE 8501

# 设置 Streamlit 的运行环境变量
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# 禁用在容器内部自动打开浏览器
ENV STREAMLIT_SERVER_HEADLESS=true 
# 解决可能出现的跨域/内网穿透白屏问题
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# 配置容器健康检查 (Streamlit 的标准检查路径)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动应用程序
CMD ["streamlit", "run", "app.py"]
