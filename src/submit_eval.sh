#!/bin/bash
#PBS -N Run_RAG
#PBS -l walltime=36:00:00
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=2
#PBS -j oe
#PBS -o job_log.out

# 1. 切换到提交任务时所在的当前工作目录
cd $PBS_O_WORKDIR

# 2. 激活 Python/Conda 虚拟环境
source /scratch/e1538713/miniconda3/bin/activate
conda activate /scratch/e1538713/envs/langgraph

# 3. 记录开始时间
echo "=================================================="
echo "任务开始时间: $(date)"
START_TIME=$(date +%s)
echo "运行节点: $(hostname)"
echo "工作目录: $PBS_O_WORKDIR"
echo "=================================================="

# 4. 执行 Python 批量测试脚本
python batch_evaluate.py

# 获取 Python 脚本的退出状态码，判断是否成功
EXIT_CODE=$?

# 5. 记录结束时间并计算总耗时
echo "=================================================="
echo "任务结束时间: $(date)"
END_TIME=$(date +%s)

# 计算差值（秒）并转换为 小时:分钟:秒 格式
TOTAL_SECONDS=$((END_TIME - START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

echo "任务总共运行时间: ${HOURS} 小时 ${MINUTES} 分钟 ${SECONDS} 秒"

if [ $EXIT_CODE -eq 0 ]; then
    echo "状态: 任务成功完成！"
else
    echo "状态: 任务由于错误退出 (退出码: $EXIT_CODE)，请检查日志。"
fi
echo "=================================================="