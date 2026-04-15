conda activate unipercept

docker compose --profile gpu up --build

bash ./src//eval/conversation.sh

# 8bit 开关（docker-compose.yml 默认已开启）
# 回滚到 bf16：
# LOAD_IN_8BIT=false LOAD_IN_4BIT=false docker compose --profile gpu up --build

# 2026-04-15 验证记录（8bit）
# - /health: inference_profile.load_in_8bit=true
# - 空载显存: 9162 MiB（基线: 15664 MiB）
# - 单图非流式推理耗时: 13.844s
# - 输出内容正常（美学/质量/结构评价可读）
