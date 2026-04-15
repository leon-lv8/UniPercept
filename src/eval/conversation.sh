#!/usr/bin/env bash
# 任意目录执行均可：bash /path/to/UniPercept/src/eval/conversation.sh

# ========== 方式 A：本地 PyTorch 直推（与 conversation.py 一致）==========
# 使用 --image 传入宿主机上的图片路径；模型内部会拼成 "<image>\\n" + 文本（见 conversation.py）
python src/eval/conversation.py \
    --model_path ckpt/unipercept \
    --image examples/test.jpg \
    --prompt "这张图片的构图如何？描述了什么内容？详细评价这张图片，并说出优缺点。需要详细的评价。IAA，IQA，ISTA的评分都是多少？"

# ========== 方式 B：HTTP /v1/chat/completions（OpenAI 多模态格式）==========
# 与方式 A 等价：把图片放在 messages[].content 数组里，type 为 image_url。
# - url 支持 data:image/...;base64,... 与 http(s)://
# - 容器内需可读路径时，可挂载 examples 后使用 ALLOW_FILE_IMAGE_URL=true 与 file:///workspace/examples/...（见 docker-compose）
#
# 取消下面整段注释并先启动服务（如 docker compose）即可试跑：
#
# API_BASE="${API_BASE:-http://127.0.0.1:8000}"
# python3 - <<'PY'
# import base64, json, os, urllib.request
#
# api = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
# img_path = os.path.join("examples", "test1.jpg")
# with open(img_path, "rb") as f:
#     b64 = base64.standard_b64encode(f.read()).decode("ascii")
# body = {
#     "model": "unipercept-vl",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
#                 {
#                     "type": "text",
#                     "text": "这张图片的构图如何？简要描述内容。",
#                 },
#             ],
#         }
#     ],
# }
# req = urllib.request.Request(
#     f"{api}/v1/chat/completions",
#     data=json.dumps(body).encode("utf-8"),
#     headers={"Content-Type": "application/json"},
#     method="POST",
# )
# with urllib.request.urlopen(req, timeout=600) as resp:
#     print(resp.read().decode("utf-8"))
# PY
