# Telegram 私聊防广告 Userbot（多模态 LLM + 辅助通知 Bot）

基于 **Python + Telethon（单进程双客户端）** 与 **OpenAI 兼容多模态大模型** 的 Telegram 私聊防广告助手：自动对陌生人来意进行防抖聚合、多维度账号画像与多模态识图审核，礼貌询问来意，对高置信度广告自动拉黑并双向删除记录，对可疑对话静音归档并推送交互卡片由你一键裁决。

## 核心特性

- **双客户端零打扰架构**：Userbot 负责监听与静默处置，独立 Helper Bot 负责向你推送带 Inline 按钮的审核卡片与留档快照，不污染主号 `Saved Messages`。
- **自动白名单与防呆**：通讯录好友、历史已有双向聊天的对象、官方系统号自动放行；只要你在私聊中手动回复，立即触发 **机主接管（Human Takeover）** 永久加白并退出干预。
- **审核期自动静音归档**：陌生人进入审核期间自动静音并移入 Archive；审核通过后自动移出归档、取消静音并标记未读提醒你。
- **消息防抖聚合 + 追问上限**：合并陌生人连续分条发送的消息（默认 5 秒窗口，上限 15 秒）；自动礼貌询问上限默认 2 轮，彻底防止与广告机器人死循环互聊。
- **多维度 + 多模态识图审核**：采集昵称、Bio、账号标记、共同群组、转发来源；支持图片缩放压缩识图、视频/GIF 仅提取内置缩略图识图、可执行高危文件（`.apk`/`.exe` 等）禁止自动放行；内置 `<untrusted_user_data>` 防 Prompt 注入隔离。
- **三级安全运行模式**：支持 `/mode dryrun`（演习观察）、`/mode review_only`（半自动追问 + 全人工确认，**默认初始模式**）、`/mode auto`（全自动分级处置）。
- **风控与预算保护**：拟人化 `typing` 随机延迟、每分钟回复/拉黑滑窗限流、Telegram `FloodWaitError` 捕获、每日 LLM 调用上限自动熔断。

---

## 快速开始

### 1. 准备凭证

1. **Telegram API 凭证**：前往 [https://my.telegram.org](https://my.telegram.org) 获取 `TG_API_ID` 和 `TG_API_HASH`。
2. **Helper Bot Token**：在 Telegram 找到 `@BotFather`，发送 `/newbot` 创建一个辅助通知机器人，拿到 `BOT_TOKEN`，并**主动给这个机器人发一条 `/start`**（否则机器人无法向你发消息）。
3. **你的数字 ID（`OWNER_ID`）**：可通过 `@userinfobot` 或 `@RawDataBot` 获取。
4. **LLM API Key**：任意 OpenAI 兼容接口（OpenAI / DeepSeek / OpenRouter / Gemini OpenAI 兼容端点等）。

### 2. 复制配置文件

```bash
cp .env.example .env
cp config.example.yaml config.yaml
```

编辑 `.env` 填入凭证，并根据需要在 `config.yaml` 中调整 `llm.owner_context`（你的个人背景与放行/拦截偏好）。

### 3. Docker Compose 部署（推荐）

**首次运行必须先交互式登录一次主号**以生成持久化 Session（存入 `./data/userbot.session`）：

```bash
docker compose run --rm antispam --login-only
```

按提示输入手机号、Telegram 验证码与两步验证密码（如有）。登录成功后后台启动：

```bash
docker compose up -d
```

查看日志：

```bash
docker compose logs -f
```

> 若在无交互平台部署，可先在本地运行 `python -m antispam --login-only --export-session` 获取 `TG_SESSION_STRING` 写入 `.env`。

### 4. 本地 Python 运行（uv）

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# 首次登录
uv run antispam --login-only

# 启动服务
uv run antispam
```

---

## 辅助 Bot 常用指令

仅 `OWNER_ID` 可使用以下指令：

| 指令 | 说明 |
| :--- | :--- |
| `/help` | 查看帮助与当前运行模式 |
| `/status` | 查看今日拦截/放行/追问统计、待审人数与 LLM 额度消耗 |
| `/mode dryrun\|review_only\|auto` | 实时切换并持久化运行模式 |
| `/allow <user_id\|@username>` | 手动放行/解封并加白（自动移出归档与取消静音） |
| `/block <user_id\|@username>` | 手动拉黑并双向删除记录 |
| `/pending` | 查看待人工确认列表并直接展示操作按钮 |
| `/rules list` | 查看本地前置规则 |
| `/rules add <keyword\|regex> <spam\|review> <内容>` | 热新增本地规则 |
| `/rules del <id>` | 删除本地规则 |

---

## 安全建议

- 建议刚部署时保持默认的 `mode: "review_only"` 运行 1~2 天，观察 Helper Bot 卡片的判断准确率，按实际场景微调 `config.yaml` 中的 `owner_context` 和阈值，再通过 `/mode auto` 切到全自动分级拦截。
- 切勿泄露 `.env`、`data/userbot.session` 或 `TG_SESSION_STRING`。

## 运行测试

```bash
uv run pytest
```
