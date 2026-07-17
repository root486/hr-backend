# 智能招聘系统 (HR Backend)

基于 AI Agent 的全自动招聘后端，覆盖简历解析、候选人评分、面试邀约与协商全流程。

## 技术栈

| 层 | 技术 |
|---|---|
| Web 框架 | FastAPI |
| AI 框架 | LangChain / LangGraph |
| 大模型 | 通义千问（阿里云百炼 DashScope） |
| 数据库 | PostgreSQL |
| 缓存 | Redis |
| 邮件 | QQ 邮箱 SMTP/IMAP |
| 日程 | 钉钉开放平台 |
| OCR | PaddleOCR / QwenOcr |
| 任务调度 | APScheduler |

## 项目结构

```
hr-backend/
├── agents/           # AI Agent：简历解析、候选人评分、面试流程
│   ├── llms.py       # 大模型实例定义
│   ├── prompts.py    # System Prompt 模板
│   ├── resume.py     # 简历信息提取 Agent
│   └── candidate.py  # 面试流程 Agent + 工具函数
├── core/             # 基础设施：钉钉、邮件、OCR、缓存、鉴权
├── models/           # SQLAlchemy ORM 模型
├── routers/          # API 路由（候选人、职位、用户、仪表盘）
├── schemas/          # Pydantic 数据模型
├── repository/       # 数据库操作层
├── tasks/            # 后台任务（OCR 解析、Agent 调度）
├── scheduler/        # 定时任务（邮件轮询）
├── utils/            # 工具函数（时间处理、空闲时段计算）
├── alembic/          # 数据库迁移
├── settings/         # 配置管理
└── main.py           # 应用入口
```

## 大模型架构

| 变量名 | 模型 | 单价（每百万 token） | 角色 |
|---|---|---|---|
| `qwen_llm` | qwen3.5-flash | ¥0.2 | 主力（评分、简历解析） |
| `deepseek_llm` | qwen3.6-flash | ¥1.2 | Agent 主力 + 兜底 |

```
简历解析 Agent:      qwen3.5-flash (主力) → qwen3.6-flash (兜底)
评分工具:            qwen3.5-flash (主力) → qwen3.6-flash (兜底)
面试流程 Agent:      qwen3.6-flash (主力) → qwen3.6-flash (自身兜底)
```

## 核心流程

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1.简历上传   │ →  │  2.OCR解析   │ →  │  3.AI提取    │ →  │  4.创建候选  │
│  upload/     │    │  PaddleOCR   │    │  resume.py   │    │  + 启动Agent │
│  Word→PDF    │    │  ↓fallback   │    │  结构化信息   │    │              │
│              │    │  QwenOcr     │    │  (姓名/技能…) │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  8.流程结束   │ ←  │  7.确认面试   │ ←  │  6.邮件协商   │ ←  │  5.AI评分    │
│  状态更新     │    │  钉钉日程+邮件 │    │  15s轮询收件  │    │  5维度打分   │
│              │    │  +DB记录      │    │  候选人回复   │    │  >8分通过    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 详细步骤

#### 1. 简历上传 `POST /candidate/resume/upload`
- 支持 PDF、Word (.doc/.docx)、图片 (JPEG/PNG)
- Word 文档自动转换为 PDF
- 文件以 UUID 重命名存储到 `upload/` 目录

#### 2. OCR 解析（后台任务）
- 主力：**PaddleOCR** — 创建任务 → 轮询状态 → 拉取解析结果
- 兜底：**QwenOcr** — PaddleOCR 失败时自动切换
- 将图片/PDF 中的文字提取为纯文本

#### 3. AI 提取结构化信息 (`agents/resume.py`)
- 模型：qwen3.5-flash → qwen3.6-flash（兜底）
- 输入：OCR 后的原始文本
- 输出：结构化 JSON（姓名、性别、年龄、技能、教育经历、工作经历等）
- 结果存入 Redis，前端通过 `task_id` 轮询获取

#### 4. 创建候选人 `POST /candidate/create`
- 将解析后的候选人信息入库
- 关联职位和面试官
- **自动启动面试 Agent** 作为后台任务

#### 5. AI 评分 (`score_for_candidate` 工具)
- 模型：qwen3.5-flash → qwen3.6-flash（兜底）
- 5 个维度各 1-10 分：

| 维度 | 权重 |
|---|---|
| 工作经验匹配度 | 30% |
| 技术技能匹配度 | 30% |
| 项目经验匹配度 | 20% |
| 软技能潜力 | 10% |
| 教育背景 | 10% |

- 总分 > 8 → `AI_FILTER_PASSED`，进入面试邀约
- 总分 ≤ 8 → `AI_FILTER_FAILED`，流程终止

#### 6. 邮件协商（定时轮询）
- APScheduler 每 **15 秒** 检查收件箱（IMAP）
- 根据 `thread_id`（候选人邮箱）恢复 Agent 上下文，继续对话
- 发送面试邀请邮件 → 等待候选人回复

#### 7. 确认面试 (`confirm_interview_time` 工具)
- 发最终确认邮件给候选人
- 给面试官创建钉钉日程安排
- 数据库中创建面试记录
- 更新候选人状态为 `WAITING_FOR_INTERVIEW`

### 候选人状态流转

```
APPLICATION → AI_FILTER_PASSED/FAILED → WAITING_FOR_INTERVIEW
    → REFUSED_INTERVIEW / INTERVIEW_PASSED/REJECTED → HIRED/REJECTED
```

状态只能向前流转，不可回退。

## 快速开始

### 环境要求
- Python 3.12+
- PostgreSQL 15+
- Redis 7+

### 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd hr-backend

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key 和邮箱配置

# 5. 数据库迁移
alembic upgrade head

# 6. 初始化数据
python init_data.py

# 7. 启动服务
uvicorn main:app --reload
```

### 环境变量

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（通义千问） |
| `MAIL_USERNAME` | QQ 邮箱地址 |
| `MAIL_PASSWORD` | QQ 邮箱 SMTP 授权码 |
| `DINGTALK_APP_KEY` | 钉钉应用 AppKey |
| `DINGTALK_APP_SECRET` | 钉钉应用 AppSecret |
| `PADDLE_OCR_ACCESS_TOKEN` | PaddleOCR Access Token |

## Agent 工具一览

| 工具 | 功能 |
|---|---|
| `score_for_candidate` | 5 维度评分 + 存入数据库 + 更新状态 |
| `get_interviewer_available_slot` | 查询面试官钉钉日历空闲时段（7 天内） |
| `send_interview_email` | 发送面试时间协商邮件 |
| `confirm_interview_time` | 确认面试：发邮件 + 创建钉钉日程 + 写入 DB |
| `refuse_interview` | 候选人拒绝面试，更新状态 |
| `get_current_time` | 获取当前北京时间（年月日+星期几） |

## License

Private
