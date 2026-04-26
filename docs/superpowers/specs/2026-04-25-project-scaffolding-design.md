# SemanticVibe 專案骨架設計

**日期**：2026-04-25
**狀態**：Draft（待使用者審閱）
**對應規格書**：[SemanticVibe_Spec.docx](../../../SemanticVibe_Spec.docx) v1.0
**範圍**：定義專案的初始檔案結構、套件管理、模組邊界、共用 schema、雙 LLM provider 抽象，以及 Week 1 第一個可跑的 deliverable。本文件**不**涵蓋各 stage 的內部演算法細節（那是後續每個 stage 自己的 design doc）。

---

## 1. 設計目標

1. **配合 spec §8 的 6 週時程**——骨架要在 Week 1 前期建好，不能拖到 Week 2。
2. **保留 spec §4 的關鍵架構決策**——5 stages 與「不餵影片給 LLM」的成本最佳化策略。
3. **針對 Windows + RTX 3060 12GB 本機開發環境**最佳化（dev 環境已確認）。
4. **schemas 是系統最窄腰**，必須先於任何 stage 實作之前定義並進 git。
5. 後期（Week 5 Streamlit、Week 6 demo）不需要 refactor 才能展示。

## 2. 開發環境（已確認）

| 項目 | 決定 | 備註 |
|---|---|---|
| 主要平台 | Windows 11 本機 | Colab 僅在 SDXL 預生成卡關時備援 |
| GPU | RTX 3060 12GB（CUDA 已裝） | 可同時載入 Whisper + BLIP-2 + CLIP，且 SDXL 1024×1024 可跑 |
| Python | 3.10（鎖死） | MediaPipe 在 3.13 無 wheel；3.11 雖可用但 spec 寫 3.10 |
| 套件管理 | `uv` + `pyproject.toml` + `uv.lock` | 比 poetry 快 10–100×；原生支援 PyTorch CUDA 特殊 index |
| 環境變數 | `.env` + `python-dotenv`（透過 `pydantic-settings` 讀） | `.env` 進 .gitignore；`.env.example` 進 git |

**LLM API**：使用實驗室共用金鑰；雙 provider（Claude + GPT-4o）抽象介面，預設 dev 模式跑 Claude Haiku 4.5，prod 模式跑 Claude Sonnet 4.6。

## 3. 目錄結構

```
AI人文/
├── SemanticVibe_Spec.docx              # 既有規格書
├── CLAUDE.md                            # 既有
├── pyproject.toml                       # uv
├── uv.lock                              # 進 git
├── .python-version                      # 內容: 3.10
├── .env.example                         # 進 git
├── .env                                 # .gitignore
├── .gitignore
├── README.md
│
├── src/semanticvibe/
│   ├── __init__.py
│   ├── config.py                        # style presets, 模型名, COST_MODES
│   ├── pipeline.py                      # 頂層 orchestrator
│   │
│   ├── schemas/                         # 系統最窄腰
│   │   ├── __init__.py
│   │   ├── feature_summary.py           # Stage 1 → 2 的契約
│   │   └── decision.py                  # Stage 2 → 3-5 的契約
│   │
│   ├── preprocess/                      # Stage 1
│   │   ├── __init__.py
│   │   ├── whisper_asr.py
│   │   ├── librosa_beats.py
│   │   ├── mediapipe_pose.py
│   │   ├── blip2_caption.py
│   │   ├── keyframes.py
│   │   └── pipeline.py                  # extract_features(video) → FeatureSummary
│   │
│   ├── llm/                             # Stage 2
│   │   ├── __init__.py
│   │   ├── client.py                    # LLMClient Protocol + Claude/OpenAI 實作
│   │   ├── prompts.py                   # system prompt + few-shot
│   │   └── decide.py                    # 含 Pydantic 驗證 + tenacity retry
│   │
│   ├── assets/                          # Stage 3
│   │   ├── __init__.py
│   │   ├── library.py
│   │   └── clip_search.py
│   │
│   ├── layout/                          # Stage 4
│   │   ├── __init__.py
│   │   ├── occupancy.py
│   │   ├── placement.py
│   │   └── bin_packing.py
│   │
│   └── render/                          # Stage 5
│       ├── __init__.py
│       ├── text_render.py               # Pillow，不用 MoviePy TextClip
│       ├── animations.py                # bounce_in, typewriter, wiggle, draw_in, fade
│       └── composite.py                 # MoviePy + ffmpeg
│
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_render.py                   # Week 1 重點
│   └── ...
│
├── notebooks/
│   └── playground.ipynb                 # 只做實驗、視覺化；正式邏輯一律在 src/
│
├── examples/
│   └── hand_written_decision.json       # Week 1 手寫範例 JSON
│
├── data/                                # .gitignore
│   ├── README.md                        # 寫怎麼下載/重建素材
│   ├── assets_lib/
│   │   ├── metadata.json
│   │   └── *.png                        # 200–300 張
│   ├── fonts/
│   │   ├── KleeOne-Regular.ttf
│   │   ├── KleeOne-SemiBold.ttf
│   │   └── NotoSansTC-Regular.ttf       # CJK fallback
│   └── test_videos/
│       └── sample_30s.mp4
│
├── outputs/                             # .gitignore
│
├── app.py                               # Streamlit (Week 5 才實作)
└── docs/superpowers/specs/
    └── 2026-04-25-project-scaffolding-design.md   # 本檔
```

### 關鍵約定

- **每個模型一個檔**（`whisper_asr.py`、`mediapipe_pose.py`...）：日後換模型只動一檔，且各自 model loading / device 邏輯獨立。
- **`schemas/` 是 package**：避免 stage 間 circular import；分檔便於後續 schema 版本管理。
- **`data/` 整個排除 git**：素材有授權標示但不適合進 git；改用 `data/README.md` 描述重建步驟。
- **`notebooks/` 進 git，但用 `nbstripout` pre-commit hook 清 cell output**：避免 diff 噪音。

## 4. 套件相依（`pyproject.toml` 大綱）

```toml
[project]
name = "semanticvibe"
version = "0.1.0"
requires-python = ">=3.10,<3.11"
dependencies = [
    # Schemas + LLM
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "anthropic>=0.40",
    "openai>=1.40",
    "python-dotenv>=1.0",
    "tenacity>=8.2",            # LLM JSON parse 失敗重試

    # Preprocess
    "faster-whisper>=1.0",      # 比 openai-whisper 快 4-5×, Windows wheel OK
    "librosa>=0.10",
    "mediapipe>=0.10",
    "transformers>=4.40",       # BLIP-2
    "torch>=2.2",               # CUDA index 在 [tool.uv.sources]
    "opencv-python>=4.9",

    # Assets
    "open-clip-torch>=2.24",
    "pillow>=10.0",

    # Render
    "moviepy>=1.0.3",
    "imageio-ffmpeg>=0.4",      # bundle ffmpeg, 免裝系統 PATH

    # UI
    "streamlit>=1.30",
]

[project.optional-dependencies]
dev  = ["pytest>=8.0", "pytest-cov", "ruff", "nbstripout"]
sdxl = ["diffusers>=0.27", "accelerate", "rembg"]   # nice-to-have

[tool.uv]
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
```

**安裝**：`uv sync`（含 PyTorch CUDA 12.1）；nice-to-have：`uv sync --extra sdxl`。

## 5. 共用 Schema（必須先於任何 stage 完成）

直接對應 spec §5.2.1 / §5.2.2。

### 5.1 `FeatureSummary`

```python
class LyricSegment(BaseModel):
    time: float
    text: str

class FeatureSummary(BaseModel):
    lyrics: list[LyricSegment]
    video_description: str
    beat_times: list[float]
    chorus_segments: list[tuple[float, float]]
    video_duration: float
    style_preset: str
```

### 5.2 `Decision`（Discriminated Union）

```python
class TextElement(BaseModel):
    type: Literal["text"]
    content: str
    start_time: float
    end_time: float
    anchor: Literal["auto"] | tuple[int, int] = "auto"
    font: str
    size: int
    color: str
    outline_color: str
    outline_width: int
    animation: Literal["bounce_in", "typewriter", "wiggle", "draw_in", "fade"]
    rotation_jitter: float = 0
    reasoning: str          # spec 強制要求 chain-of-thought

class DecorationElement(BaseModel):
    type: Literal["decoration"]
    asset_tag: str
    near_text_id: int | None = None
    start_time: float
    end_time: float
    scale_jitter: float = 0
    rotation_jitter: float = 0
    reasoning: str

Element = Annotated[TextElement | DecorationElement, Field(discriminator="type")]

class GlobalStyle(BaseModel):
    color_palette: list[str]
    vibe: str

class Decision(BaseModel):
    elements: list[Element]
    global_style: GlobalStyle
```

**這兩個檔在 Week 1 第一天就要寫死**——後續所有 stage 的開發都依賴這個契約。一旦進 git，視為穩定 API；改動需要顯式版本升級。

## 6. LLM 抽象層

`llm/client.py` 提供統一 interface，讓 Claude 與 GPT-4o 可互換：

```python
class LLMClient(Protocol):
    def decide(self, summary: FeatureSummary, *, model: str) -> Decision: ...

class ClaudeClient:    # 用 anthropic SDK + tool use 強制 JSON schema
    ...

class OpenAIClient:    # 用 openai SDK + response_format=json_schema
    ...

def get_client(provider: str | None = None) -> LLMClient:
    # settings 為 config.py 內以 pydantic-settings 從 .env 讀取的單例
    provider = provider or settings.llm_provider
    return {"claude": ClaudeClient, "openai": OpenAIClient}[provider]()
```

`config.py` 定義成本模式：

```python
COST_MODES = {
    "dev":  {"claude": "claude-haiku-4-5",  "openai": "gpt-4o-mini"},
    "prod": {"claude": "claude-sonnet-4-6", "openai": "gpt-4o"},
}
```

- 預設 `dev` 模式（成本約 prod 的 1/20，適合反覆 iteration）
- Streamlit `app.py` / 期末 demo 切 `prod`
- Claude 走 prompt caching（system prompt + few-shot 重複部分自動快取，spec §7.1 預估成本可再降約 50%）

## 7. Week 1 Deliverable

對應 spec §8.1：

> 能跑通 `render_from_json(video_path, hand_written_json) → output.mp4`
> 輸出影片包含 1 段帶描邊的中文文字 + 1 個貼紙 + 基本動畫

骨架立完後，Week 1 第一個衝刺目標：

```bash
uv run python -m semanticvibe.render_demo \
    --video data/test_videos/sample_30s.mp4 \
    --json examples/hand_written_decision.json \
    --output outputs/week1_demo.mp4
```

僅需動到：

- `schemas/decision.py`
- `render/text_render.py`（Pillow 雙描邊中文渲染）
- `render/composite.py`（MoviePy 疊加 + ffmpeg 輸出）
- `examples/hand_written_decision.json`（一個手寫範例）
- `tests/test_render.py`（驗證輸出 mp4 存在 + 基本 metadata）

跑通即達成 Week 1 驗收。Stage 1 / 2 / 3 / 4 在後續週次接入。

## 8. 工具與慣例

| 項目 | 選擇 | 理由 |
|---|---|---|
| Version control | `git`（plan 階段執行 `git init`） | 所有 ML 專案 commit 史救過命；本案還沒 init |
| Lint / format | `ruff`（format + lint 一個工具） | 比 black + flake8 + isort 簡化 10× |
| Testing | `pytest` + `pytest-cov` | spec §9 自動化指標需要可重複量測 |
| Entry points | CLI（`python -m semanticvibe.cli`）+ Streamlit `app.py` 並存 | CLI 給 dev 用、Streamlit 給 demo |
| Config | `pydantic-settings` 讀 .env + 預設值 | 與 schemas 同生態，型別安全 |
| 中文字型 | Klee One（spec 指定）+ Noto Sans TC fallback | 都是 OFL，皆放 `data/fonts/` |
| 測試影片 | 自錄一支 30 秒；找不到時用 CC0 sample | Week 1 必備，否則無法驗證 render |
| LLM 結構化輸出 | Claude → tool use；OpenAI → `response_format=json_schema` | 兩家都有原生支援，將 retry 機率降到 <5% |
| Prompt caching | Claude 開、OpenAI 無此功能 | 進一步降低 spec §7.1 成本估算 |

## 9. 已知風險

### 9.1 Spec §10 已列風險

皆已在本架構內處理：
- **手繪風格還原度**：保留 `style_preset` 欄位 + Week 5 加 2–3 種風格包
- **LLM JSON 不穩**：Pydantic + tenacity retry + 兩家原生 structured output
- **MoviePy 中文字體**：用 Pillow 自渲染（spec §5.5.1 已決策）
- **渲染速度**：可降 720p / 預覽模式
- **素材版權**：嚴格 CC0 + 自製 + SD 生成
- **參考案例不足**：擴增 30+ 支多元參考（屬內容工作，本架構不阻擋）

### 9.2 本次新增風險

| 風險 | 對策 |
|---|---|
| MediaPipe 在 Python 3.13 無 wheel | 鎖 3.10 於 `.python-version` 與 `pyproject.toml` |
| Windows ffmpeg PATH 問題 | 用 `imageio-ffmpeg` bundle 版本，不依賴系統安裝 |
| `data/` 不進 git，新環境如何重建素材庫 | `data/README.md` 寫明：素材來源、SD 預生成腳本路徑、字型下載連結 |
| 實驗室共用 API quota 被燒光 | 預設 dev 模式（Haiku，成本約 prod 1/20）；每次呼叫前 log token 數；單次 cost ceiling 寫在 `config.py` |
| 雙 provider 抽象增加 1–2 天工 | 接受——換來 spec §3.3 貢獻 1 的 A/B 比較資料 |

## 10. 不在本範圍

以下事項屬於後續 design doc 的範圍：

- 各 stage 內部演算法細節（layout 評分函式、CLIP 檢索 top-k 策略、prompt 全文）
- 素材庫 metadata.json 的精確 schema
- Streamlit UI 設計
- 使用者研究問卷
- SDXL 預生成腳本

## 11. 下一步

1. 使用者審閱本文件
2. 進入 writing-plans skill，產出可執行的逐步實作 plan
3. Plan 第一步預計為：`git init` + 建立目錄結構 + `pyproject.toml` + `uv sync` + 寫 `schemas/` 兩檔
