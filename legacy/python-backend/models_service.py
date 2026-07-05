from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KATAGO_ROOT = REPO_ROOT.parent / "KataGo"
KATAGO_EXECUTABLE = KATAGO_ROOT / "cpp" / "build-metal" / "katago"
KATAGO_MODEL_PATH = KATAGO_ROOT / "cpp" / "tests" / "models" / "g170-b6c96-s175395328-d26788732.bin.gz"
KATAGO_CONFIG_PATH = KATAGO_ROOT / "cpp" / "configs" / "gtp_example.cfg"


def discover_models() -> list[dict[str, object]]:
    if not (KATAGO_EXECUTABLE.exists() and KATAGO_MODEL_PATH.exists() and KATAGO_CONFIG_PATH.exists()):
        return []

    return [
        {
            "path": str(KATAGO_MODEL_PATH.resolve()),
            "name": "KataGo 19路 Metal",
            "round": 0,
            "type": "katago",
            "priority": 0,
        }
    ]


def get_default_model_path() -> str | None:
    models = discover_models()
    if not models:
        return None
    return str(models[0]["path"])


def get_default_battle_model_paths() -> tuple[str | None, str | None]:
    default_model = get_default_model_path()
    return default_model, default_model


def normalize_model_path(candidate: str | None) -> str:
    models = discover_models()
    if not models:
        raise ValueError("未找到 KataGo 可用模型，请先确认 KataGo 已完成 Metal 构建")

    if candidate is None or candidate == "":
        return str(models[0]["path"])

    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = (REPO_ROOT / candidate_path).resolve()
    else:
        candidate_path = candidate_path.resolve()

    for model in models:
        if Path(str(model["path"])).resolve() == candidate_path:
            return str(candidate_path)
    raise ValueError("模型不存在或不在允许列表中")
