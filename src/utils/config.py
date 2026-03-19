import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


_config = load_config()


def get_config() -> dict:
    return _config


def reload_config() -> dict:
    global _config
    _config = load_config()
    return _config


# Environment shortcuts
IB_HOST = os.getenv("IB_HOST", _config["broker"]["host"])
IB_PORT = int(os.getenv("IB_PORT", _config["broker"]["port"]))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", _config["broker"]["client_id"]))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TRADING_MODE = os.getenv("TRADING_MODE", _config["trading"]["mode"])
