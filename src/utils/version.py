"""Bot version tracking with changelog."""

VERSION = "1.3.0"
VERSION_NAME = "Sell Machine"

# Changelog - newest first
CHANGELOG = [
    {
        "version": "1.3.0",
        "name": "Sell Machine",
        "changes": [
            "/sellall - Close ALL open positions at market",
            "/update - Git pull & restart from Telegram",
            "Startup notification on Telegram when bot starts",
            "Version tracking with changelog",
        ],
    },
    {
        "version": "1.2.0",
        "name": "Strategy Overhaul",
        "changes": [
            "Fixed 7.5% win rate with new signal scoring",
            "ADX & volume pre-filters for better entries",
            "Multi-timeframe alignment bonus",
        ],
    },
    {
        "version": "1.1.0",
        "name": "Stability Fix",
        "changes": [
            "Fixed scan delays and stale order cleanup",
            "Fixed signal bias causing only SHORT trades",
            "Fixed SL/TP side mismatch",
        ],
    },
    {
        "version": "1.0.0",
        "name": "Initial Release",
        "changes": [
            "AI-powered trading with Claude Vision",
            "TradingView + multi-timeframe analysis",
            "IBKR integration with trailing stops",
            "Telegram bot with 14 commands",
        ],
    },
]


def get_version_string() -> str:
    return f"v{VERSION} \"{VERSION_NAME}\""


def get_latest_changelog() -> str:
    """Get formatted changelog for the current version."""
    entry = CHANGELOG[0]
    lines = [f"v{entry['version']} \"{entry['name']}\"", ""]
    for change in entry["changes"]:
        lines.append(f"  - {change}")
    return "\n".join(lines)
