"""Internationalization strings — Chinese / English.

Usage:
    from i18n.strings import t, set_lang

    set_lang("zh")  # or "en"
    label = t("nav_overview")
"""

_lang = "zh"

_T = {
    "nav_overview":   {"zh": "概览",           "en": "Overview"},
    "nav_b0":         {"zh": "B0 图处理",      "en": "B0 Map"},
    "nav_optimize":   {"zh": "一键匀场",       "en": "One-Click Shim"},
    "nav_results":    {"zh": "结果分析",       "en": "Results"},
    "nav_teensy":     {"zh": "Teensy 上传",    "en": "Teensy Upload"},
    "status_title":   {"zh": "流程状态",       "en": "Workflow Status"},
    "s_b0":           {"zh": "B0 图已加载",    "en": "B0 Map loaded"},
    "s_ch":           {"zh": "通道图已加载",    "en": "Channel maps loaded"},
    "s_curr":         {"zh": "优化电流已生成",  "en": "Currents optimized"},
    "s_teensy":       {"zh": "Teensy 已连接",  "en": "Teensy connected"},
    "connect":        {"zh": "连接",           "en": "Connect"},
    "disconnect":     {"zh": "断开",           "en": "Disconnect"},
    "send":           {"zh": "发送",           "en": "Send"},
    "refresh":        {"zh": "刷新",           "en": "Refresh"},
    "upload":         {"zh": "上传电流",        "en": "Upload Currents"},
}


def set_lang(lang: str):
    global _lang
    _lang = lang


def t(key: str) -> str:
    return _T.get(key, {}).get(_lang, key)
