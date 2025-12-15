# src/nexus/core/logging.py
# 作用：统一日志配置。

# 内容：

# 设置日志格式、log level（读取 settings.LOG_LEVEL）。

# 可添加 JSON 日志、请求 ID、链路跟踪信息等。

# 好处：

# 所有模块调用 logging.getLogger(__name__) 都能享受统一格式和输出。

#为跟logger配置日志，disconfig负责分析整理传入信息
from logging.config import dictConfig

def setup_logging():
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
            }
        },
        "root": {   # 用 root 更清晰
            "handlers": ["console"],
            "level": "INFO",
        },
        "loggers": {
            # 只让你自己的代码包名输出 DEBUG（把 nexus 换成你的顶层包名）
            "nexus": {"level": "DEBUG", "propagate": True},

            # 压制 multipart 的 DEBUG（常见 logger 名）
            "multipart": {"level": "WARNING", "propagate": True},

           
        },
    })

#关于logger的重复打印问题，：当下层looger在冒泡到根logger时，如果根logger和下层logger都有handler，就会导致日志被打印多次。
#解决方法：确保只有根logger有handler
#过滤不来自于自己包的debug级别的日志