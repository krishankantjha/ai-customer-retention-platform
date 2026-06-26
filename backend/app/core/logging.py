"""
Logging configuration module.
Loads logging dictConfig parameters from configs/logging_config.yaml.
"""

import os
import logging
import logging.config
import yaml
import re

try:
    from pythonjsonlogger.json import JsonFormatter
    has_json_logger = True
except ImportError:
    class JsonFormatter(logging.Formatter):
        pass
    has_json_logger = False

class SensitiveDataFilter(logging.Filter):
    """
    Logging filter to redact sensitive fields (passwords, tokens, keys, PII)
    from log messages and structured dictionaries before they are written.
    """
    SENSITIVE_KEYS = {
        "password", "passwd", "pwd", "token", "jwt", "api_key", "secret", 
        "authorization", "monthly_charges", "monthlycharges", "total_charges", "totalcharges"
    }

    def redact_message(self, text: str) -> str:
        keys_pattern = r'(?:password|passwd|pwd|token|jwt|api_key|secret|authorization|monthly_charges|monthlycharges|total_charges|totalcharges)'
        pattern = rf'(?i)(\b{keys_pattern}\b\s*[:=]\s*)(?:(?P<quote>["\'])(.*?)(?P=quote)|([^\n,\'"}}]+))'
        
        def repl(match):
            prefix = match.group(1)
            quote = match.group('quote')
            if quote:
                return f"{prefix}{quote}[REDACTED]{quote}"
            else:
                return f"{prefix}[REDACTED]"
                
        return re.sub(pattern, repl, text)

    def redact_dict(self, d: dict) -> dict:
        new_dict = {}
        for k, v in d.items():
            if isinstance(k, str) and k.lower() in self.SENSITIVE_KEYS:
                new_dict[k] = "[REDACTED]"
            elif isinstance(v, dict):
                new_dict[k] = self.redact_dict(v)
            elif isinstance(v, list):
                new_dict[k] = [self.redact_dict(item) if isinstance(item, dict) else item for item in v]
            else:
                new_dict[k] = v
        return new_dict

    def scrub_value(self, val):
        if isinstance(val, dict):
            return self.redact_dict(val)
        elif isinstance(val, list):
            return [self.scrub_value(item) for item in val]
        elif isinstance(val, str):
            return self.redact_message(val)
        elif hasattr(val, "model_dump"): # Pydantic v2
            try:
                return self.redact_dict(val.model_dump())
            except Exception:
                return "[REDACTED_OBJECT]"
        elif hasattr(val, "dict"): # Pydantic v1
            try:
                return self.redact_dict(val.dict())
            except Exception:
                return "[REDACTED_OBJECT]"
        elif hasattr(val, "__dict__"): # Custom objects
            try:
                return self.redact_dict(vars(val))
            except Exception:
                return "[REDACTED_OBJECT]"
        else:
            val_str = str(val)
            redacted_str = self.redact_message(val_str)
            if redacted_str != val_str:
                return "[REDACTED]"
            return val

    def filter(self, record: logging.LogRecord) -> bool:
        # Non-destructive filter. Redaction is handled inside custom formatters.
        return True


class SensitiveDataFormatter(logging.Formatter):
    """
    Custom Formatter that redacts sensitive information from formatted text logs.
    """
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.redactor = SensitiveDataFilter()

    def format(self, record: logging.LogRecord) -> str:
        import copy
        record_copy = copy.copy(record)
        
        # Format the copy
        formatted = super().format(record_copy)
        redacted = self.redactor.redact_message(formatted)
        
        # Also redact the traceback if exc_text was already generated
        if record_copy.exc_text:
            record_copy.exc_text = self.redactor.redact_message(record_copy.exc_text)
            
        return redacted

    def formatException(self, ei) -> str:
        formatted_exc = super().formatException(ei)
        redactor = SensitiveDataFilter()
        return redactor.redact_message(formatted_exc)


class SensitiveJsonFormatter(JsonFormatter):
    """
    Custom JSON Formatter that redacts sensitive information from structured logs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redactor = SensitiveDataFilter()

    def format(self, record: logging.LogRecord) -> str:
        import copy
        record_copy = copy.copy(record)
        
        # Format positional args into record_copy.msg first
        try:
            if record_copy.args:
                record_copy.msg = record_copy.msg % record_copy.args
                record_copy.args = ()
        except Exception:
            pass
            
        record_copy.msg = self.redactor.redact_message(str(record_copy.msg))
        
        # Redact other custom keys recursively on the copy
        for key in list(record_copy.__dict__.keys()):
            if key.startswith("_") or key in {
                "name", "msg", "args", "levelname", "levelno", "pathname", "filename", 
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", 
                "created", "msecs", "relativeCreated", "thread", "threadName", 
                "processName", "process"
            }:
                continue
            val = record_copy.__dict__[key]
            record_copy.__dict__[key] = self.redactor.scrub_value(val)
            
        # Redact tracebacks
        if record_copy.exc_text:
            record_copy.exc_text = self.redactor.redact_message(record_copy.exc_text)
            
        return super().format(record_copy)

    def formatException(self, ei) -> str:
        formatted_exc = super().formatException(ei)
        return self.redactor.redact_message(formatted_exc)


def setup_logging() -> None:
    """
    Locates logging_config.yaml, parses parameters, performs dynamic fallback,
    scans paths, sets log levels, filters sensitive info, and runs dictConfig.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    config_path = os.path.join(project_root, "configs", "logging_config.yaml")

    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 1. Dependency Resilience: Fallback if python-json-logger is missing
        if not has_json_logger:
            # Fall back to standard detailed text formatter for all handlers using json
            if "handlers" in config:
                for handler_conf in config["handlers"].values():
                    if handler_conf.get("formatter") == "json":
                        handler_conf["formatter"] = "detailed"
            if "formatters" in config:
                config["formatters"].pop("json", None)

        # 2. Path Reliability: Force absolute logs directory
        logs_dir = os.path.join(project_root, "backend", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        if "handlers" in config and "file" in config["handlers"]:
            config["handlers"]["file"]["filename"] = os.path.join(logs_dir, "backend.log")

        # 3. Environment-Aware Log Levels
        app_env = os.getenv("APP_ENV", "development").lower()
        default_log_level = "DEBUG" if app_env == "development" else "INFO"
        log_level = os.getenv("LOG_LEVEL", default_log_level).upper()
        
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if log_level not in valid_levels:
            log_level = default_log_level

        # Update root logger level
        if "root" in config:
            config["root"]["level"] = log_level

        # Update specific loggers
        if "loggers" in config:
            for logger_name, logger_conf in config["loggers"].items():
                if logger_name == "app":
                    logger_conf["level"] = log_level
                elif logger_name.startswith("sqlalchemy"):
                    logger_conf["level"] = "INFO" if log_level == "DEBUG" else "WARNING"
                else:
                    logger_conf["level"] = log_level

        # Update handlers level independently to support environment-level override per handler
        if "handlers" in config:
            for handler_name, handler_conf in config["handlers"].items():
                if handler_name == "console":
                    handler_conf["level"] = os.getenv("CONSOLE_LOG_LEVEL", log_level)
                elif handler_name == "file":
                    handler_conf["level"] = os.getenv("FILE_LOG_LEVEL", handler_conf.get("level", log_level))
                else:
                    handler_conf["level"] = log_level

        # 4. Inject Custom Formatters dynamically into dictConfig structure
        # Inject SensitiveDataFormatter and SensitiveJsonFormatter
        if "formatters" in config:
            if "default" in config["formatters"]:
                config["formatters"]["default"]["()"] = SensitiveDataFormatter
            if "detailed" in config["formatters"]:
                config["formatters"]["detailed"]["()"] = SensitiveDataFormatter
            if "json" in config["formatters"]:
                config["formatters"]["json"]["()"] = SensitiveJsonFormatter

        logging.config.dictConfig(config)
        logging.info(f"Logging configured successfully (Env: {app_env}, Level: {log_level}, JSON: {has_json_logger})")

    except Exception as e:
        # Fallback to basic stdout configuration if anything fails (YAML error, permission error, etc.)
        root_logger = logging.getLogger()
        for h in list(root_logger.handlers):
            root_logger.removeHandler(h)
            
        fallback_handler = logging.StreamHandler()
        fallback_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fallback_handler.setFormatter(fallback_formatter)
        root_logger.addHandler(fallback_handler)
        root_logger.setLevel(logging.INFO)
        
        # Reset all other loggers to propagate and clear their handlers
        for logger_name in logging.root.manager.loggerDict:
            logger_inst = logging.getLogger(logger_name)
            for h in list(logger_inst.handlers):
                logger_inst.removeHandler(h)
            logger_inst.propagate = True
            
        logging.error(f"Fallback Logger Triggered: Logging initialization failed: {e}")


# Automatically configure logging when module is imported
setup_logging()

