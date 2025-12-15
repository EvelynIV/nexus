# src/nexus/core/middleware.py
#中间件逻辑不是有我们直接编写的路由处理函数调用的，而是由框架在请求进入和响应返回时自动调用的。
#由BaseHTTPMiddleware继承而来，可以在FastAPI应用中注册使用。是它实现的，我们只负责把想要的中间件写入即可
import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from .logging import setup_logging
import logging


setup_logging()


class CustomMiddleware(BaseHTTPMiddleware):
    #核心，是中间件的入口，每个请求调用，从base..类继承过来
    #使用BaseHTTPMiddleware：通过覆写dispatch方法来实现自定义中间件逻辑。
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())  # Generate a unique request ID
        start_time = time.time()
        
        response = await call_next(request)

        # Log request and response details (including execution time)
        exec_time = time.time() - start_time
        logging.info(
            f"Request ID: {request_id}, Method: {request.method}, Path: {request.url.path}, Time: {exec_time:.4f}s"
        )
        #建议写 logger.info("x=%s", x)
        #logging.info(...) 是模块级快捷函数，本质上是对 root logger 发日志。

        # Attach request_id to response headers
        response.headers['X-Request-ID'] = request_id

        return response
