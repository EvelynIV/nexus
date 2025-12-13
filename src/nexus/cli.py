import typer
from .main import run as run_server
import requests
app = typer.Typer()

@app.command()
def serve():
    """
    启动服务端
    """
    typer.echo("启动服务端...")
    run()

@app.command()
def run(file_path: str = typer.Argument(..., help="音频文件路径"),
                       model: str = typer.Option("ux_speech_grpc_proxy", help="使用的模型名称")):
    """
    测试转录接口
    """
    url = "http://127.0.0.1:8000/v1/audio/transcriptions"

    files = { "file": (file_path, open(file_path, "rb")) }
    payload = { "model": model }
    headers = {"Authorization": "Bearer <token>"}

    response = requests.post(url, data=payload, files=files, headers=headers)

    print(response.text)
if __name__ == "__main__":
    app()
