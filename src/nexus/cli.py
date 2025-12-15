import typer
from .main import run as run_server
import requests
from pathlib import Path

app = typer.Typer()

@app.command()
def serve():
    """启动服务端"""
    typer.echo("启动服务端...")
    run_server()

@app.command("transcribe")
def transcribe(
    file_path: Path = typer.Argument(..., exists=True, readable=True, help="音频文件路径"),
    
):
    """测试转录接口"""
    url = "http://127.0.0.1:8000/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer <token>"}

    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f)}
        payload = {"model": "ux_speech_grpc_proxy"}
        response = requests.post(url, data=payload, files=files, headers=headers)

    print(response.text)

if __name__ == "__main__":
    app()