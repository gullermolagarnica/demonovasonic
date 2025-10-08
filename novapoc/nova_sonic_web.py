import argparse
import asyncio
import contextlib
import datetime
import ipaddress
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:  # Allow running as module or standalone script
    from .nova_sonic_full import BedrockStreamManager
except ImportError:  # pragma: no cover - fallback when executed directly
    from nova_sonic_full import BedrockStreamManager

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "web"), name="static")

FRONTEND_PATH = BASE_DIR / "web" / "index.html"


async def forward_bedrock_output(manager: BedrockStreamManager, websocket: WebSocket) -> None:
    """Relay Bedrock events to the connected websocket client."""
    try:
        while manager.is_active:
            payload: Dict[str, Any] = await manager.output_queue.get()
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            try:
                await websocket.send_text(json.dumps(payload))
            except RuntimeError:
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:  # pragma: no cover - defensive logging
        await websocket.send_text(json.dumps({"event": {"error": str(exc)}}))


@app.get("/")
async def index() -> HTMLResponse:
    """Serve the simple single page application."""
    html = FRONTEND_PATH.read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    manager = BedrockStreamManager(model_id="amazon.nova-sonic-v1:0", region="us-east-1")

    forward_task: Optional[asyncio.Task] = None
    try:
        await manager.initialize_stream()
        await websocket.send_json({"type": "session_ready"})

        forward_task = asyncio.create_task(forward_bedrock_output(manager, websocket))

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                audio_bytes: bytes = message["bytes"]
                manager.add_audio_chunk(audio_bytes)
                continue

            text_data = message.get("text")
            if not text_data:
                continue

            try:
                payload = json.loads(text_data)
            except json.JSONDecodeError:
                continue

            msg_type = payload.get("type")

            if msg_type == "stop_audio":
                await manager.send_audio_content_end_event()
            elif msg_type == "close_session":
                await websocket.close()
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        with contextlib.suppress(RuntimeError):
            await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        if forward_task:
            forward_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await forward_task
        await manager.close()


def ensure_self_signed_cert(
    cert_dir: Path, common_name: str = "nova-sonic-local", extra_hosts: Optional[list[str]] = None
) -> Tuple[Path, Path]:
    """Create (or reuse) a self-signed certificate for HTTPS local hosting."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Nova Sonic Local"),
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
    ])

    alt_names = {
        x509.DNSName("localhost"),
        x509.DNSName(common_name),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    }

    if extra_hosts:
        for host in extra_hosts:
            try:
                alt_names.add(x509.IPAddress(ipaddress.ip_address(host)))
            except ValueError:
                alt_names.add(x509.DNSName(host))

    san = x509.SubjectAlternativeName(list(alt_names))

    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(san, critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    with key_path.open("wb") as key_file:
        key_file.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with cert_path.open("wb") as cert_file:
        cert_file.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Nova Sonic web interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Serve over plain HTTP (not recommended – mic capture needs HTTPS if not localhost)",
    )
    parser.add_argument(
        "--cert-dir",
        default=str(BASE_DIR / "certs"),
        help="Directory to store the auto-generated self-signed certificate",
    )

    args = parser.parse_args()

    ssl_kwargs: Dict[str, str] = {}
    if not args.insecure:
        try:
            extra_hosts = None
            if args.host not in {"0.0.0.0", "::", ""}:
                extra_hosts = [args.host]
            cert_file, key_file = ensure_self_signed_cert(Path(args.cert_dir), extra_hosts=extra_hosts)
            ssl_kwargs["ssl_certfile"] = str(cert_file)
            ssl_kwargs["ssl_keyfile"] = str(key_file)
            scheme = "https"
        except Exception as exc:  # pragma: no cover - fallback for environments without cryptography
            print(f"No se pudo preparar el certificado automático: {exc}")
            print("Sirviendo vía HTTP sin cifrado. Usa --insecure para evitar este mensaje.")
            scheme = "http"
    else:
        scheme = "http"

    print(f"Nova Sonic Web UI disponible en {scheme}://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, **ssl_kwargs)
