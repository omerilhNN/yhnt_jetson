#!/usr/bin/env python3
"""
MJPEG HTTP server — Jetson tarafı.


Pipeline'dan gelen JPEG frame'leri bellekte tutar ve HTTP üzerinden
multipart/x-mixed-replace ("motion JPEG") olarak servis eder.


- Tek bir "latest frame" buffer'ı tutulur (paket kaybı yok, her zaman en güncel kare).
- Her client kendi thread'inde, kondisyon değişkeni ile yeni kareyi bekler.
- Yavaş client hızlı client'ı bloklamaz; herkes son kareyi alır (frame drop, queue yok).


Kullanım:
    server = MjpegServer(host="0.0.0.0", port=8090)
    server.start()
    ...
    server.update_frame(jpeg_bytes)   # her yeni JPEG karede çağır
    ...
    server.stop()
"""


import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_BOUNDARY = "highwayframe"




class _FrameHub:
    """En güncel JPEG karesini tutar, client'ları uyandırır."""


    def __init__(self):
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0
        self._closed = False


    def update(self, jpeg: bytes):
        with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()


    def wait_for(self, last_seq: int, timeout: float = 5.0):
        """last_seq'ten yeni bir kare gelene kadar bekler. (frame, seq) veya (None, seq)."""
        with self._cond:
            if self._closed:
                return None, self._seq
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._closed:
                return None, self._seq
            return self._frame, self._seq


    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()




class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"


    def log_message(self, *args):
        pass  # erişim loglarını sustur


    def _send_cors_preflight(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


    def do_OPTIONS(self):
        self._send_cors_preflight()


    def do_GET(self):
        if self.path.rstrip("/") not in ("", "/stream", "/mjpeg"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return


        hub: _FrameHub = self.server.frame_hub  # type: ignore[attr-defined]


        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )
        self.end_headers()


        last_seq = -1
        try:
            while True:
                frame, last_seq = hub.wait_for(last_seq)
                if frame is None:
                    if getattr(hub, "_closed", False):
                        break
                    continue
                chunk = (
                    f"--{_BOUNDARY}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode("ascii") + frame + b"\r\n"
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass




class MjpegServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8090):
        self.host = host
        self.port = port
        self._hub = _FrameHub()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None


    def start(self):
        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.frame_hub = self._hub  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="mjpeg-http"
        )
        self._thread.start()
        print(f"[mjpeg] HTTP MJPEG server başlatıldı: http://{self.host}:{self.port}/stream")


    def update_frame(self, jpeg: bytes):
        self._hub.update(jpeg)


    def stop(self):
        self._hub.close()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=2.0)




