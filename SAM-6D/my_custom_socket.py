# === Imports ===
import socket
import struct
from io import BytesIO

import cv2
import numpy as np


# === Helper: Receive exactly N bytes ===
def recv_exact(conn, num_bytes):
    """Receive exactly num_bytes from the socket."""
    data = b""
    while len(data) < num_bytes:
        packet = conn.recv(num_bytes - len(data))
        if not packet:
            raise ConnectionError(
                "Connection closed unexpectedly while receiving data."
            )
        data += packet
    return data


# === Helper: Decode bytes into Python object based on type code ===
def decode(data, type_decode):
    if type_decode == b"I":  # image2d (grayscale/mask)
        msg = np.load(BytesIO(data), allow_pickle=False)
    elif type_decode == b"J":  # image3d (RGB image)
        msg = np.load(BytesIO(data), allow_pickle=False)
    elif type_decode == b"i":  # integer
        msg = struct.unpack(">i", data)[0]
    elif type_decode == b"f":  # float
        msg = struct.unpack(">d", data)[0]
    elif type_decode == b"s":  # string
        msg = data.decode()
    elif type_decode == b"b":  # bool
        msg = bool(data[0])
    elif type_decode == b"n":  # numpy array
        msg = np.load(BytesIO(data), allow_pickle=False)
    else:  # fallback: raw bytes
        msg = data
    return msg


# === Helper: Encode Python object into bytes based on type string ===
def encode(data, type_encode):
    if type_encode == "string":
        msg = data.encode()
        payload = b"s"
    elif type_encode == "int":
        msg = struct.pack(">i", data)
        payload = b"i"
    elif type_encode == "float":
        msg = struct.pack(">d", data)
        payload = b"f"
    elif type_encode == "bool":
        msg = b"\x01" if data else b"\x00"
        payload = b"b"
    elif type_encode == "image2d":
        assert data.ndim == 2
        buff = BytesIO()
        np.save(buff, data, allow_pickle=False)
        msg = buff.getvalue()
        payload = b"I"
    elif type_encode == "image3d":
        assert data.ndim == 3 and data.shape[2] == 3
        buff = BytesIO()
        np.save(buff, data, allow_pickle=False)
        msg = buff.getvalue()
        payload = b"J"
    elif type_encode == "numpyarray":
        buff = BytesIO()
        np.save(buff, data, allow_pickle=False)
        msg = buff.getvalue()
        payload = b"n"
    else:
        raise ValueError(f"Unsupported return type: {type_encode}")
    return msg, payload


# ======================
# === MyServer Class ===
# ======================
class MyServer:
    def __init__(self, host="127.0.0.1", port=65432, server_name="My Server"):
        self.host = host
        self.port = port
        self.server_name = server_name

    def start(self):
        """Start the TCP server and wait for one client connection."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"[{self.server_name}] Listening on {self.host}:{self.port}")
            self.conn, addr = s.accept()
            print(f"[{self.server_name}] Connected by {addr}")

    def restart(self):
        """Close current connection and restart listening."""
        self.conn.close()
        self.start()

    def wait_for_image(self):
        """Receive a single image using legacy image-only method."""
        length = struct.unpack(">I", recv_exact(self.conn, 4))[0]
        image_data = recv_exact(self.conn, length)
        np_data = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return img

    def send_result(self, result):
        """Send a float result (8 bytes) using legacy image-only method."""
        data = struct.pack(">d", result)
        self.conn.sendall(data)

    def wait_for_msg(self):
        """Receive a flexible message consisting of multiple typed items."""
        msg = []

        try:
            count_recv = recv_exact(self.conn, 4)
        except Exception as e:
            print(e)
            return None
        count = struct.unpack(">I", count_recv)[0]

        for _ in range(count):
            type_decode = recv_exact(self.conn, 1)
            length = struct.unpack(">I", recv_exact(self.conn, 4))[0]
            data = recv_exact(self.conn, length)
            decoded_msg = decode(data=data, type_decode=type_decode)
            msg.append(decoded_msg)

        return msg

    def send_response(self, msg_type_out, msg_out):
        """Send a response containing multiple typed items."""
        payload = struct.pack(">I", len(msg_out))

        for type_encode, data in zip(msg_type_out, msg_out):
            encoded_data, type_payload = encode(data, type_encode=type_encode)
            payload += type_payload
            payload += struct.pack(">I", len(encoded_data))
            payload += encoded_data

        self.conn.sendall(payload)


# ======================
# === MyClient Class ===
# ======================
class MyClient:
    def __init__(self, host="127.0.0.1", port=65432, client_name="My Client"):
        self.host = host
        self.port = port
        self.client_name = client_name

    def connect(self):
        """Start the client and connect to the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"[{self.client_name}] Connected to server.")
            return True
        except Exception as e:
            print(e)
            return False

    def disconnect(self):
        self.socket.close()

    def request_image(self, img: np.array):
        """Send one image (legacy API) and get a float result back."""
        if img is None:
            print(f"[{self.client_name}] Failed to load image.")
            return

        _, img_encoded = cv2.imencode(".png", img)
        data = img_encoded.tobytes()
        msg = struct.pack(">I", len(data)) + data

        self.socket.sendall(msg)
        result_data = recv_exact(self.socket, 8)
        recv_value = struct.unpack(">d", result_data)[0]
        return recv_value

    def request_msg(self, msg_type_in, msg_in):
        """Send a list of inputs with types, receive a list of typed outputs."""
        payload = struct.pack(">I", len(msg_in))

        for type_encode, data in zip(msg_type_in, msg_in):
            encoded_data, type_payload = encode(data, type_encode=type_encode)
            payload += type_payload
            payload += struct.pack(">I", len(encoded_data))
            payload += encoded_data

        self.socket.sendall(payload)

        count_bytes = recv_exact(self.socket, 4)
        msg_count = struct.unpack(">I", count_bytes)[0]

        msg = []
        for _ in range(msg_count):
            type_decode = recv_exact(self.socket, 1)
            length = struct.unpack(">I", recv_exact(self.socket, 4))[0]
            data = recv_exact(self.socket, length)
            decoded_msg = decode(data=data, type_decode=type_decode)
            msg.append(decoded_msg)

        return msg
