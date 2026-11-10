"""E23: VPN crypto — ChaCha20-Poly1305 from scratch.

WireGuard's transport encryption is ChaCha20-Poly1305. We implement the
primitive from the RFC 8439 specification and verify:
  * the RFC 8439 ChaCha20 keystream test vector,
  * the full AEAD against the independent `cryptography` implementation
    on random keys/nonces/AAD/messages,
  * tamper detection (a flipped ciphertext bit must fail authentication).
"""
from __future__ import annotations

import os
import struct

import numpy as np

from fsl.common import ExperimentResult, run_experiment

MASK = 0xFFFFFFFF


def rotl(v: int, c: int) -> int:
    return ((v << c) & MASK) | (v >> (32 - c))


def qr(s: list[int], a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & MASK; s[d] = rotl(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & MASK; s[b] = rotl(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b]) & MASK; s[d] = rotl(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & MASK; s[b] = rotl(s[b] ^ s[c], 7)


def chacha_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    const = b"expand 32-byte k"
    st = list(struct.unpack("<4I", const)) + list(struct.unpack("<8I", key)) \
        + [counter] + list(struct.unpack("<3I", nonce))
    w = st.copy()
    for _ in range(10):
        qr(w, 0, 4, 8, 12); qr(w, 1, 5, 9, 13); qr(w, 2, 6, 10, 14); qr(w, 3, 7, 11, 15)
        qr(w, 0, 5, 10, 15); qr(w, 1, 6, 11, 12); qr(w, 2, 7, 8, 13); qr(w, 3, 4, 9, 14)
    return struct.pack("<16I", *[(w[i] + st[i]) & MASK for i in range(16)])


def chacha_xor(key: bytes, nonce: bytes, data: bytes, counter: int = 1) -> bytes:
    out = bytearray()
    for i in range(0, len(data), 64):
        ks = chacha_block(key, counter + i // 64, nonce)
        chunk = data[i:i + 64]
        out += bytes(a ^ b for a, b in zip(chunk, ks))
    return bytes(out)


def poly1305(key: bytes, msg: bytes) -> bytes:
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:], "little")
    p = (1 << 130) - 5
    acc = 0
    for i in range(0, len(msg), 16):
        chunk = msg[i:i + 16]
        n = int.from_bytes(chunk + b"\x01", "little")
        acc = ((acc + n) * r) % p
    return ((acc + s) % (1 << 128)).to_bytes(16, "little")


def pad16(b: bytes) -> bytes:
    return b"\x00" * ((16 - len(b) % 16) % 16)


def aead_encrypt(key: bytes, nonce: bytes, msg: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
    otk = chacha_block(key, 0, nonce)[:32]
    ct = chacha_xor(key, nonce, msg, counter=1)
    mac_data = aad + pad16(aad) + ct + pad16(ct) + struct.pack("<QQ", len(aad), len(ct))
    return ct, poly1305(otk, mac_data)


def run() -> ExperimentResult:
    def body(res: ExperimentResult) -> None:
        key = bytes(range(32))
        nonce = bytes.fromhex("000000090000004a00000000")
        ks = chacha_block(key, 1, nonce)
        expect = bytes.fromhex(
            "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
            "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
        res.check("ChaCha20 matches RFC 8439 keystream vector", ks == expect,
                  f"first bytes={ks[:8].hex()}")

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        from cryptography.exceptions import InvalidTag
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(20):
            k = rng.bytes(32); n = rng.bytes(12)
            msg = rng.bytes(int(rng.integers(0, 200)))
            aad = rng.bytes(int(rng.integers(0, 40)))
            ct, tag = aead_encrypt(k, n, msg, aad)
            ref = ChaCha20Poly1305(k).encrypt(n, msg, aad)
            if ct + tag == ref:
                ok += 1
        res.check("AEAD matches the cryptography library on 20 random cases",
                  ok == 20, f"{ok}/20 exact matches")

        k = os.urandom(32); n = os.urandom(12); msg = b"attack at dawn"
        ct, tag = aead_encrypt(k, n, msg)
        bad = bytearray(ct); bad[0] ^= 1
        detected = False
        try:
            ChaCha20Poly1305(k).decrypt(n, bytes(bad) + tag, b"")
        except InvalidTag:
            detected = True
        res.check("tampered ciphertext fails authentication", detected, "InvalidTag raised")

        res.metrics.update({"rfc_vector_ok": ks == expect, "aead_matches": ok,
                            "tag_bytes": len(tag)})
    return run_experiment("e23_crypto", body)
