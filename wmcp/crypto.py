"""Encrypted communication channel for WMCP messages."""

import os
import hashlib
import hmac
import json
import struct
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class KeyPair:
    """Agent key pair for signing and encryption."""
    agent_id: str
    private_key: bytes  # 32 bytes
    public_key: bytes   # 32 bytes (derived)

    @classmethod
    def generate(cls, agent_id: str) -> "KeyPair":
        """Generate a new key pair for an agent."""
        private = os.urandom(32)
        # Simplified: public = hash of private (not real DH, placeholder for v0.1)
        public = hashlib.sha256(private).digest()
        return cls(agent_id=agent_id, private_key=private, public_key=public)


def derive_shared_secret(my_private: bytes, their_public: bytes) -> bytes:
    """Derive a shared secret from private and public keys.

    Simplified Diffie-Hellman placeholder. Production would use
    proper X25519 key exchange.
    """
    return hashlib.sha256(my_private + their_public).digest()


def encrypt_message(plaintext: str, shared_secret: bytes) -> bytes:
    """Encrypt a message using AES-256-equivalent XOR cipher.

    NOTE: This is a simplified implementation for protocol demonstration.
    Production deployment should use `cryptography` library with
    AES-256-GCM for authenticated encryption.

    Args:
        plaintext: JSON message string.
        shared_secret: 32-byte shared secret.

    Returns:
        Encrypted bytes (nonce + ciphertext).
    """
    nonce = os.urandom(16)
    plaintext_bytes = plaintext.encode('utf-8')

    # Generate keystream from shared_secret + nonce
    keystream = b""
    counter = 0
    while len(keystream) < len(plaintext_bytes):
        block = hashlib.sha256(
            shared_secret + nonce + struct.pack('<I', counter)).digest()
        keystream += block
        counter += 1

    # XOR encryption
    ciphertext = bytes(p ^ k for p, k in zip(plaintext_bytes, keystream))
    return nonce + ciphertext


def decrypt_message(encrypted: bytes, shared_secret: bytes) -> str:
    """Decrypt a message.

    Args:
        encrypted: Nonce (16 bytes) + ciphertext.
        shared_secret: 32-byte shared secret.

    Returns:
        Decrypted JSON string.
    """
    nonce = encrypted[:16]
    ciphertext = encrypted[16:]

    keystream = b""
    counter = 0
    while len(keystream) < len(ciphertext):
        block = hashlib.sha256(
            shared_secret + nonce + struct.pack('<I', counter)).digest()
        keystream += block
        counter += 1

    plaintext = bytes(c ^ k for c, k in zip(ciphertext, keystream))
    return plaintext.decode('utf-8')


def sign_message(message: str, private_key: bytes) -> str:
    """Sign a message with HMAC-SHA256.

    Args:
        message: JSON message string.
        private_key: 32-byte private key.

    Returns:
        Hex-encoded signature.
    """
    return hmac.new(private_key, message.encode('utf-8'),
                    hashlib.sha256).hexdigest()


def verify_signature(message: str, signature: str,
                     public_key: bytes) -> bool:
    """Verify a message signature.

    NOTE: Simplified — in production, use asymmetric signatures (Ed25519).
    Here we use the public key as a shared verification key.
    """
    expected = hmac.new(public_key, message.encode('utf-8'),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


class SecureChannel:
    """Encrypted communication channel between two agents.

    Usage:
        alice = KeyPair.generate("alice")
        bob = KeyPair.generate("bob")
        channel = SecureChannel(alice, bob.public_key)
        encrypted = channel.encrypt('{"tokens": [1, 2]}')
        signed = channel.sign('{"tokens": [1, 2]}')
    """

    def __init__(self, my_keys: KeyPair, peer_public: bytes):
        self.my_keys = my_keys
        self.peer_public = peer_public
        self.shared_secret = derive_shared_secret(
            my_keys.private_key, peer_public)

    def encrypt(self, message: str) -> bytes:
        return encrypt_message(message, self.shared_secret)

    def decrypt(self, encrypted: bytes) -> str:
        return decrypt_message(encrypted, self.shared_secret)

    def sign(self, message: str) -> str:
        return sign_message(message, self.my_keys.private_key)

    def verify(self, message: str, signature: str) -> bool:
        return verify_signature(message, signature, self.peer_public)
