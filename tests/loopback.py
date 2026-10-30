"""Synthetic HTTP test network fence, independent of private experiment bundles."""
import socket

REAL_SOCKET = socket.socket
REAL_GETADDRINFO = socket.getaddrinfo


class LoopbackSocket(REAL_SOCKET):
    def bind(self, address):
        if address[0] != "127.0.0.1":
            raise AssertionError("non-loopback bind forbidden")
        return super().bind(address)

    def connect(self, address):
        if address[0] != "127.0.0.1":
            raise AssertionError("external network forbidden")
        return super().connect(address)

    def connect_ex(self, address):
        if address[0] != "127.0.0.1":
            raise AssertionError("external network forbidden")
        return super().connect_ex(address)


def loopback_resolve(host, *args, **kwargs):
    if host != "127.0.0.1":
        raise AssertionError("external DNS forbidden")
    return REAL_GETADDRINFO(host, *args, **kwargs)
