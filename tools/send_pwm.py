#!/usr/bin/env python3
import argparse
import time

import serial


BAUD = 115200
RATE_HZ = 100
DEFAULT_DEVICE = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0"


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack7(raw: bytes) -> bytes:
    acc = 0
    nbits = 0
    out = bytearray()
    for b in raw:
        acc = (acc << 8) | b
        nbits += 8
        while nbits >= 7:
            nbits -= 7
            out.append((acc >> nbits) & 0x7F)
    if nbits > 0:
        out.append((acc << (7 - nbits)) & 0x7F)
    return bytes(out)


def build_frame(motors, seq):
    payload = b"".join(int(m & 0xFFFF).to_bytes(2, "big") for m in motors)
    crc = crc16_ccitt(payload)
    raw = payload + crc.to_bytes(2, "big")
    return bytes([0x80 | (seq & 0x0F)]) + pack7(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("motor_0", type=int)
    ap.add_argument("motor_1", type=int)
    ap.add_argument("motor_2", type=int)
    ap.add_argument("motor_3", type=int)
    args = ap.parse_args()

    motors = [args.motor_0, args.motor_1, args.motor_2, args.motor_3]
    for i, v in enumerate(motors):
        if not 0 <= v <= 0xFFFF:
            raise SystemExit(f"motor_{i} out of uint16 range: {v}")

    ser = serial.Serial(args.device, BAUD, timeout=0)
    seq = 0
    period = 1.0 / RATE_HZ
    try:
        while True:
            ser.write(build_frame(motors, seq))
            seq = (seq + 1) & 0x0F
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
