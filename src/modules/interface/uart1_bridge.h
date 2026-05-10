#ifndef UART1_BRIDGE_H
#define UART1_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "eprintf.h"
#include "stabilizer_types.h"

void uart1BridgeInit(void);

bool uart1BridgeSend(const uint8_t *data, size_t n);

void uart1BridgeApplyOverride(motors_thrust_pwm_t *motorPwm);

void uart1BridgeSetOutputConditions(bool noHealthTest, bool supervisorAllowsMotors);

void uart1BridgeSendAttitudeSetpoint(const setpoint_t *setpoint);

int uart1BridgePutc(int c);

#define uart1BridgePrintf(FMT, ...) eprintf(uart1BridgePutc, FMT, ## __VA_ARGS__)

#endif
