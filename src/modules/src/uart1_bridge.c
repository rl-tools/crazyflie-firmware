#define DEBUG_MODULE "U1BR"

#include <math.h>

#include "uart1_bridge.h"

#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"

#include "config.h"
#include "crtp.h"
#include "debug.h"
#include "log.h"
#include "param.h"
#include "static_mem.h"
#include "system.h"
#include "uart1.h"

#define CRTP_PORT_OFFBOARD_ARM 0x0E

#define UART1_BRIDGE_BAUDRATE        115200

#define UART1_BRIDGE_RX_TASK_NAME    "U1BR_RX"
#define UART1_BRIDGE_TX_TASK_NAME    "U1BR_TX"
#define UART1_BRIDGE_TASK_STACKSIZE  configMINIMAL_STACK_SIZE
#define UART1_BRIDGE_RX_TASK_PRI     2
#define UART1_BRIDGE_TX_TASK_PRI     1

#define TX_QUEUE_LENGTH              128

// Wire format: one start/flags byte with bit 7 set, followed by 12 7-bit
// payload bytes. The CRC is stored in raw bytes 8..9 and calculated over the
// full start/flags byte followed by raw bytes 0..7.
#define FRAME_DATA_BYTES             12
#define FRAME_RAW_BYTES              10
#define FRAME_START_MASK             0x80
#define FRAME_FLAGS_MASK             0x7F
#define FRAME_CRC_PAYLOAD_BYTES      9
#define OFFBOARD_FLAG_SELF_ACTIVATE  0x01
#define OFFBOARD_FLAGS_KNOWN_MASK    OFFBOARD_FLAG_SELF_ACTIVATE
#define OFFBOARD_TIMEOUT_MS          50
#define ARM_TIMEOUT_MS               200

// CF -> OpenMV attitude-setpoint frame:
// byte 0 has bit 7 set and type 0x02 in bits 0..6, followed by 13 bytes of
// 7-bit-packed payload. Raw payload is seq + four int16/uint16 fixed-point
// setpoint values + CRC16 over the start byte and raw bytes 0..8.
#define SETPOINT_FRAME_TYPE_ATTITUDE  0x02
#define SETPOINT_FRAME_START_BYTE     (FRAME_START_MASK | SETPOINT_FRAME_TYPE_ATTITUDE)
#define SETPOINT_RAW_BYTES            11
#define SETPOINT_DATA_BYTES           13
#define SETPOINT_FRAME_BYTES          (1 + SETPOINT_DATA_BYTES)
#define SETPOINT_CRC_PAYLOAD_BYTES    10
#define SETPOINT_SCALE                10000.0f

#define SETPOINT_MAX_TILT_RAD         0.5235987755982988f
#define SETPOINT_MAX_YAW_RATE_RAD_S   2.0f
#define SETPOINT_MIN_THRUST_G         0.4f
#define SETPOINT_MAX_THRUST_G         1.4f
#define SETPOINT_DEFAULT_THRUST_1G    39000.0f

static xQueueHandle txQueue;
STATIC_MEM_QUEUE_ALLOC(txQueue, TX_QUEUE_LENGTH, sizeof(uint8_t));

static void uart1BridgeRxTask(void *arg);
static void uart1BridgeTxTask(void *arg);
static void offboardArmPortHandler(CRTPPacket *pk);
STATIC_MEM_TASK_ALLOC(uart1BridgeRxTask, UART1_BRIDGE_TASK_STACKSIZE);
STATIC_MEM_TASK_ALLOC(uart1BridgeTxTask, UART1_BRIDGE_TASK_STACKSIZE);

static bool isInit = false;

static volatile uint8_t offboardArm = 0;
static volatile TickType_t lastArmPacketTick = 0;

static uint16_t latestPwm[4] = {0, 0, 0, 0};
static uint8_t latestFlags = 0;
static TickType_t lastFrameTick = 0;
static bool haveFrame = false;

static uint32_t framesOk = 0;
static uint32_t framesBadCrc = 0;
static uint32_t framesMsbErr = 0;
static uint32_t framesBadFlags = 0;

static bool directEngaged = false;
static bool selfActivated = false;
static bool frameFresh = false;
static bool armActive = false;
static bool activateOk = false;
static bool motorDividerOk = true;
static bool noHealthTest = true;
static bool supervisorAllowsMotors = true;

static float motorDivider = 20.0f;
static float setpointThrust1g = SETPOINT_DEFAULT_THRUST_1G;

static uint8_t setpointSeq = 0;
static uint32_t setpointFramesSent = 0;
static uint32_t setpointFramesDropped = 0;
static uint32_t setpointUnsupportedMode = 0;
static float latestSetpointRollRad = 0.0f;
static float latestSetpointPitchRad = 0.0f;
static float latestSetpointYawRateRad = 0.0f;
static float latestSetpointThrustG = 1.0f;

static uint16_t crc16_ccitt(const uint8_t *data, size_t n)
{
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int j = 0; j < 8; j++) {
      crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static void unpack7(const uint8_t in[FRAME_DATA_BYTES], uint8_t out[FRAME_RAW_BYTES])
{
  uint32_t acc = 0;
  int nbits = 0;
  int outIdx = 0;
  for (int i = 0; i < FRAME_DATA_BYTES; i++) {
    acc = (acc << 7) | (in[i] & 0x7F);
    nbits += 7;
    if (nbits >= 8 && outIdx < FRAME_RAW_BYTES) {
      nbits -= 8;
      out[outIdx++] = (uint8_t)((acc >> nbits) & 0xFF);
    }
  }
}

static void pack7(const uint8_t *in, size_t rawLen, uint8_t *out)
{
  uint32_t acc = 0;
  int nbits = 0;
  size_t w = 0;
  for (size_t i = 0; i < rawLen; i++) {
    acc = (acc << 8) | in[i];
    nbits += 8;
    while (nbits >= 7) {
      nbits -= 7;
      out[w++] = (uint8_t)((acc >> nbits) & 0x7F);
    }
  }
  if (nbits > 0) {
    out[w++] = (uint8_t)((acc << (7 - nbits)) & 0x7F);
  }
}

static float clampf_local(float v, float lo, float hi)
{
  if (v < lo) {
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return v;
}

static int16_t fixedPointS16(float v)
{
  float scaled = v * SETPOINT_SCALE;
  scaled = clampf_local(scaled, -32768.0f, 32767.0f);
  return (int16_t)(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

static uint16_t fixedPointU16(float v)
{
  float scaled = v * SETPOINT_SCALE;
  scaled = clampf_local(scaled, 0.0f, 65535.0f);
  return (uint16_t)(scaled + 0.5f);
}

static void putU16Be(uint8_t *dst, uint16_t v)
{
  dst[0] = (uint8_t)((v >> 8) & 0xFF);
  dst[1] = (uint8_t)(v & 0xFF);
}

static void projectTiltCone(float *roll, float *pitch)
{
  float sinRoll = sinf(*roll);
  float cosRoll = cosf(*roll);
  float sinPitch = sinf(*pitch);
  float cosPitch = cosf(*pitch);

  float worldZBodyX = -sinPitch;
  float worldZBodyY = cosPitch * sinRoll;
  float worldZBodyZ = cosPitch * cosRoll;

  const float cosTiltMax = cosf(SETPOINT_MAX_TILT_RAD);
  if (worldZBodyZ >= cosTiltMax) {
    return;
  }

  float horizontal = sqrtf(worldZBodyX * worldZBodyX + worldZBodyY * worldZBodyY);
  if (horizontal < 1.0e-6f) {
    *roll = 0.0f;
    *pitch = 0.0f;
    return;
  }

  const float sinTiltMax = sinf(SETPOINT_MAX_TILT_RAD);
  worldZBodyX *= sinTiltMax / horizontal;
  worldZBodyY *= sinTiltMax / horizontal;
  worldZBodyZ = cosTiltMax;

  *roll = atan2f(worldZBodyY, worldZBodyZ);
  *pitch = atan2f(-worldZBodyX, sqrtf(worldZBodyY * worldZBodyY + worldZBodyZ * worldZBodyZ));
}

static bool attitudeSetpointModeSupported(const setpoint_t *setpoint)
{
  return setpoint != NULL &&
         setpoint->mode.x == modeDisable &&
         setpoint->mode.y == modeDisable &&
         setpoint->mode.z == modeDisable &&
         setpoint->mode.roll == modeAbs &&
         setpoint->mode.pitch == modeAbs &&
         setpoint->mode.yaw == modeVelocity;
}

static bool uart1BridgeSendAllOrDrop(const uint8_t *data, size_t n)
{
  if (!isInit || data == NULL) {
    return false;
  }
  if (uxQueueSpacesAvailable(txQueue) < n) {
    return false;
  }

  for (size_t i = 0; i < n; i++) {
    if (xQueueSend(txQueue, &data[i], 0) != pdTRUE) {
      return false;
    }
  }
  return true;
}

static void applyFrame(uint8_t startByte, const uint8_t raw[FRAME_RAW_BYTES])
{
  uint16_t rxCrc = ((uint16_t)raw[8] << 8) | raw[9];
  uint8_t crcPayload[FRAME_CRC_PAYLOAD_BYTES];
  crcPayload[0] = startByte;
  for (int i = 0; i < 8; i++) {
    crcPayload[i + 1] = raw[i];
  }

  uint16_t exCrc = crc16_ccitt(crcPayload, FRAME_CRC_PAYLOAD_BYTES);
  if (rxCrc != exCrc) {
    framesBadCrc++;
    return;
  }

  uint8_t flags = startByte & FRAME_FLAGS_MASK;
  if (flags & ~OFFBOARD_FLAGS_KNOWN_MASK) {
    framesBadFlags++;
    return;
  }

  taskENTER_CRITICAL();
  latestPwm[0] = ((uint16_t)raw[0] << 8) | raw[1];
  latestPwm[1] = ((uint16_t)raw[2] << 8) | raw[3];
  latestPwm[2] = ((uint16_t)raw[4] << 8) | raw[5];
  latestPwm[3] = ((uint16_t)raw[6] << 8) | raw[7];
  latestFlags = flags;
  lastFrameTick = xTaskGetTickCount();
  haveFrame = true;
  taskEXIT_CRITICAL();

  framesOk++;
}

static void uart1BridgeRxTask(void *arg)
{
  systemWaitStart();

  uint8_t dataBuf[FRAME_DATA_BYTES];
  uint8_t startByte = 0;
  int dataIdx = -1;

  while (1) {
    char c;
    uart1Getchar(&c);
    uint8_t b = (uint8_t)c;

    if (b & FRAME_START_MASK) {
      startByte = b;
      dataIdx = 0;
      continue;
    }

    if (dataIdx < 0) {
      continue;
    }

    dataBuf[dataIdx++] = b;
    if (dataIdx == FRAME_DATA_BYTES) {
      uint8_t raw[FRAME_RAW_BYTES];
      unpack7(dataBuf, raw);
      applyFrame(startByte, raw);
      dataIdx = -1;
    }
  }
}

static void uart1BridgeTxTask(void *arg)
{
  systemWaitStart();

  while (1) {
    uint8_t byte;
    if (xQueueReceive(txQueue, &byte, portMAX_DELAY) == pdTRUE) {
      uart1SendData(1, &byte);
    }
  }
}

static void offboardArmPortHandler(CRTPPacket *pk)
{
  if (pk == NULL || pk->size < 1) {
    return;
  }
  lastArmPacketTick = xTaskGetTickCount();

  uint8_t newArm = pk->data[0] ? 1 : 0;
  if (newArm != offboardArm) {
    uart1BridgePrintf("[u1br] arm %s edge (%u -> %u)\r\n",
                      newArm ? "rising" : "falling",
                      (unsigned)offboardArm, (unsigned)newArm);
  }
  offboardArm = newArm;
}

void uart1BridgeInit(void)
{
  if (isInit) {
    return;
  }

  uart1Init(UART1_BRIDGE_BAUDRATE);

  txQueue = STATIC_MEM_QUEUE_CREATE(txQueue);

  crtpRegisterPortCB(CRTP_PORT_OFFBOARD_ARM, offboardArmPortHandler);

  STATIC_MEM_TASK_CREATE(uart1BridgeRxTask, uart1BridgeRxTask,
                         UART1_BRIDGE_RX_TASK_NAME, NULL,
                         UART1_BRIDGE_RX_TASK_PRI);
  STATIC_MEM_TASK_CREATE(uart1BridgeTxTask, uart1BridgeTxTask,
                         UART1_BRIDGE_TX_TASK_NAME, NULL,
                         UART1_BRIDGE_TX_TASK_PRI);

  isInit = true;
}

bool uart1BridgeSend(const uint8_t *data, size_t n)
{
  return uart1BridgeSendAllOrDrop(data, n);
}

int uart1BridgePutc(int c)
{
  uint8_t b = (uint8_t)c;
  if (!isInit) {
    return c;
  }
  xQueueSend(txQueue, &b, 0);
  return c;
}

void uart1BridgeSendAttitudeSetpoint(const setpoint_t *setpoint)
{
  float rollRad = 0.0f;
  float pitchRad = 0.0f;
  float yawRateRad = 0.0f;
  float thrustG = 1.0f;

  if (attitudeSetpointModeSupported(setpoint)) {
    rollRad = setpoint->attitude.roll * (float)M_PI / 180.0f;
    pitchRad = setpoint->attitude.pitch * (float)M_PI / 180.0f;
    yawRateRad = setpoint->attitudeRate.yaw * (float)M_PI / 180.0f;
    if (setpointThrust1g > 1.0f) {
      thrustG = setpoint->thrust / setpointThrust1g;
    }
  } else {
    setpointUnsupportedMode++;
  }

  if (!isfinite(rollRad)) {
    rollRad = 0.0f;
  }
  if (!isfinite(pitchRad)) {
    pitchRad = 0.0f;
  }
  if (!isfinite(yawRateRad)) {
    yawRateRad = 0.0f;
  }
  if (!isfinite(thrustG)) {
    thrustG = 1.0f;
  }

  projectTiltCone(&rollRad, &pitchRad);
  yawRateRad = clampf_local(yawRateRad, -SETPOINT_MAX_YAW_RATE_RAD_S, SETPOINT_MAX_YAW_RATE_RAD_S);
  thrustG = clampf_local(thrustG, SETPOINT_MIN_THRUST_G, SETPOINT_MAX_THRUST_G);

  uint8_t raw[SETPOINT_RAW_BYTES];
  uint8_t crcPayload[SETPOINT_CRC_PAYLOAD_BYTES];
  uint8_t frame[SETPOINT_FRAME_BYTES];

  raw[0] = setpointSeq++;
  putU16Be(&raw[1], (uint16_t)fixedPointS16(rollRad));
  putU16Be(&raw[3], (uint16_t)fixedPointS16(pitchRad));
  putU16Be(&raw[5], (uint16_t)fixedPointS16(yawRateRad));
  putU16Be(&raw[7], fixedPointU16(thrustG));

  crcPayload[0] = SETPOINT_FRAME_START_BYTE;
  for (int i = 0; i < 9; i++) {
    crcPayload[i + 1] = raw[i];
  }
  uint16_t crc = crc16_ccitt(crcPayload, SETPOINT_CRC_PAYLOAD_BYTES);
  putU16Be(&raw[9], crc);

  frame[0] = SETPOINT_FRAME_START_BYTE;
  pack7(raw, SETPOINT_RAW_BYTES, &frame[1]);

  if (uart1BridgeSendAllOrDrop(frame, sizeof(frame))) {
    setpointFramesSent++;
    latestSetpointRollRad = rollRad;
    latestSetpointPitchRad = pitchRad;
    latestSetpointYawRateRad = yawRateRad;
    latestSetpointThrustG = thrustG;
  } else {
    setpointFramesDropped++;
  }
}

static void refreshOutputConditions(void)
{
  const TickType_t now = xTaskGetTickCount();

  if (offboardArm &&
      ((now - lastArmPacketTick) >= M2T(ARM_TIMEOUT_MS))) {
    uart1BridgePrintf("[u1br] arm timeout (>%u ms), forcing 0\r\n",
                      (unsigned)ARM_TIMEOUT_MS);
    offboardArm = 0;
  }

  uint8_t flags;
  taskENTER_CRITICAL();
  frameFresh = haveFrame && ((now - lastFrameTick) < M2T(OFFBOARD_TIMEOUT_MS));
  flags = latestFlags;
  taskEXIT_CRITICAL();

  armActive = (offboardArm != 0);
  selfActivated = frameFresh && ((flags & OFFBOARD_FLAG_SELF_ACTIVATE) != 0);
  activateOk = armActive || selfActivated;
  motorDividerOk = (motorDivider > 0.0f);
}

void uart1BridgeSetOutputConditions(bool noHealthTestActive, bool supervisorAllows)
{
  noHealthTest = noHealthTestActive;
  supervisorAllowsMotors = supervisorAllows;
  refreshOutputConditions();

  if (!noHealthTest || !supervisorAllowsMotors) {
    directEngaged = false;
  }
}

void uart1BridgeApplyOverride(motors_thrust_pwm_t *motorPwm)
{
  if (!isInit || motorPwm == NULL) {
    directEngaged = false;
    return;
  }

  refreshOutputConditions();

  bool fresh;
  uint8_t flags;
  uint16_t pwm[4];
  taskENTER_CRITICAL();
  fresh = frameFresh;
  flags = latestFlags;
  pwm[0] = latestPwm[0];
  pwm[1] = latestPwm[1];
  pwm[2] = latestPwm[2];
  pwm[3] = latestPwm[3];
  taskEXIT_CRITICAL();

  bool selfActivate = selfActivated;
  bool engage = noHealthTest && supervisorAllowsMotors && fresh && activateOk;

  float divider = motorDivider;
  uint16_t scaledPwm[4];
  if (divider > 0.0f) {
    for (int i = 0; i < 4; i++) {
      float s = (float)pwm[i] / divider;
      if (s < 0.0f) s = 0.0f;
      if (s > 65535.0f) s = 65535.0f;
      scaledPwm[i] = (uint16_t)s;
    }
  } else {
    scaledPwm[0] = scaledPwm[1] = scaledPwm[2] = scaledPwm[3] = 0;
  }

  if (engage) {
    motorPwm->motors.m1 = scaledPwm[0];
    motorPwm->motors.m2 = scaledPwm[1];
    motorPwm->motors.m3 = scaledPwm[2];
    motorPwm->motors.m4 = scaledPwm[3];
  }

  directEngaged = engage;
  selfActivated = selfActivate;

  static uint32_t heartbeatCounter = 0;
  if (++heartbeatCounter >= 1000) {
    heartbeatCounter = 0;
    if (engage) {
      uart1BridgePrintf("[u1br] active arm=%u self=%u flags=0x%02x pwm=[%u %u %u %u] scaled=[%u %u %u %u] div=%d/1000\r\n",
                        (unsigned)armActive, (unsigned)selfActivate, (unsigned)flags,
                        (unsigned)pwm[0], (unsigned)pwm[1],
                        (unsigned)pwm[2], (unsigned)pwm[3],
                        (unsigned)scaledPwm[0], (unsigned)scaledPwm[1],
                        (unsigned)scaledPwm[2], (unsigned)scaledPwm[3],
                        (int)(divider * 1000.0f));
    } else {
      uart1BridgePrintf("[u1br] inactive (arm=%u self=%u fresh=%u flags=0x%02x)\r\n",
                        (unsigned)armActive, (unsigned)selfActivate,
                        (unsigned)fresh, (unsigned)flags);
    }
  }
}

PARAM_GROUP_START(u1br)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, engaged, &directEngaged)
PARAM_ADD(PARAM_FLOAT, motorDiv, &motorDivider)
PARAM_ADD(PARAM_FLOAT, thrust1g, &setpointThrust1g)
PARAM_GROUP_STOP(u1br)

LOG_GROUP_START(u1br)
LOG_ADD(LOG_UINT8,  noHealthTest, &noHealthTest)
LOG_ADD(LOG_UINT8,  supervisorOk, &supervisorAllowsMotors)
LOG_ADD(LOG_UINT8,  hasFrame, &haveFrame)
LOG_ADD(LOG_UINT8,  frameFresh, &frameFresh)
LOG_ADD(LOG_UINT8,  offboardArm, &offboardArm)
LOG_ADD(LOG_UINT8,  armActive, &armActive)
LOG_ADD(LOG_UINT8,  flags, &latestFlags)
LOG_ADD(LOG_UINT8,  selfActive, &selfActivated)
LOG_ADD(LOG_UINT8,  activateOk, &activateOk)
LOG_ADD(LOG_UINT8,  divOk, &motorDividerOk)
LOG_ADD(LOG_UINT8,  motorOutActive, &directEngaged)
LOG_ADD(LOG_UINT32, framesOk, &framesOk)
LOG_ADD(LOG_UINT32, framesBadCrc, &framesBadCrc)
LOG_ADD(LOG_UINT32, framesMsbErr, &framesMsbErr)
LOG_ADD(LOG_UINT32, framesBadFlags, &framesBadFlags)
LOG_ADD(LOG_UINT32, spSent, &setpointFramesSent)
LOG_ADD(LOG_UINT32, spDrop, &setpointFramesDropped)
LOG_ADD(LOG_UINT32, spBadMode, &setpointUnsupportedMode)
LOG_ADD(LOG_FLOAT, spRoll, &latestSetpointRollRad)
LOG_ADD(LOG_FLOAT, spPitch, &latestSetpointPitchRad)
LOG_ADD(LOG_FLOAT, spYawRate, &latestSetpointYawRateRad)
LOG_ADD(LOG_FLOAT, spThrustG, &latestSetpointThrustG)
LOG_ADD(LOG_UINT16, pwm0, &latestPwm[0])
LOG_ADD(LOG_UINT16, pwm1, &latestPwm[1])
LOG_ADD(LOG_UINT16, pwm2, &latestPwm[2])
LOG_ADD(LOG_UINT16, pwm3, &latestPwm[3])
LOG_GROUP_STOP(u1br)
