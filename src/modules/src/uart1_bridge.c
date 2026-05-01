#define DEBUG_MODULE "U1BR"

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
  if (!isInit || data == NULL) {
    return false;
  }

  for (size_t i = 0; i < n; i++) {
    if (xQueueSend(txQueue, &data[i], 0) != pdTRUE) {
      return false;
    }
  }
  return true;
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
LOG_ADD(LOG_UINT16, pwm0, &latestPwm[0])
LOG_ADD(LOG_UINT16, pwm1, &latestPwm[1])
LOG_ADD(LOG_UINT16, pwm2, &latestPwm[2])
LOG_ADD(LOG_UINT16, pwm3, &latestPwm[3])
LOG_GROUP_STOP(u1br)
