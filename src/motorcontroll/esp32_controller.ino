// Copyright (c) 2026 陳發達_楊瑋竣
// Tatung University — I4210 AI實務專題

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SERVOMIN  150
#define SERVOMAX  600

// 各類別對應的 PCA9685 馬達通道
const int CH_PET = 0;
const int CH_GLS = 1;
const int CH_MTL = 2;
const int CH_PAP = 3;

// 各通道對應的 LED GPIO
const int LED_PET = 23;  // 通道 0 → GPIO 23
const int LED_GLS = 5;   // 通道 1 → GPIO 5
const int LED_MTL = 4;   // 通道 2 → GPIO 4
const int LED_PAP = 2;   // 通道 3 → GPIO 2

#define ANGLE_PET  120
#define ANGLE_GLS  120
#define ANGLE_MTL  120
#define ANGLE_PAP  120

bool busy = false;

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void runSorter(int channel, int angle, int ledPin, const char* label) {
  Serial.print("【收到】");
  Serial.print(label);
  Serial.print(" → 通道 ");
  Serial.print(channel);
  Serial.println(" 馬達動作中");

  // 馬達執行時 LED 亮
  digitalWrite(ledPin, HIGH);
  pwm.setPWM(channel, 0, angleToPulse(angle));
  delay(5000);

  // 馬達回到 0 度，LED 熄滅
  pwm.setPWM(channel, 0, angleToPulse(0));
  digitalWrite(ledPin, LOW);

  // 清空 buffer 中累積的多餘指令
  while (Serial.available()) Serial.read();

  // 回傳結果給 Jetson
  Serial.print("DONE:");
  Serial.println(label);
}

void setup() {
  Serial.begin(115200);

  // 初始化四個 LED
  pinMode(LED_PET, OUTPUT); digitalWrite(LED_PET, LOW);
  pinMode(LED_GLS, OUTPUT); digitalWrite(LED_GLS, LOW);
  pinMode(LED_MTL, OUTPUT); digitalWrite(LED_MTL, LOW);
  pinMode(LED_PAP, OUTPUT); digitalWrite(LED_PAP, LOW);

  Wire.begin(21, 22);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  delay(10);

  // 四個通道都歸零
  pwm.setPWM(CH_PET, 0, angleToPulse(0));
  pwm.setPWM(CH_GLS, 0, angleToPulse(0));
  pwm.setPWM(CH_MTL, 0, angleToPulse(0));
  pwm.setPWM(CH_PAP, 0, angleToPulse(0));

  Serial.println("Ready. Waiting for commands from Jetson...");
}

void loop() {
  if (busy || !Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if      (cmd == "PET") { busy = true; runSorter(CH_PET, ANGLE_PET, LED_PET, "PET bottle");  busy = false; }
  else if (cmd == "GLS") { busy = true; runSorter(CH_GLS, ANGLE_GLS, LED_GLS, "Glass bottle"); busy = false; }
  else if (cmd == "MTL") { busy = true; runSorter(CH_MTL, ANGLE_MTL, LED_MTL, "Aluminum can"); busy = false; }
  else if (cmd == "PAP") { busy = true; runSorter(CH_PAP, ANGLE_PAP, LED_PAP, "Tetra Pak");    busy = false; }
}
