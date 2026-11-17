#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include "DFRobot_AirQualitySensor.h"
#include "Adafruit_LTR390.h"

// =====================================================
// KONFIGURASI SENSOR
// =====================================================

#define AIR_QUALITY_I2C_ADDRESS 0x19

DFRobot_AirQualitySensor particle(
  &Wire,
  AIR_QUALITY_I2C_ADDRESS
);

Adafruit_LTR390 ltr = Adafruit_LTR390();

// =====================================================
// KONFIGURASI WIFI
// =====================================================

const char* wifi_ssid = "<WIFI_SSID>";
const char* wifi_password = "<WIFI_PASSWORD>";

// =====================================================
// KONFIGURASI THINGSBOARD MQTT
// =====================================================

// IP Raspberry Pi, tanpa http:// dan tanpa port 8080
const char* mqtt_server = "<THINGSBOARD_HOST>";

// Port MQTT ThingsBoard
const uint16_t mqtt_port = 1883;

// Access Token device ThingsBoard
const char* mqtt_username = "<DEVICE_ACCESS_TOKEN>";

// Password dikosongkan untuk Access Token
const char* mqtt_password = "";

// Topic telemetry ThingsBoard
const char* telemetry_topic = "v1/devices/me/telemetry";

// =====================================================
// MQTT CLIENT
// =====================================================

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// =====================================================
// BUFFER WEATHER STATION
// Digunakan untuk Temperature, Humidity, BarPressure
// =====================================================

char databuffer[35];

// Waktu pengiriman
unsigned long lastPublishTime = 0;
const unsigned long publishInterval = 60000;   // 1 menit (60 * 1000 ms)

// =====================================================
// KONEKSI WIFI
// =====================================================

void connectWiFi() {
  Serial.println();
  Serial.print("Menghubungkan ke WiFi: ");
  Serial.println(wifi_ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(wifi_ssid, wifi_password);

  unsigned long startAttempt = millis();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    if (millis() - startAttempt > 30000) {
      Serial.println();
      Serial.println("Gagal terhubung WiFi. Restart koneksi...");

      WiFi.disconnect();
      delay(1000);
      WiFi.begin(wifi_ssid, wifi_password);

      startAttempt = millis();
    }
  }

  Serial.println();
  Serial.println("WiFi terhubung");

  Serial.print("IP ESP32: ");
  Serial.println(WiFi.localIP());

  Serial.print("Gateway: ");
  Serial.println(WiFi.gatewayIP());
}

// =====================================================
// KONEKSI MQTT THINGSBOARD
// =====================================================

void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Menghubungkan ke ThingsBoard MQTT...");

    String clientId = "ESP32-Hydro-";
    clientId += String(
      (uint32_t)(ESP.getEfuseMac() & 0xFFFFFFFF),
      HEX
    );

    bool connected = mqttClient.connect(
      clientId.c_str(),
      mqtt_username,
      mqtt_password
    );

    if (connected) {
      Serial.println(" berhasil");

      Serial.print("Client ID: ");
      Serial.println(clientId);
    } else {
      Serial.println(" gagal");

      Serial.print("MQTT state: ");
      Serial.println(mqttClient.state());

      Serial.println("Coba lagi dalam 5 detik...");
      delay(5000);
    }
  }
}

// =====================================================
// MEMBACA DATA WEATHER STATION DARI SERIAL2
// =====================================================

bool getWeatherBuffer() {
  int index = 0;
  unsigned long startTime = millis();

  while (index < 35) {
    if (millis() - startTime > 3000) {
      Serial.println("Timeout membaca weather station");
      return false;
    }

    if (Serial2.available()) {
      char receivedChar = Serial2.read();

      // Awal paket harus karakter 'c'
      if (index == 0 && receivedChar != 'c') {
        continue;
      }

      databuffer[index] = receivedChar;
      index++;
    }

    delay(1);
  }

  return true;
}

// =====================================================
// KONVERSI DATA WEATHER STATION
// =====================================================

int transCharToInt(
  char* buffer,
  int startIndex,
  int stopIndex
) {
  int result = 0;

  for (int i = startIndex; i <= stopIndex; i++) {
    if (buffer[i] >= '0' && buffer[i] <= '9') {
      result = result * 10 + (buffer[i] - '0');
    }
  }

  return result;
}

int transCharToTemperatureInt(char* buffer) {
  int result = 0;

  if (buffer[13] == '-') {
    result = -(
      ((buffer[14] - '0') * 10) +
      (buffer[15] - '0')
    );
  } else {
    result =
      ((buffer[13] - '0') * 100) +
      ((buffer[14] - '0') * 10) +
      (buffer[15] - '0');
  }

  return result;
}

float readTemperatureCelsius() {
  float temperatureFahrenheit =
    transCharToTemperatureInt(databuffer);

  return (temperatureFahrenheit - 32.0) * 5.0 / 9.0;
}

int readHumidity() {
  return transCharToInt(databuffer, 25, 26);
}

float readBarPressure() {
  return transCharToInt(databuffer, 28, 32) / 10.0;
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(9600);

  /*
    Jika ESP32 menggunakan pin RX2 dan TX2 default:
    RX2 = GPIO 16
    TX2 = GPIO 17
  */
  Serial2.begin(
    9600,
    SERIAL_8N1,
    16,
    17
  );

  Wire.begin();

  connectWiFi();

  mqttClient.setServer(
    mqtt_server,
    mqtt_port
  );

  // Memperbesar ukuran payload MQTT
  mqttClient.setBufferSize(512);

  // ===================================================
  // SENSOR PM
  // ===================================================

  Serial.println(
    "Menginisialisasi sensor kualitas udara..."
  );

  while (!particle.begin()) {
    Serial.println(
      "Sensor kualitas udara tidak ditemukan"
    );
    delay(1000);
  }

  Serial.println(
    "Sensor kualitas udara berhasil diinisialisasi"
  );

  // ===================================================
  // SENSOR UV LTR390
  // ===================================================

  Serial.println("Menginisialisasi LTR390...");

  if (!ltr.begin()) {
    Serial.println("LTR390 tidak ditemukan");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("LTR390 berhasil ditemukan");

  ltr.setMode(LTR390_MODE_UVS);
  ltr.setGain(LTR390_GAIN_3);
  ltr.setResolution(LTR390_RESOLUTION_16BIT);
  ltr.setThresholds(100, 1000);
  ltr.configInterrupt(true, LTR390_MODE_UVS);
}

// =====================================================
// LOOP
// =====================================================

void loop() {
  // Periksa WiFi
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Periksa MQTT
  if (!mqttClient.connected()) {
    connectMQTT();
  }

  mqttClient.loop();

  // Kirim setiap 1 menit
  if (millis() - lastPublishTime < publishInterval) {
    return;
  }

  lastPublishTime = millis();

  // ===================================================
  // BACA WEATHER STATION
  // ===================================================

  bool weatherDataValid = getWeatherBuffer();

  float temperature = 0.0;
  int humidity = 0;
  float barPressure = 0.0;

  if (weatherDataValid) {
    temperature = readTemperatureCelsius();
    humidity = readHumidity();
    barPressure = readBarPressure();
  } else {
    Serial.println(
      "Data weather station tidak tersedia"
    );
  }

  // ===================================================
  // BACA SENSOR PARTICULATE MATTER
  // ===================================================

  uint16_t pm25 =
    particle.gainParticleConcentration_ugm3(
      PARTICLE_PM2_5_STANDARD
    );

  uint16_t pm10 =
    particle.gainParticleConcentration_ugm3(
      PARTICLE_PM1_0_STANDARD
    );

  uint16_t pm100 =
    particle.gainParticleConcentration_ugm3(
      PARTICLE_PM10_STANDARD
    );

  // ===================================================
  // BACA SENSOR UV
  // ===================================================

  uint32_t uvData = 0;

  if (ltr.newDataAvailable()) {
    uvData = ltr.readUVS();
  }

  // ===================================================
  // BUAT JSON
  // ===================================================

  String payload = "{";

  payload += "\"PM2_5\":";
  payload += String(pm25);
  payload += ",";

  payload += "\"PM1_0\":";
  payload += String(pm10);
  payload += ",";

  payload += "\"PM10\":";
  payload += String(pm100);
  payload += ",";

  payload += "\"Temperature\":";
  payload += String(temperature, 2);
  payload += ",";

  payload += "\"Humidity\":";
  payload += String(humidity);
  payload += ",";

  payload += "\"BarPressure\":";
  payload += String(barPressure, 2);
  payload += ",";

  payload += "\"UV\":";
  payload += String(uvData);

  payload += "}";

  // ===================================================
  // TAMPILKAN DI SERIAL MONITOR
  // ===================================================

  Serial.println();
  Serial.println("Data telemetry:");
  Serial.println(payload);

  // ===================================================
  // KIRIM KE THINGSBOARD
  // ===================================================

  bool publishSuccess = mqttClient.publish(
    telemetry_topic,
    payload.c_str()
  );

  if (publishSuccess) {
    Serial.println(
      "Data berhasil dikirim ke ThingsBoard"
    );
  } else {
    Serial.println(
      "Data gagal dikirim ke ThingsBoard"
    );

    Serial.print("MQTT state: ");
    Serial.println(mqttClient.state());
  }
}
