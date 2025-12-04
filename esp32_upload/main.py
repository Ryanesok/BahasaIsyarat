import network
import time
from machine import I2C, Pin
from umqtt.simple import MQTTClient
from mpu6050 import accel

# ==========================================
# KONFIGURASI WIFI (GANTI INI!)
# ==========================================
SSID = "Ryanesok"
PASSWORD = ""

# ==========================================
# KONFIGURASI MQTT
# ==========================================
MQTT_BROKER = "broker.hivemq.com"
CLIENT_ID = "esp32-kelompok-4-sensor"
TOPIC = "pameran/gerakan"

# ==========================================
# SETUP PIN & SENSOR
# ==========================================
# I2C untuk ESP32 standar: SDA=Pin 21, SCL=Pin 22
i2c = I2C(0, scl=Pin(22), sda=Pin(21))
sensor = accel(i2c)

# LED Indikator (Biasanya Pin 2 untuk onboard, atau sesuaikan jika pakai eksternal)
led = Pin(2, Pin.OUT) 

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Menghubungkan ke WiFi...')
        wlan.connect(SSID, PASSWORD)
        while not wlan.isconnected():
            led.value(not led.value()) # Kedip saat mencoba konek
            time.sleep(0.5)
    print('WiFi Terhubung:', wlan.ifconfig())
    led.value(1) # Nyala terus jika sudah konek WiFi

def connect_mqtt():
    try:
        client = MQTTClient(CLIENT_ID, MQTT_BROKER)
        client.connect()
        print('✓ Terhubung ke MQTT Broker:', MQTT_BROKER)
        print('✓ Topic:', TOPIC)
        # Blink LED to indicate connection
        for _ in range(3):
            led.value(0)
            time.sleep(0.1)
            led.value(1)
            time.sleep(0.1)
        return client
    except Exception as e:
        print('✗ Gagal konek MQTT:', e)
        return None

def main():
    print("="*40)
    print("ESP32 - Gesture Control System")
    print("Kelompok 4")
    print("="*40)
    
    connect_wifi()
    client = connect_mqtt()
    
    if not client:
        print("✗ MQTT gagal, sistem akan restart dalam 5 detik...")
        time.sleep(5)
        import machine
        machine.reset()
    
    # Kalibrasi manual / Threshold
    # Sesuaikan angka ini jika sensor terlalu sensitif atau kurang sensitif
    # Nilai AcX berkisar antara -16000 sampai 16000
    BATAS_MIRING = 7000  
    
    last_msg_time = 0
    COOLDOWN = 1500  # Jeda 1.5 detik antar gestur agar tidak spam
    
    # Rate limiting untuk data streaming
    PUBLISH_RATE = 30  # Hz (30 kali per detik)
    publish_interval = 1000 // PUBLISH_RATE  # milliseconds
    last_publish_time = 0
    
    # WiFi reconnect tracking
    wifi_check_interval = 10000  # Check every 10 seconds
    last_wifi_check = 0
    
    # Error counter untuk auto-restart
    error_count = 0
    MAX_ERRORS = 10

    print("\n✓ Sistem Siap!")
    print("Instruksi:")
    print("  - Miring KANAN   = SPACE (Spasi)")
    print("  - Miring KIRI    = BACKSPACE (Hapus)")
    print("  - Miring DEPAN   = CLEAR (Reset)")
    print(f"  - Rate Limit: {PUBLISH_RATE} Hz")
    print("="*40)

    while True:
        try:
            now = time.ticks_ms()
            
            # ===== WiFi AUTO-RECONNECT =====
            if time.ticks_diff(now, last_wifi_check) > wifi_check_interval:
                wlan = network.WLAN(network.STA_IF)
                if not wlan.isconnected():
                    print("⚠ WiFi terputus! Reconnecting...")
                    connect_wifi()
                    client = connect_mqtt()
                last_wifi_check = now
            
            # ===== BACA SENSOR DENGAN ERROR HANDLING =====
            try:
                data = sensor.get_values()
                ax = data["AcX"] # Kemiringan Kiri/Kanan
                ay = data["AcY"] # Kemiringan Depan/Belakang
                az = data["AcZ"] # Kemiringan Atas/Bawah
                
                # Reset error counter on successful read
                error_count = 0
                
            except OSError as e:
                # I2C timeout or bus error
                print(f"⚠ I2C Error: {e}")
                error_count += 1
                
                if error_count >= MAX_ERRORS:
                    print("✗ Terlalu banyak error I2C, restart sistem...")
                    time.sleep(2)
                    import machine
                    machine.reset()
                
                time.sleep(0.1)
                continue
            
            # ===== RATE-LIMITED DATA PUBLISHING =====
            # Publish sensor data stream untuk sensor fusion
            if time.ticks_diff(now, last_publish_time) >= publish_interval:
                if client:
                    try:
                        # Kirim data sensor mentah sebagai JSON untuk sensor fusion
                        import ujson
                        sensor_data = ujson.dumps({
                            "AcX": ax,
                            "AcY": ay,
                            "AcZ": az,
                            "timestamp": now
                        })
                        # Optional: uncomment untuk streaming data ke desktop
                        # client.publish(TOPIC, sensor_data)
                        last_publish_time = now
                    except Exception as e:
                        print(f"⚠ Publish error: {e}")
            
            # ===== GESTURE DETECTION =====
            # Cek apakah masa tenggang (cooldown) sudah lewat
            if time.ticks_diff(now, last_msg_time) > COOLDOWN:
                
                pesan = ""
                
                # --- LOGIKA DETEKSI ---
                
                # 1. MIRING KANAN -> SPASI
                if ax > BATAS_MIRING:
                    pesan = "SPACE"
                    print(">> Deteksi: KANAN (Kirim SPACE)")
                    
                # 2. MIRING KIRI -> HAPUS
                elif ax < -BATAS_MIRING:
                    pesan = "BACKSPACE"
                    print(">> Deteksi: KIRI (Kirim BACKSPACE)")
                    
                # 3. MIRING DEPAN -> RESET
                # Jika akselerasi Y miring ke depan (positif)
                elif ay > BATAS_MIRING: 
                    pesan = "CLEAR"
                    print(">> Deteksi: DEPAN (Kirim CLEAR)")
                
                # 4. MIRING BELAKANG -> RESET (alternatif)
                # Uncomment jika ingin miring belakang juga bisa clear
                # elif ay < -BATAS_MIRING: 
                #     pesan = "CLEAR"
                #     print(">> Deteksi: BELAKANG (Kirim CLEAR)")
                
                # --- KIRIM DATA ---
                if pesan != "":
                    if client:
                        try:
                            client.publish(TOPIC, pesan)
                            last_msg_time = now  # Reset cooldown
                            
                            # Kedip LED cepat sebagai tanda data terkirim
                            for _ in range(2):
                                led.value(0)
                                time.sleep(0.05)
                                led.value(1)
                                time.sleep(0.05)
                        except Exception as e:
                            print(f"⚠ MQTT publish error: {e}")
                            client = connect_mqtt()
                    else:
                        print("⚠ Client MQTT putus, mencoba reconnect...")
                        client = connect_mqtt()

            time.sleep(0.01)  # 10ms loop (100 Hz sampling)

        except KeyboardInterrupt:
            print("\n✓ Program dihentikan oleh user")
            break
            
        except Exception as e:
            print(f"✗ Error Loop: {e}")
            error_count += 1
            
            if error_count >= MAX_ERRORS:
                print("✗ Terlalu banyak error, restart sistem...")
                time.sleep(2)
                import machine
                machine.reset()
            
            time.sleep(0.5)

if __name__ == "__main__":
    main()