import paho.mqtt.client as mqtt
import time
import random

BROKER = "broker.hivemq.com"
TOPIC = "pameran/gerakan"

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected to broker RC : {rc}")
    
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.connect(BROKER, 1883, 60)
client.loop_start()

print("==== MOCK ESP32 CONTROLLER ====")
print("Press button to send sensor signal :")
print("1. SPACE [Tilt Right]")
print("2. BACKSPACE [Tilt Left]")
print("3. CLEAR [Shake]")
print("4. ESCAPE [Exit]")

try:
    while True:
        cmd = input("Enter command (1/2/3): ")
        if cmd == '1':
            client.publish(TOPIC, "SPACE")
            print("Sent: SPACE")
        elif cmd == '2':
            client.publish(TOPIC, "BACKSPACE")
            print("Sent: BACKSPACE")
        elif cmd == '3':
            client.publish(TOPIC, "SHAKE")
            print("Sent: SHAKE")
        elif cmd == '4':
            print("Exiting...")
            break
        else:
            print("Invalid command. Please enter 1, 2, 3, or 4.")
except KeyboardInterrupt: 
    print("\nExiting...")
    client.loop_stop()
    client.disconnect()