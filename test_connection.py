import serial
import time

# REPLACE WITH YOUR PORT
PORT = 'COM3' 
BAUD = 9600

print(f"Attempting to connect to {PORT}...")

try:
    # 1. Open Connection
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    print("SUCCESS: Port opened!")
    
    # 2. Wait for Arduino Auto-Reset (Crucial step!)
    print("Waiting 2 seconds for Arduino to boot...")
    time.sleep(2)
    
    # 3. Send a Signal
    print("Sending 'D' command...")
    arduino.write(b'D')
    
    # 4. Listen for Reply
    time.sleep(1)
    if arduino.in_waiting > 0:
        response = arduino.readline().decode('utf-8').strip()
        print(f"ARDUINO REPLIED: {response}")
    else:
        print("NO REPLY (But connection is open)")
        
    arduino.close()
    
except serial.SerialException as e:
    print(f"\nERROR: Could not connect.")
    print(f"Details: {e}")