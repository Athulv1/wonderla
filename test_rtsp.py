import cv2

# Different RTSP URL formats to try
urls = [
    "rtsp://Testing:Test@1234#@10.196.211.60:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://Testing:Test%401234%23@10.196.211.60:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://Testing:Test@10.196.211.60:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://10.196.211.60:554/cam/realmonitor?channel=1&subtype=0",
    "rtsp://Testing:Test@1234#@10.196.211.60/cam/realmonitor?channel=1&subtype=0",
]

for i, url in enumerate(urls, 1):
    print(f"\nTrying URL {i}: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ SUCCESS! Connected and received frame: {frame.shape}")
            cap.release()
            break
        else:
            print("✗ Opened but no frame received")
    else:
        print("✗ Failed to open stream")
    cap.release()
