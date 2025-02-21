import subprocess
import time

def run_services():
    # Start video processor
    video_process = subprocess.Popen(['python', 'yolov5/video_processor.py'], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    # Wait for video processor to initialize
    time.sleep(5)
    
    # Start Flask app
    app_process = subprocess.Popen(['python', 'examguard/app.py'],
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    try:
        video_process.wait()
        app_process.wait()
    except KeyboardInterrupt:
        video_process.terminate()
        app_process.terminate()

if __name__ == '__main__':
    run_services()