Start-Process python -ArgumentList "yolov5/video_processor.py"
Start-Sleep -Seconds 5
Start-Process python -ArgumentList "examguard/app.py"