@echo off
echo ============================================================
echo HEAD COUNTER - Setup and Run
echo ============================================================
echo.

REM Set video path
set VIDEO_PATH=C:\Users\athul\Documents\wonderla\WAVEPOOL TEST CAMERA_2025-12-01_03-55-00-2025-12-01_04-05-59.mp4

echo Step 1: Setup ROI Line (Interactive)
echo ============================================================
echo This will open a window to draw your counting line
echo Press [H] for horizontal, [V] for vertical, [C] for custom
echo Click to draw, [S] to save, [Q] to quit
echo.
pause

python setup_roi_head_counter.py --video "%VIDEO_PATH%"

echo.
echo ============================================================
echo Step 2: Run Head Counter
echo ============================================================
echo Processing video with head detection and counting...
echo.

python head_counter.py --video "%VIDEO_PATH%" --output "wavepool_head_count.mp4" --show --conf 0.3 --save-json "wavepool_results.json"

echo.
echo ============================================================
echo DONE!
echo ============================================================
echo Output video: wavepool_head_count.mp4
echo Results JSON: wavepool_results.json
echo.
pause
