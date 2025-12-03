# Push to GitHub Instructions

## Option 1: Install Git and Push via Command Line

### 1. Install Git
Download and install Git from: https://git-scm.com/download/win

### 2. Open PowerShell and run these commands:

```powershell
cd C:\Users\athul\Documents\wonderla\live_detection_app

# Initialize git repository
git init

# Add remote repository
git remote add origin https://github.com/Athulv1/wonderla.git

# Add all files (videos will be excluded by .gitignore)
git add .

# Commit changes
git commit -m "Add head counter system with zone-based tracking"

# Push to GitHub
git push -u origin main
```

If you get an error about 'main' vs 'master', try:
```powershell
git branch -M main
git push -u origin main
```

### 3. If prompted for credentials:
- Username: Athulv1
- Password: Use a Personal Access Token (not your GitHub password)
  - Generate token at: https://github.com/settings/tokens

## Option 2: Use GitHub Desktop (Easier)

### 1. Download GitHub Desktop
https://desktop.github.com/

### 2. Sign in with your GitHub account

### 3. Add repository:
- File → Add Local Repository
- Choose: C:\Users\athul\Documents\wonderla\live_detection_app
- Click "create a repository"

### 4. Publish:
- Click "Publish repository"
- Uncheck "Keep this code private" if you want it public
- Click "Publish repository"

## Files That Will Be Pushed

✅ Python scripts:
- head_counter_zones.py
- head_counter.py
- setup_zones.py
- setup_roi_head_counter.py
- inference.py
- app.py

✅ Configuration:
- head_counter_config.json
- roi_config.json
- .gitignore

✅ Documentation:
- README_HEAD_COUNTER.md
- HEAD_COUNTER_GUIDE.md
- All other .md files

✅ Requirements:
- head_counter_requirements.txt
- requirements.txt
- flask_requirements.txt

✅ Templates:
- templates/index.html

✅ Scripts:
- start.bat
- run_head_counter.bat

❌ Files EXCLUDED by .gitignore:
- *.mp4 (all videos)
- *.avi, *.mov, *.mkv
- venv/ (virtual environment)
- __pycache__/
- *.pt (model weights)
- outputs/
- uploads/

## Verify What Will Be Pushed

After adding files, you can check what will be pushed:
```powershell
git status
```

This shows all tracked files (videos will be excluded).
