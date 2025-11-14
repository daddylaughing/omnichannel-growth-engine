@echo off
setlocal ENABLEDELAYEDEXPANSION

echo =====================================================
echo   OMNICHANNEL GROWTH ENGINE - ONE CLICK GIT SYNC
echo   Author : Derrick Wong
echo   Action : add + commit + pull (rebase) + push
echo =====================================================
echo.

REM === CONFIG: CHANGE ONLY IF YOU MOVE THE FOLDER OR RENAME REPO ===
set "REPO_DIR=C:\Users\inchr\Downloads\Capstone Associate Data Analyst\omnichannel-growth-engine"
set "REPO_URL=https://github.com/daddylaughing/omnichannel-growth-engine"
REM ================================================================

REM --- Check Git is available ---
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed or not in PATH.
    echo    Please install Git from https://git-scm.com/downloads
    echo.
    pause
    exit /b 1
)

REM --- Go to repo directory ---
echo 📁 Moving to repository folder:
echo     %REPO_DIR%
cd /d "%REPO_DIR%" 2>nul
if errorlevel 1 (
    echo ❌ Could not find the folder. Check REPO_DIR in this .bat file.
    echo.
    pause
    exit /b 1
)

echo.
echo 🔍 Current branch and status:
git branch --show-current
git status -sb
echo.

REM --- Stage all changes ---
echo 🔄 Staging all files and folders (git add -A)...
git add -A

REM --- Create commit message with date + time ---
for /f "tokens=1-3 delims=/ " %%a in ("%date%") do (
    set TODAY=%%a-%%b-%%c
)
set NOW=%time:~0,8%
set "COMMIT_MSG=Auto-update full project - %TODAY% %NOW%"

echo.
echo 📝 Committing: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%" >nul 2>&1

if errorlevel 1 (
    echo ⚠️ No new changes to commit (or commit failed). Continuing anyway...
) else (
    echo ✅ Commit created successfully.
)

REM --- Pull latest from origin/main with rebase + autostash ---
echo.
echo ⬇️ Pulling latest changes from origin/main (rebase + autostash)...
git pull origin main --rebase --autostash
if errorlevel 1 (
    echo ❌ git pull --rebase failed.
    echo    This usually means there is a merge conflict.
    echo    Open Git Bash/Anaconda Prompt and resolve manually.
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Pull (rebase) completed.
)

REM --- Push to origin/main ---
echo.
echo ⬆️ Pushing to origin/main...
git push origin main
if errorlevel 1 (
    echo ❌ Push failed.
    echo    Most common reasons:
    echo      - Network issue
    echo      - Credentials/token expired
    echo.
    pause
    exit /b 1
) else (
    echo ✅ Push successful! All files and folders are now on GitHub.
)

echo.
echo 🌐 Opening GitHub repository in your browser...
start "" "%REPO_URL%"

echo.
echo =====================================================
echo   🎉 DONE! Your project is synced with GitHub.
echo   Repo: %REPO_URL%
echo =====================================================
echo.
pause
endlocal
