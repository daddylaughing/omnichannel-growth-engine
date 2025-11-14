@echo off
echo ==============================================
echo   GIT AUTO PUSH SCRIPT - OMNICHANNEL PROJECT
echo ==============================================
echo.

cd "C:\Users\inchr\Downloads\Capstone Associate Data Analyst\omnichannel-growth-engine"

echo Adding all files...
git add -A

echo.
echo Committing changes...
git commit -m "Auto-push: full project update"

echo.
echo Pulling latest from GitHub...
git pull origin main --rebase

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo ==============================================
echo   DONE! Everything pushed to GitHub.
echo ==============================================
pause
