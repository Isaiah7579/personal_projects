# Installs the background sync (sync_cache.py) as "RS GE Sync": builds .venv\Scripts\RS GE Sync.exe
# (build_sync_exe.py), registers a scheduled task that starts it at logon (restarted if it crashes), adds a
# Start menu shortcut, and starts it. Safe to re-run, e.g. after uv upgrades Python. Run from this folder:
#     powershell -ExecutionPolicy Bypass -File install_sync_task.ps1
# Uninstall:
#     Unregister-ScheduledTask -TaskName "RS GE Sync" -Confirm:$false
#     Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\RS GE Sync.lnk"

$ErrorActionPreference = 'Stop'
$AppName = 'RS GE Sync'
$Here = $PSScriptRoot
$Exe = Join-Path $Here ".venv\Scripts\$AppName.exe"
$Script = Join-Path $Here 'sync_cache.py'

# Stop whatever is running now (older installs used a different task name) so the exe can be replaced
foreach ($name in @('RS GE cache sync', $AppName)) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $name
        if ($name -ne $AppName) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
    }
}
Get-CimInstance Win32_Process -Filter "Name like 'python%' or Name = '$AppName.exe'" |
    Where-Object { $_.CommandLine -like '*sync_cache.py*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

uv sync --project $Here
uv run --project $Here (Join-Path $Here 'build_sync_exe.py')
if ($LASTEXITCODE -ne 0) { throw 'build_sync_exe.py failed' }

$action = New-ScheduledTaskAction -Execute $Exe -Argument "`"$Script`" --background" -WorkingDirectory $Here
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 5) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew -Priority 7
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $AppName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description 'Keeps the GE price cache in runescape/data up to date (sync_cache.py).' -Force | Out-Null

$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut(
    (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$AppName.lnk"))
$shortcut.TargetPath = $Exe
$shortcut.Arguments = "`"$Script`""
$shortcut.WorkingDirectory = $Here
$shortcut.IconLocation = "$Exe,0"
$shortcut.Description = 'GE price cache sync (tray icon: status, pause, quit)'
$shortcut.Save()

Start-ScheduledTask -TaskName $AppName
Write-Host "Installed '$AppName': starts at logon, Start menu shortcut added, running now (tray icon)."
Write-Host "Progress: uv run sync_cache.py --status   Log: data\sync.log"
