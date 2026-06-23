#requires -Version 5.1
# Transcribe everything in the "in" folder -> results land in "out".
# Speaker diarization is ON (needs hf_token.txt). GPU (RTX 3070), large-v3-turbo.

$ErrorActionPreference = "Stop"
$root = "C:\Users\djgra\Transcribe"

# Make bundled ffmpeg/ffprobe visible to whisply for this session.
$env:Path = "$root\bin;$env:Path"

# Windows blocks HuggingFace's default symlink cache without admin/Dev Mode.
# Tell it to copy instead, or model downloads fail with WinError 1314.
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Auto-load our startup patch (fixes a Windows speechbrain bug that breaks
# speaker diarization). See pylib\sitecustomize.py for details.
$env:PYTHONPATH = "$root\pylib"

$whisply   = "$root\.venv\Scripts\whisply.exe"
$inDir     = "$root\in"
$outDir    = "$root\out"
$tokenFile = "$root\hf_token.txt"

# Any audio/video to process?
$files = Get-ChildItem -Path $inDir -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -match '\.(mp3|wav|m4a|aac|flac|ogg|opus|wma|mp4|mov|mkv|webm|avi|m4v)$' }
if (-not $files) {
    Write-Host "No audio/video files found in $inDir" -ForegroundColor Yellow
    Write-Host "Drop files there and run this again." -ForegroundColor Yellow
    Read-Host "Press Enter to close"; exit 0
}

# Speaker diarization (pyannote, legacy path) needs a HuggingFace token.
# Read from .env (HF_TOKEN=...), falling back to the old hf_token.txt.
$token = ""
$envFile = "$root\.env"
if (Test-Path $envFile) {
    $line = (Get-Content $envFile | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1)
    if ($line) { $token = ($line -replace '^\s*HF_TOKEN\s*=', '').Trim().Trim('"') }
}
if (-not $token -and (Test-Path $tokenFile)) { $token = (Get-Content $tokenFile -Raw).Trim() }
if (-not $token) {
    Write-Host "No HuggingFace token found." -ForegroundColor Red
    Write-Host "Copy .env.example to .env and set HF_TOKEN=... (see README)." -ForegroundColor Red
    Write-Host "(Only the legacy pyannote path needs this - the app needs no token.)" -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}

Write-Host ("Transcribing {0} file(s) on GPU with speaker labels..." -f $files.Count) -ForegroundColor Cyan
# NOTE: settings passed as CLI flags, not via -c config.json. Whisply has a bug
# where combining -c with -f double-wraps the file list and crashes.
& $whisply run -f $inDir -o $outDir -d gpu -m large-v3-turbo -a -hf $token -e all
$code = $LASTEXITCODE
Write-Host ""

if ($code -eq 0) {
    Write-Host "Done. Results are in: $outDir" -ForegroundColor Green

    # Archive the source files so the in\ folder is clear for next time WITHOUT
    # ever deleting anything. Files are MOVED into archive\<timestamp>\.
    # We never overwrite (timestamped folder is unique) and never delete inputs.
    $stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $archiveDir = Join-Path "$root\archive" $stamp
    New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null
    $moved = 0
    foreach ($f in (Get-ChildItem -Path $inDir -File -ErrorAction SilentlyContinue)) {
        # whisply's *_converted.wav is a throwaway regenerated from the source -
        # delete it (Graham's instruction), archive only the original source.
        if ($f.Name -like '*_converted.wav') {
            Remove-Item -LiteralPath $f.FullName -Force
            continue
        }
        $dest = Join-Path $archiveDir $f.Name
        if (Test-Path -LiteralPath $dest) { continue }   # never clobber
        Move-Item -LiteralPath $f.FullName -Destination $dest
        $moved++
    }
    if ($moved -gt 0) {
        Write-Host "Archived $moved source file(s) to: $archiveDir" -ForegroundColor Green
        Write-Host "(your originals are preserved there, not deleted)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "whisply exited with code $code." -ForegroundColor Yellow
    Write-Host "Input files were left in $inDir (not archived)." -ForegroundColor Yellow
}
Read-Host "Press Enter to close"
