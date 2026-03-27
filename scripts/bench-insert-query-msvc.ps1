# Build and run the insert+query benchmark using MSVC on Windows.
# Usage:
#   .\scripts\bench-insert-query-msvc.ps1                    # scalar (no SIMD)
#   .\scripts\bench-insert-query-msvc.ps1 -Avx               # with AVX2
#   .\scripts\bench-insert-query-msvc.ps1 -Threads            # with multithreaded KNN
#   .\scripts\bench-insert-query-msvc.ps1 -Avx -Threads       # AVX2 + threads

param(
    [switch]$Avx,
    [switch]$Threads
)

$ErrorActionPreference = "Stop"

$repoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoDir

$simdFlags = @()
if ($Avx) {
    $simdFlags = @("/DSQLITE_VEC_ENABLE_AVX", "/arch:AVX2")
    Write-Host "Building with AVX2 enabled..."
}

$threadFlags = @()
if ($Threads) {
    $threadFlags = @("/DSQLITE_VEC_ENABLE_THREADS")
    Write-Host "Building with threading enabled..."
}

if (-not (Test-Path dist)) { New-Item -ItemType Directory -Path dist | Out-Null }

# Generate sqlite-vec.h from template if it doesn't exist
if (-not (Test-Path "sqlite-vec.h")) {
    $version = (Get-Content "VERSION").Trim()
    $parts = $version -split '[-.]'
    $major = $parts[0]
    $minor = $parts[1]
    $patch = $parts[2]
    $content = (Get-Content "sqlite-vec.h.tmpl" -Raw)
    $content = $content -replace '\$\{VERSION\}', $version
    $content = $content -replace '\$\{DATE\}', (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
    $content = $content -replace '\$\{SOURCE\}', (git rev-parse HEAD 2>$null)
    $content = $content -replace '\$\{VERSION_MAJOR\}', $major
    $content = $content -replace '\$\{VERSION_MINOR\}', $minor
    $content = $content -replace '\$\{VERSION_PATCH\}', $patch
    Set-Content "sqlite-vec.h" $content
    Write-Host "Generated sqlite-vec.h"
}

# Find and import VS environment
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvars)) {
    Write-Error "Could not find vcvars64.bat at $vcvars"
    exit 1
}

# Import MSVC environment into PowerShell
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# Compile
$clArgs = @(
    "/nologo", "/O2",
    "/DSQLITE_CORE"
) + $simdFlags + $threadFlags + @(
    "benchmarks\bench-insert-query.c", "sqlite-vec.c", "vendor\sqlite3.c",
    "/I.", "/Ivendor",
    "/Fe:dist\bench-insert-query.exe",
    "/Fodist\"
)

Write-Host "Compiling benchmark..."
& cl @clArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Compilation failed"
    exit 1
}

Write-Host ""
Write-Host "Running benchmark..."
& .\dist\bench-insert-query.exe
if ($LASTEXITCODE -ne 0) {
    Write-Error "Benchmark failed"
    exit 1
}
