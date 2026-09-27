param([string]$OutputDirectory = 'dist/fp16-checks')
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
    foreach ($fallback in @($false, $true)) {
        $name = if ($fallback) { 'fallback' } else { 'dispatch' }
        $flags = @('/nologo', '/O2', '/fp:precise', '/Ivendor', '/DSQLITE_VEC_ENABLE_F16C')
        if ($fallback) { $flags += @('/DSQLITE_VEC_TEST', '/DSQLITE_VEC_TEST_NO_F16C') }
        & cl.exe @flags tests/test-fp16-kernels.c vendor/sqlite3.c "/Fo:$OutputDirectory/" "/Fe:$OutputDirectory/$name.exe"
        if ($LASTEXITCODE -ne 0) { throw "Failed to compile $name checks" }
        & "./$OutputDirectory/$name.exe"
        if ($LASTEXITCODE -ne 0) { throw "$name checks failed" }
    }
} finally { Pop-Location }
