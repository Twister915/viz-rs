# build-release.ps1 - turnkey Windows release build for vis-rs.
#
# Why this exists:
#   * SDL2 is compiled from source and statically linked (see the sdl2
#     'bundled' + 'static-link' features in Cargo.toml), so there is NO
#     SDL2.dll to find. That source build runs cmake, which ships inside
#     Visual Studio but is not on the default PATH - so this script loads the
#     VS x64 build environment first (cl.exe + cmake + ninja).
#   * FFTW still links dynamically. Its DLL is downloaded from the official
#     FFTW project during the build but lands in a cargo build subfolder, so
#     this script copies it next to the exe so the binary actually runs.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\build-release.ps1

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# --- 1. Locate Visual Studio and load its x64 build environment -------------
$pf86 = ${env:ProgramFiles(x86)}
$vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio 2022 with the 'Desktop development with C++' workload (provides cl.exe, cmake and ninja)."
}

$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) {
    throw "No Visual Studio install with the C++ toolchain was found. Install the 'Desktop development with C++' workload."
}

$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

Write-Host "Loading VS build environment from: $vsPath" -ForegroundColor Cyan

# Run vcvars64.bat in a child cmd, then import the resulting environment back
# into this PowerShell session so cargo (and the cmake/cc build scripts) see it.
$cmdLine = '"' + $vcvars + '" >nul 2>&1 && set'
$envDump = & cmd.exe /c $cmdLine
foreach ($line in $envDump) {
    $i = $line.IndexOf('=')
    if ($i -gt 0) {
        $name = $line.Substring(0, $i)
        $value = $line.Substring($i + 1)
        Set-Item -Path "Env:\$name" -Value $value
    }
}

# --- 2. Build --------------------------------------------------------------
Write-Host "Building release (compiles + statically links SDL2 from source)..." -ForegroundColor Cyan
Push-Location $root
try {
    & cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

# --- 3. Stage the FFTW DLL next to the exe ---------------------------------
$exeDir = Join-Path $root 'target\release'
$buildDir = Join-Path $exeDir 'build'
$fftwDll = Get-ChildItem -Path $buildDir -Recurse -Filter 'libfftw3-3.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($fftwDll) {
    Copy-Item $fftwDll.FullName -Destination $exeDir -Force
    Write-Host ("Copied " + $fftwDll.Name + " next to vis-rs.exe") -ForegroundColor Cyan
}
else {
    Write-Warning "libfftw3-3.dll not found under target\release\build - the exe may fail to start."
}

Write-Host ""
Write-Host "Done. Run it with:" -ForegroundColor Green
Write-Host "    .\target\release\vis-rs.exe path\to\song.wav"
