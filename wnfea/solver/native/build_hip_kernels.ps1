# Build WNFEA Architecture-Tuned HIP Kernels DLL for AMD Radeon RX 7800 XT (gfx1101)
$ErrorActionPreference = "Stop"

$workspaceRoot = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$rocmClang = "C:\Program Files\AMD\ROCm\7.1\bin\clang++.exe"
$patchDir = "$workspaceRoot\rocm_patch"
$sourceFile = "$PSScriptRoot\hip_kernels.cpp"
$outputDll = "$PSScriptRoot\wnfea_hip_kernels.dll"

Write-Host "================================================================="
Write-Host " Compiling WNFEA Architecture-Tuned HIP Kernels (gfx1101)"
Write-Host " Compiler: $rocmClang"
Write-Host " Target:   gfx1101 (AMD Radeon RX 7800 XT / RDNA 3)"
Write-Host " Patch:    $patchDir"
Write-Host "================================================================="

$args = @(
    "-shared",
    "-isystem", "$patchDir",
    "-D_USE_MATH_DEFINES",
    "-O3",
    "-DNDEBUG",
    "-std=gnu++17",
    "-x", "hip",
    "--offload-arch=gfx1101",
    "-Rpass-analysis=kernel-resource-usage",
    "$sourceFile",
    "-o", "$outputDll"
)

& $rocmClang $args

if ($LASTEXITCODE -eq 0 -and (Test-Path $outputDll)) {
    $dllSize = (Get-Item $outputDll).Length
    Write-Host "`n[SUCCESS] Compiled $outputDll ($dllSize bytes)"
} else {
    Write-Error "[FAILURE] Compilation failed with exit code $LASTEXITCODE"
}
