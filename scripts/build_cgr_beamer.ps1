param(
    [string]$Source = "docs/presentation/cgr_v3_2_project_beamer.tex"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourcePath = Join-Path $ProjectRoot $Source
$TempDir = Join-Path $ProjectRoot "tmp/pdfs/cgr_v3_2_project_beamer"
$OutputDir = Join-Path $ProjectRoot "output/pdf"
$RenderDir = Join-Path $TempDir "rendered"

if (-not (Test-Path -LiteralPath $SourcePath)) {
    throw "Beamer source not found: $SourcePath"
}

New-Item -ItemType Directory -Force -Path $TempDir, $OutputDir, $RenderDir | Out-Null

$XeLaTeX = Get-Command xelatex -ErrorAction SilentlyContinue
$Tectonic = Get-Command tectonic -ErrorAction SilentlyContinue
$PortableTectonic = Join-Path $ProjectRoot "tmp/tooling/tectonic-0.16.9/tectonic.exe"
$TectonicPath = if ($Tectonic) { $Tectonic.Source } else { $null }

if ($XeLaTeX) {
    for ($Pass = 1; $Pass -le 2; $Pass++) {
        & $XeLaTeX.Source `
            -interaction=nonstopmode `
            -halt-on-error `
            -file-line-error `
            -output-directory=$TempDir `
            $SourcePath
        if ($LASTEXITCODE -ne 0) {
            throw "XeLaTeX pass $Pass failed. See $TempDir/cgr_v3_2_project_beamer.log"
        }
    }
} else {
    if (-not $Tectonic -and (Test-Path -LiteralPath $PortableTectonic)) {
        $TectonicPath = $PortableTectonic
        $env:TECTONIC_CACHE_DIR = Join-Path $ProjectRoot "tmp/tooling/tectonic-cache"
    }
    if (-not $TectonicPath) {
        throw "No TeX engine is available. Install XeLaTeX or Tectonic."
    }

    $StagedSource = Join-Path $TempDir "cgr_v3_2_project_beamer.tex"
    $FontDir = Join-Path $TempDir "fonts"
    New-Item -ItemType Directory -Force -Path $FontDir | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $StagedSource -Force

    $WindowsFonts = "C:/Windows/Fonts"
    foreach ($Font in "NotoSansSC-VF.ttf", "NotoSerifSC-VF.ttf") {
        $FontPath = Join-Path $WindowsFonts $Font
        if (Test-Path -LiteralPath $FontPath) {
            Copy-Item -LiteralPath $FontPath -Destination $FontDir -Force
        }
    }

    Push-Location $TempDir
    try {
        & $TectonicPath `
            "cgr_v3_2_project_beamer.tex" `
            --outdir $TempDir `
            --keep-logs `
            --reruns 1
        if ($LASTEXITCODE -ne 0) {
            throw "Tectonic failed. See $TempDir/cgr_v3_2_project_beamer.log"
        }
    } finally {
        Pop-Location
    }
}

$BuiltPdf = Join-Path $TempDir "cgr_v3_2_project_beamer.pdf"
$FinalPdf = Join-Path $OutputDir "cgr_v3_2_project_beamer.pdf"
Copy-Item -LiteralPath $BuiltPdf -Destination $FinalPdf -Force

$PdfToPpm = Get-Command pdftoppm -ErrorAction SilentlyContinue
$Rendered = $false
if ($PdfToPpm) {
    & $PdfToPpm.Source -png -r 150 $FinalPdf (Join-Path $RenderDir "page")
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "PDF rendering was skipped because pdftoppm failed. The PDF build itself succeeded."
    } else {
        $Rendered = $true
    }
}

Write-Host "Built: $FinalPdf"
if ($Rendered) {
    Write-Host "Rendered pages: $RenderDir"
}
