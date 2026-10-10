<#
.SYNOPSIS
    Set up local-music: a fully local instrumental music generator (Stable Audio 3 Medium)
    with automatic checks, on this machine's NVIDIA GPU.

.DESCRIPTION
    Clones the Stable Audio 3 inference code at a pinned commit into engine\, builds .venv\
    with a CUDA 12.8 build of PyTorch, and fetches the weights with checksum verification
    into models\. Nothing here uploads anything.

    Re-running is safe: every finished step is skipped and downloads resume.

    Stable Audio 3 is gated on Hugging Face: before installing, accept its licence at
    https://huggingface.co/stabilityai/stable-audio-3-medium and run `hf auth login`
    (or `huggingface-cli login`) yourself. This script never touches your token.

    Works on Windows PowerShell 5.1 and PowerShell 7+.

.PARAMETER AcceptModelLicenses
    Accept the model weights' licences (Stable Audio 3: Stability AI Community License;
    the checkers: Apache-2.0 and CC-BY-4.0). Required to download them; without it the
    installer lists them and stops.

.PARAMETER PythonVersion
    Python version for the venv, passed to the 'py' launcher (3.10 or newer).

.PARAMETER CudaIndex
    PyTorch wheel index. cu128 is the oldest that supports RTX 50-series (Blackwell) cards.

.PARAMETER ReuseModels
    The models\ folder of another install (of local-music, or of local-sounds: they share
    the Stable Audio 3 text encoder and the Audiobox Aesthetics checker). Matching weights are
    hard-linked (or copied, across drives) instead of downloaded, then checksum-verified. A
    local-sounds next to this folder is found on its own. Files already in your Hugging Face
    cache are reused the same way, automatically.

.PARAMETER WithSmall
    Also download Stable Audio 3 Small-Music (2.3 GB): a lighter, lower-quality engine.

.PARAMETER SkipModels
    Build the environment only; fetch the weights later with scripts\download_models.py.

.PARAMETER NoSkillLink
    Don't link this folder into ~\.claude\skills\local-music (the Claude Code skill).

.EXAMPLE
    .\install.ps1 -AcceptModelLicenses

.EXAMPLE
    .\install.ps1 -AcceptModelLicenses -ReuseModels D:\old\local-music\models -WithSmall
#>
[CmdletBinding()]
param(
    [string] $PythonVersion = '3.13',
    [string] $CudaIndex     = 'cu128',
    [switch] $AcceptModelLicenses,
    [string] $ReuseModels   = '',
    [switch] $WithSmall,
    [switch] $SkipModels,
    [switch] $NoSkillLink,
    [switch] $NoPipCache
)

$ErrorActionPreference = 'Stop'

$RepoRoot = $PSScriptRoot
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }

# The engine's inference code, at the commit this repo was built and tested against.
$SA3_REPO   = 'https://github.com/Stability-AI/stable-audio-3.git'
$SA3_COMMIT = '3a82c807b69cf4b7c5c05270011a5d5e47abac18'
$SA3_TORCH  = '2.7.1'   # what the stable-audio-3 library pins (with cu128 wheels instead of cu126)

function Step ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "    $m"   -ForegroundColor Green }
function Note ($m) { Write-Host "    $m"   -ForegroundColor DarkGray }
function Warn ($m) { Write-Host "    $m"   -ForegroundColor Yellow }
function Fail ($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

# Windows PowerShell 5.1 turns any stderr output from a native command into a
# terminating error while $ErrorActionPreference is 'Stop'; probes go through here.
function Probe {
    param([Parameter(Mandatory)][string] $Exe, [string[]] $Arguments = @())
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = (& $Exe @Arguments 2>&1 | Out-String)
        [pscustomobject]@{ Code = $LASTEXITCODE; Out = $out.Trim() }
    } catch {
        [pscustomobject]@{ Code = -1; Out = $_.Exception.Message }
    } finally {
        $ErrorActionPreference = $prev
    }
}

function LastLine ($s) { (($s -split "`r?`n") | Select-Object -Last 1).Trim() }

$pipFlags = @('--disable-pip-version-check')
if ($NoPipCache) { $pipFlags += '--no-cache-dir' }

# --- licences: before anything heavy ----------------------------------------
if (-not $SkipModels -and -not $AcceptModelLicenses) {
    Write-Host @"

  local-music downloads these models. Read their licences first:

    Stable Audio 3 Medium   Stability AI Community License: free for non-commercial use, and
                            for commercial use by individuals and organisations under US`$1M
                            a year in revenue (registration required); larger ones need an
                            Enterprise licence. Not OSI open source. Gated on Hugging Face.
                            https://huggingface.co/stabilityai/stable-audio-3-medium
                            Its T5Gemma text encoder: Gemma Terms of Use.
    CLAP                    Apache-2.0, used only to score tracks against their prompt
                            https://huggingface.co/laion/larger_clap_general
    Audiobox Aesthetics     CC-BY-4.0, used only to score production quality
                            https://huggingface.co/facebook/audiobox-aesthetics

  This repo's own code is MIT; see LICENSE-MODELS.md. To accept the model licences and
  install, re-run with -AcceptModelLicenses (or -SkipModels to set up without weights).

"@ -ForegroundColor Yellow
    exit 1
}

# --- prerequisites --------------------------------------------------------
Step 'Checking prerequisites'
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail 'git is not on PATH. Install Git for Windows: https://git-scm.com/download/win'
}
Ok "git $((git --version) -replace 'git version ')"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Fail "The 'py' launcher is not on PATH. Install Python from python.org with the launcher option ticked."
}
$probe = Probe 'py' @("-$PythonVersion", '-c', 'import sys; sys.stdout.write(sys.executable)')
if ($probe.Code -ne 0) {
    Fail "Python $PythonVersion not found. Installed:`n$((Probe 'py' @('--list')).Out)`n       Pass -PythonVersion with one of those (3.10 or newer)."
}
Ok "Python $PythonVersion at $($probe.Out)"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Ok "GPU: $((Probe 'nvidia-smi' @('--query-gpu=name,memory.total', '--format=csv,noheader')).Out)"
} else {
    Warn 'nvidia-smi not found. The engine needs an NVIDIA GPU.'
}
$drive = Get-PSDrive -Name ($RepoRoot.Substring(0, 1)) -ErrorAction SilentlyContinue
if ($drive) {
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Ok "$freeGB GB free"
    if ($freeGB -lt 25) { Warn 'Under 25 GB free. Weights are ~11.6 GB, the venv ~6 GB, plus pip cache.' }
}

# --- engine code ----------------------------------------------------------
function Clone-Pinned ($url, $dest, $commit) {
    if (Test-Path (Join-Path $dest '.git')) {
        $head = (Probe 'git' @('-C', $dest, 'rev-parse', 'HEAD')).Out
        if ($head -eq $commit) { Ok "Already at $($commit.Substring(0, 10))"; return }
        Note "At $head; checking out the pinned commit"
        git -C $dest fetch --quiet origin
        git -C $dest checkout --quiet $commit
        if ($LASTEXITCODE -ne 0) { Fail "could not check out $commit in $dest (local changes? move it aside)" }
        return
    }
    if (Test-Path $dest) { Fail "$dest exists but is not a git checkout. Move it aside." }
    New-Item -ItemType Directory -Path (Split-Path $dest -Parent) -Force | Out-Null
    git clone --quiet $url $dest
    if ($LASTEXITCODE -ne 0) { Fail "git clone $url failed" }
    git -C $dest checkout --quiet $commit
    if ($LASTEXITCODE -ne 0) { Fail "could not check out $commit in $dest" }
    Ok 'Cloned'
}

$sa3Code = Join-Path $RepoRoot 'engine\stable-audio-3'
Step 'Stable Audio 3 inference code (engine\stable-audio-3)'
Clone-Pinned $SA3_REPO $sa3Code $SA3_COMMIT

# --- venv -------------------------------------------------------------------
Step 'Python environment (.venv): server, checks and the engine'
$venv = Join-Path $RepoRoot '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    & py "-$PythonVersion" -m venv $venv
    if (-not (Test-Path $py)) { Fail '.venv creation failed' }
    Ok '.venv created'
}
$have = Probe $py @('-c', 'import torch;print(torch.__version__)')
if ($have.Code -eq 0 -and (LastLine $have.Out) -eq "$SA3_TORCH+$CudaIndex") {
    Ok "torch $(LastLine $have.Out) already installed"
} else {
    Note "Installing torch $SA3_TORCH from the $CudaIndex index (~3 GB download unless cached)..."
    & $py -m pip install --quiet @pipFlags --upgrade pip
    & $py -m pip install @pipFlags --index-url "https://download.pytorch.org/whl/$CudaIndex" "torch==$SA3_TORCH" "torchaudio==$SA3_TORCH"
    if ($LASTEXITCODE -ne 0) { Fail 'torch install failed' }
}
# A constraints file keeps every later install from swapping the CUDA build for a CPU one.
$constraints = Join-Path $env:TEMP 'local-music-constraints.txt'
Set-Content -Path $constraints -Value @("torch==$SA3_TORCH+$CudaIndex", "torchaudio==$SA3_TORCH+$CudaIndex") -Encoding ascii
# --no-deps: the library pins torch==2.7.1 without a CUDA index on Windows.
& $py -m pip install --quiet @pipFlags --no-deps -e $sa3Code
if ($LASTEXITCODE -ne 0) { Fail 'stable-audio-3 install failed' }
& $py -m pip install --quiet @pipFlags -c $constraints -r (Join-Path $RepoRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { Fail 'dependency install failed' }
if ((Probe $py @('-W', 'ignore', '-c', 'import stable_audio_3')).Code -ne 0) { Fail 'stable_audio_3 does not import' }
$t = Probe $py @('-W', 'ignore', '-c', 'import torch;print(torch.__version__, torch.cuda.is_available())')
$last = LastLine $t.Out
if ($last -notmatch '\+cu\S* True$') { Note $t.Out; Fail ".venv has no working CUDA torch ($last)" }
Ok ".venv ready (torch $last)"
# FlashAttention 2 isn't needed: without it the engine computes the autoencoder's
# sliding-window attention with chunked SDPA, which gives the same result.

# --- weights --------------------------------------------------------------
$modelsOk = $true
if ($SkipModels) {
    Step 'Skipping model download (-SkipModels)'
    Note "Fetch them later with:  & `"$py`" scripts\download_models.py"
} else {
    $gb = if ($WithSmall) { '~13.9' } else { '~11.6' }
    Step "Model weights ($gb GB, checksum-verified)"
    Note 'Resumable - if this is interrupted, just re-run install.ps1.'
    Note 'Stable Audio 3 is gated: accept its licence on Hugging Face and run `hf auth login` first.'
    $dlArgs = @((Join-Path $RepoRoot 'scripts\download_models.py'))
    if ($WithSmall) { $dlArgs += '--with-optional' }
    $reuse = @()
    if ($ReuseModels) {
        if (-not (Test-Path $ReuseModels)) { Fail "-ReuseModels: $ReuseModels does not exist" }
        $reuse += (Resolve-Path $ReuseModels).Path
    }
    $sibling = Join-Path (Split-Path $RepoRoot -Parent) 'local-sounds\models'
    if (Test-Path $sibling) {
        Note "Found local-sounds next to this folder: its matching weights are reused (checksums decide)."
        $reuse += (Resolve-Path $sibling).Path
    }
    foreach ($r in $reuse) { $dlArgs += @('--reuse', $r) }
    & $py @dlArgs
    if ($LASTEXITCODE -ne 0) {
        $modelsOk = $false
        Warn 'Some weights are missing (see above). Fix that and re-run install.ps1; the rest is done.'
    }
}

# --- Claude Code skill ------------------------------------------------------
if (-not $NoSkillLink) {
    Step 'Claude Code skill (~\.claude\skills\local-music)'
    $skills = Join-Path $env:USERPROFILE '.claude\skills'
    $link = Join-Path $skills 'local-music'
    if (Test-Path $link) {
        $target = (Get-Item -LiteralPath $link).Target
        if ($target -and ((@($target)[0].TrimEnd('\')) -ne $RepoRoot.TrimEnd('\'))) {
            Warn "$link points to $(@($target)[0]), not here. Leaving it; remove it to link this copy."
        } else {
            Ok 'Already there'
        }
    } else {
        New-Item -ItemType Directory -Path $skills -Force | Out-Null
        New-Item -ItemType Junction -Path $link -Target $RepoRoot | Out-Null
        Ok "Linked $link -> $RepoRoot"
    }
}

Step 'Done'
Write-Host @"

  Start it with:      .\start.cmd
  Then open:          http://127.0.0.1:8192/
  Agents use:         python scripts\music.py "warm lo-fi hip hop, dusty piano, 84 BPM" --out <folder>

  Stable Audio 3 is under the Stability AI Community License. See LICENSE-MODELS.md.

"@ -ForegroundColor Green
if (-not $modelsOk) { exit 1 }
