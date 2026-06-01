param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$NpmArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$nodeRoot = Join-Path $repoRoot ".local-node"
$nodePath = Get-ChildItem -Path $nodeRoot -Directory -Filter "node-v*-win-x64" |
    Sort-Object Name -Descending |
    Select-Object -First 1

if (-not $nodePath) {
    Write-Error "No local Node.js install found under $nodeRoot."
    exit 1
}

$env:Path = "$($nodePath.FullName);$env:Path"

if (-not $NpmArgs -or $NpmArgs.Count -eq 0) {
    $NpmArgs = @("run", "dev")
}

& "npm.cmd" @NpmArgs
exit $LASTEXITCODE
