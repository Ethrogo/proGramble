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

$nodeExe = Join-Path $nodePath.FullName "node.exe"
$nextCli = Join-Path $PSScriptRoot "node_modules\next\dist\bin\next"
$npmCli = Join-Path $nodePath.FullName "node_modules\npm\bin\npm-cli.js"

if (-not (Test-Path $nodeExe)) {
    Write-Error "Local node executable not found at $nodeExe."
    exit 1
}

if (-not (Test-Path $npmCli)) {
    Write-Error "Local npm CLI not found at $npmCli."
    exit 1
}

$env:Path = "$($nodePath.FullName);$env:Path"

if (-not $NpmArgs -or $NpmArgs.Count -eq 0) {
    $NpmArgs = @("run", "dev")
}

if ($NpmArgs.Count -ge 2 -and $NpmArgs[0] -eq "run" -and $NpmArgs[1] -in @("dev", "build", "start")) {
    $subcommand = $NpmArgs[1]
    $remaining = if ($NpmArgs.Count -gt 2) { $NpmArgs[2..($NpmArgs.Count - 1)] } else { @() }
    & $nodeExe $nextCli $subcommand @remaining
} else {
    & $nodeExe $npmCli @NpmArgs
}
exit $LASTEXITCODE
