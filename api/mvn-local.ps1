param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MavenArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$jdkRoot = Join-Path $repoRoot ".local-jdks"
$jdkPath = Get-ChildItem -Path $jdkRoot -Directory -Filter "jdk-21*" |
    Sort-Object Name -Descending |
    Select-Object -First 1

if (-not $jdkPath) {
    Write-Error "No local Java 21 install found under $jdkRoot."
    exit 1
}

$env:JAVA_HOME = $jdkPath.FullName
$env:Path = "$env:JAVA_HOME\bin;$env:Path"

if (-not $MavenArgs -or $MavenArgs.Count -eq 0) {
    $MavenArgs = @("spring-boot:run")
}

& "$PSScriptRoot\mvnw.cmd" @MavenArgs
exit $LASTEXITCODE
