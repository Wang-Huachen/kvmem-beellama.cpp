# A/B experiment: does the DRAFT(Follower) cache format decide the
# "cannot remove uncommitted MTP rows" failure?
#
# Hypothesis (from code reading, see STATUS.md):
#   the multimodal checkpoint/rollback path asks the draft context to remove a
#   mid-sequence range. A plain (row) draft cache can do that; a KVarN draft
#   cache cannot ("KVarN can only remove a complete sequence or the current/
#   previous fp16 tail groups"), so the removal fails and the server logs
#   "cannot remove uncommitted MTP rows".
#
# Case 0 -> draft cache = KVarN  (expect the error to REPRODUCE)
# Case 1 -> draft cache = q8_0   (expect the error to DISAPPEAR)
# Everything else is identical between the two cases.
#
# Usage:
#   pwsh -File scripts\nightly\kvmem-draft-cache-ab.ps1 -Case 0
#   pwsh -File scripts\nightly\kvmem-draft-cache-ab.ps1 -Case 1

param(
    [string]$Exe    = "",
    [string]$Model  = "",
    [string]$Mmproj = "",
    [int]$Case = 0,
    [int]$LongChars = 150000,
    [int]$Port = 8485,
    [int]$ReadyTimeoutSec = 150,
    [int]$RequestTimeoutSec = 600
)

$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
# Defaults: the server binary comes from the repo tree, and the model/mmproj come from
# the environment, so no machine-specific paths are stored in the repository.
if (-not $Exe)    { $Exe    = Join-Path $root 'build-win\bin\llama-kvmem-server.exe' }
if (-not $Model)  { $Model  = $env:KVMEM_TEST_MODEL }
if (-not $Mmproj) { $Mmproj = $env:KVMEM_TEST_MMPROJ }
if (-not (Test-Path $Exe) -or -not $Model -or -not (Test-Path $Model) -or -not $Mmproj -or -not (Test-Path $Mmproj)) {
    Write-Host 'need the built server plus a test model and its mmproj:'
    Write-Host '  -Exe <llama-kvmem-server.exe>   (default: <repo>\build-win\bin\llama-kvmem-server.exe)'
    Write-Host '  -Model <model.gguf> -Mmproj <mmproj.gguf>, or $env:KVMEM_TEST_MODEL / $env:KVMEM_TEST_MMPROJ'
    exit 2
}
$logdir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log  = Join-Path $logdir ("draft-ab-case{0}.log" -f $Case)
$outp = Join-Path $logdir ("draft-ab-case{0}.out" -f $Case)
Remove-Item $log, $outp -ErrorAction SilentlyContinue

$draftArgs = if ($Case -eq 0) { @('--spec-kv-dtype','kvarn6') }
             else             { @('--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0') }
$draftName = if ($Case -eq 0) { 'kvarn6 (KVarN) - the server must REFUSE this' } else { 'q8_0 (plain rows)' }
# case 2 exercises the retrieval layout on a hybrid target: a small pool forces
# retrieval + slot reordering (which case 0/1 with a large pool never reaches).
$budget = if ($Case -eq 2) { '12288' } else { '36864' }
$genReserve = if ($Case -eq 2) { '4096' } else { '16384' }

# Long synthetic prompt: ~4 chars/token, so 150000 chars is roughly 37k tokens.
$para = 'KVMem keeps a bounded GPU working set of KV blocks and re-materialises the blocks that a retrieval score selects, while the draft context speculates ahead. '
$sb = New-Object System.Text.StringBuilder
while ($sb.Length -lt $LongChars) { [void]$sb.Append($para) }
$longPrompt = $sb.ToString()

$args = @('-m',$Model,'--ctx-size','262144','--image-min-tokens','1024','--no-mmproj-offload','--mmproj',$Mmproj,
          '--n-gpu-layers','99','--spec-type','draft-mtp','--spec-draft-n-max','3','--spec-draft-p-min','0.0',
          '--jinja','--enable-thinking','--flash-attn','on','--load-mode','none','--temp','0.1','--top-p','0.95',
          '--kvmem-budget',$budget,'--kvmem-gen-reserve',$genReserve,'--kvmem-block-tokens','32','--kv-dtype','kvarn6',
          '--port',"$Port",'--alias','draft-ab','--host','127.0.0.1','--timeout','300') + $draftArgs

Write-Host ("=== case {0}: draft cache = {1} ===" -f $Case, $draftName)
Write-Host ("  exe : {0}" -f $Exe)
Write-Host ("  log : {0}" -f $log)

$p = $null
$verdict = 'UNKNOWN'
try {
    $p = Start-Process -FilePath $Exe -ArgumentList $args -RedirectStandardError $log -RedirectStandardOutput $outp -PassThru -NoNewWindow
    Write-Host ("  started pid={0}; waiting for readiness (max {1}s)" -f $p.Id, $ReadyTimeoutSec)
    $ready = $false
    for ($i = 1; $i -le [int]($ReadyTimeoutSec / 2); $i++) {
        Start-Sleep -Seconds 2
        if ($p.HasExited) { break }
        if ((Get-Content $log -Raw -ErrorAction SilentlyContinue) -match 'listening on') { $ready = $true; break }
    }
    if ($p.HasExited) {
        Write-Host ("  FAILED TO START (exit={0})" -f $p.ExitCode)
        Get-Content $log -Tail 6 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ("    " + $_) }
        $verdict = 'START-FAIL'
    } elseif (-not $ready) {
        Write-Host "  NOT READY within timeout"
        $verdict = 'READY-TIMEOUT'
    } else {
        Write-Host "  ready"
        $uri = "http://127.0.0.1:$Port/v1/chat/completions"
        function Send-Chat([string]$text, [int]$maxTok) {
            $json = @{ messages = @(@{ role = 'user'; content = $text }); max_tokens = $maxTok; temperature = 0.1; stream = $false } | ConvertTo-Json -Depth 6 -Compress
            $bytes = [Text.Encoding]::UTF8.GetBytes($json)
            return Invoke-RestMethod -Uri $uri -Method Post -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec $RequestTimeoutSec
        }
        Write-Host "  request 1: long prompt (this is the slow one)"
        try { $r1 = Send-Chat $longPrompt 64; Write-Host ("    ok, content_chars={0} reasoning_chars={1}" -f $r1.choices[0].message.content.Length, $(if ($r1.choices[0].message.reasoning_content) { $r1.choices[0].message.reasoning_content.Length } else { 0 })) }
        catch { Write-Host ("    request 1 failed: {0}" -f $_.Exception.Message) }
        Write-Host "  request 2: SAME long prompt plus a small suffix (forces a cached-prefix reuse -> the rollback path)"
        try { $r2 = Send-Chat ($longPrompt + "`n`nAnswer with one short sentence.") 64; Write-Host ("    ok, content_chars={0} reasoning_chars={1}" -f $r2.choices[0].message.content.Length, $(if ($r2.choices[0].message.reasoning_content) { $r2.choices[0].message.reasoning_content.Length } else { 0 })) }
        catch { Write-Host ("    request 2 failed: {0}" -f $_.Exception.Message) }
        Start-Sleep -Seconds 3
    }
} finally {
    if ($p -and -not $p.HasExited) { $p.Kill(); Start-Sleep -Seconds 2; Write-Host "  server stopped" }
}

$mtpRows  = @(Select-String -Path $log -Pattern 'cannot remove uncommitted MTP rows' -ErrorAction SilentlyContinue).Count
$mmErr    = @(Select-String -Path $log -Pattern 'multimodal_error|multimodal_rollback_failed' -ErrorAction SilentlyContinue).Count
$seqRm    = @(Select-String -Path $log -Pattern 'KVarN can only remove' -ErrorAction SilentlyContinue).Count
$kvarN    = @(Select-String -Path $log -Pattern 'kvarN target cache' -ErrorAction SilentlyContinue).Count
$draftLine= @(Select-String -Path $log -Pattern 'structured KVarN cache type|draft' -ErrorAction SilentlyContinue | Select-Object -First 2)

Write-Host ""
Write-Host "=== case $Case result (draft cache = $draftName) ==="
Write-Host ("  'cannot remove uncommitted MTP rows' : {0}" -f $mtpRows)
Write-Host ("  multimodal_error / rollback_failed   : {0}" -f $mmErr)
Write-Host ("  'KVarN can only remove ...' warnings : {0}" -f $seqRm)
Write-Host ("  target KVarN active ('kvarN target cache') : {0}" -f $kvarN)
$retr  = @(Select-String -Path $log -Pattern 'KVMEM_RETRIEVAL' -ErrorAction SilentlyContinue).Count
$cache = @(Select-String -Path $log -Pattern 'cache = (\d+)' -ErrorAction SilentlyContinue | ForEach-Object { $_.Matches[0].Groups[1].Value })
Write-Host ("  KVMEM_RETRIEVAL lines                 : {0}" -f $retr)
Write-Host ("  prompt-eval cache hits per request    : {0}" -f ($cache -join ', '))
$park = @(Select-String -Path $log -Pattern 'failed to park' -ErrorAction SilentlyContinue).Count
Write-Host ("  'failed to park group' lines          : {0}" -f $park)
if ($mtpRows -gt 0) { $verdict = 'REPRODUCED (error present)' }
elseif ($Case -eq 0) {
    $refused = @(Select-String -Path $log -Pattern 'must not use KVarN while KVMem is enabled' -ErrorAction SilentlyContinue).Count
    if ($refused -gt 0 -and -not $ready) { $verdict = 'REFUSED at startup (expected for case 0)' }
    elseif ($ready) { $verdict = 'STARTED - the refusal did NOT happen (unexpected)' }
}
if ($verdict -eq 'UNKNOWN') { $verdict = 'NO ERROR' }
Write-Host ("  VERDICT: {0}" -f $verdict)
exit 0
