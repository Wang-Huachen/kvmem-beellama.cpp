#!/usr/bin/env pwsh
# 长上下文「针尖」验收（KVMem + KVarN）
#
# 目的：证明长上下文不是"能跑但答错"。这个场景刻意压在最容易出错的那条路上：
#   * ~30k token 的提示（必须真正 prefill + 前缀复用）
#   * 池子故意小（--kvmem-budget 12288）⇒ 必然发生淘汰 / 回迁 / 检索重排
#   * 密文放在提示的**最末尾** ⇒ 它的 token 落在这个上下文"尚未填满的那个块"里，
#     而 KVarN 只把不完整的那一组当作 live group 从 F16 暂存区读；一旦编号错位，
#     这一块会被静默读成全 0（历史 bug：`partial block is not in the live slot`）
#   * 第二个请求复用前缀（cache > 0）后再提问 ⇒ 必须答出密文
#
# PASS 条件：①HTTP 成功 ②回答里出现密文 ③服务端日志出现 `cache = `（证明确实命中前缀复用）
#           ④日志里没有 `partial block` / `already pending` / `multimodal_error`
[CmdletBinding()]
param(
    [string]$Repo    = "",
    [string]$Model   = "",
    [string]$Mmproj  = "",
    [int]$Port        = 8486,
    [int]$PoolTokens  = 12288,
    [int]$GenReserve  = 4096,
    [int]$CtxSize     = 262144,
    [int]$TargetTokens = 30000,
    [int]$BlockTokens = 128,
    [int]$DryRun      = 0,
    [string]$Log      = ""
)
$ErrorActionPreference = 'Continue'

# Defaults: the repo root is two levels above this script, and the model/mmproj come
# from the environment, so no machine-specific paths are stored in the repository.
if (-not $Repo)   { $Repo   = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
if (-not $Model)  { $Model  = $env:KVMEM_TEST_MODEL }
if (-not $Mmproj) { $Mmproj = $env:KVMEM_TEST_MMPROJ }
if (-not $Model -or -not (Test-Path $Model) -or -not $Mmproj -or -not (Test-Path $Mmproj)) {
    Write-Host 'need a test model and its mmproj: -Model <model.gguf> -Mmproj <mmproj.gguf>,'
    Write-Host 'or set $env:KVMEM_TEST_MODEL and $env:KVMEM_TEST_MMPROJ.'
    exit 2
}
if (-not $Log) { $Log = Join-Path $Repo 'logs\longctx-needle.log' }
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
Remove-Item $Log -ErrorAction SilentlyContinue
$exe    = Join-Path $Repo 'build-win\bin\llama-kvmem-server.exe'
$needle = 'ZX-4711-QQ'
$needle2 = 'QR-8823-LL'
$tag    = 'ZX-4711'
$tag2   = 'QR-8823'

# ---------- 1) 生成确定性的"干草堆" ----------
$words = @('provenance','ledger','cedar','quantum','harbour','meridian','tundra','cobalt','lattice','fjord',
           'ember','quartz','nimbus','sable','willow','onyx','drift','kestrel','marlin','zephyr',
           'basalt','clover','aspen','garnet','hydra','juniper','kelp','lumen','mosaic','nectar')
$sb = [System.Text.StringBuilder]::new()
$i  = 0
$targetChars = $TargetTokens * 4
while ($sb.Length -lt $targetChars) {
    $null = $sb.AppendFormat("Record {0}: the {1} {2} was observed near the {3} {4} during survey {5}, and the {6} {7} was logged as stable with {8} {9} entries.{10}", `
        $i, $words[$i % 30], $words[($i*7) % 30], $words[($i*11) % 30], $words[($i*13) % 30], $i, `
        $words[($i*17) % 30], $words[($i*19) % 30], ($i % 97), $words[($i*23) % 30], "`n")
    $i++
}
$hay = $sb.ToString()
# 第二个密文放在**很靠前**的位置（约 6%）：池子小的时候这一块会被换出到主机，提问时必须先靠检索把它取回来
# ⇒ 这条路径会走到 retrieval + stage-in + 池重排（历史上每次都在这里刷 E 级 "failed to park group N"）。
$early = [int]($hay.Length * 0.5)
$hay = $hay.Substring(0, $early) + "`nNOTE: the early archive code is $needle2.`n" + $hay.Substring($early)
# 密文放在最后 —— 它的 token 会落在这个上下文尚未填满的那个块里
$prompt1 = $hay + "`nIMPORTANT: the secret access code is $needle.`n"
$prompt2 = $prompt1 + "`nQuestion: what is the secret access code? Reply with the code only.`n"
$prompt3 = $prompt1 + "`nQuestion: what is the archive code from the beginning of the notes? Reply with the code only.`n"
"干草堆: " + [int]($hay.Length/1024) + " KB, " + $i + " 条记录（目标 ~$TargetTokens token）；密文 = $needle"
if ($DryRun -eq 1) { "DryRun=1 ⇒ 只生成提示，不启动服务。"; exit 0 }

# ---------- 2) 启动服务 ----------
$srvArgs = @('-m',$Model,'--ctx-size',"$CtxSize",'--image-min-tokens','1024','--no-mmproj-offload','--mmproj',$Mmproj,
             '--n-gpu-layers','99','--threads','12','--parallel','1','--flash-attn','on','--load-mode','none',
             '--kvmem-budget',"$PoolTokens",'--kvmem-gen-reserve',"$GenReserve",'--kvmem-block-tokens',"$BlockTokens",
             '--kv-dtype','kvarn6','--spec-draft-type-k','q8_0','--spec-draft-type-v','q8_0',
             '--spec-type','draft-mtp','--spec-draft-n-max','3',
             '--port',"$Port",'--alias','longctx-needle','--host','127.0.0.1','--timeout','600')
Get-Process llama-kvmem-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
"启动服务（池 $PoolTokens token / block $BlockTokens / kvarn6 / MTP）..."
$proc = Start-Process -FilePath $exe -ArgumentList $srvArgs -RedirectStandardOutput $Log -RedirectStandardError "$Log.err" -PassThru -NoNewWindow
$uri = "http://127.0.0.1:$Port/v1/chat/completions"
function Wait-Ready([int]$TimeoutSec) {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (Select-String -Path $Log,"$Log.err" -Pattern 'listening on' -Quiet -ErrorAction SilentlyContinue) { return $true }
        if ($proc.HasExited) { return $false }
        Start-Sleep -Milliseconds 500
    }
    return $false
}
function Ask([string]$text,[int]$maxTok,[int]$timeout) {
    $json  = @{ messages = @(@{ role = 'user'; content = $text }); max_tokens = $maxTok; temperature = 0.0; stream = $false } | ConvertTo-Json -Depth 6 -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    return Invoke-RestMethod -Uri $uri -Method Post -ContentType 'application/json; charset=utf-8' -Body $bytes -TimeoutSec $timeout
}
$verdict = 'FAIL'
try {
    if (-not (Wait-Ready 300)) { throw "服务未在 300 秒内就绪" }
    "服务就绪。"
    "== 请求 1：prefill ~$TargetTokens token =="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r1 = Ask $prompt1 16 900
    $sw.Stop()
    $t1 = if ($r1.choices) { [string]$r1.choices[0].message.content } else { '' }
    "  用时 " + [int]$sw.Elapsed.TotalSeconds + " s；回答长度 " + $t1.Length

    "== 请求 2：同一前缀 + 提问（应命中前缀复用） =="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r2 = Ask $prompt2 48 900
    $sw.Stop()
    $t2 = if ($r2.choices) { [string]$r2.choices[0].message.content } else { '' }
    "  用时 " + [int]$sw.Elapsed.TotalSeconds + " s"
    "  回答: " + ($t2 -replace '\s+',' ').Trim()
    "== 请求 3：同一前缀 + 问靠前的密文（小池子时必须靠检索取回） =="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r3 = Ask $prompt3 48 900
    $sw.Stop()
    $t3 = if ($r3.choices) { [string]$r3.choices[0].message.content } else { '' }
    "  用时 " + [int]$sw.Elapsed.TotalSeconds + " s"
    "  回答: " + ($t3 -replace '\s+',' ').Trim()
    $hit   = @(Select-String -Path $Log,"$Log.err" -Pattern 'cache =\s*[1-9]' -ErrorAction SilentlyContinue).Count
    $bad   = @(Select-String -Path $Log,"$Log.err" -Pattern 'partial block|already pending|multimodal_error|is not in the live slot|failed to park' -ErrorAction SilentlyContinue).Count
    $retr  = @(Select-String -Path $Log,"$Log.err" -Pattern 'KVMEM_RETRIEVAL' -ErrorAction SilentlyContinue).Count
    $guard = @(Select-String -Path $Log,"$Log.err" -Pattern 'record layout by original position disabled' -ErrorAction SilentlyContinue).Count
    "  前缀复用命中次数: $hit    已知错误行: $bad    检索次数: $retr    跳过重排提示: $guard"
    $okAnswer  = $t2 -match $tag
    $okAnswer2 = $t3 -match $tag2
    "  尾部密文答对: $okAnswer    靠前密文答对: $okAnswer2"
    if ($okAnswer -and $okAnswer2 -and $hit -gt 0 -and $bad -eq 0) { $verdict = 'PASS' }
} catch {
    "异常: " + $_.Exception.Message
} finally {
    if ($proc -and -not $proc.HasExited) { $proc.Kill() }
    Get-Process llama-kvmem-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
"RESULT: $verdict"
if ($verdict -eq 'PASS') { exit 0 } else { exit 1 }
