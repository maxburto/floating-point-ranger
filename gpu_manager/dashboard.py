"""Dashboard page for gpu-manager."""

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Manager</title>
<style>
  :root { color-scheme: light dark; --bg:#f7f7f9; --card:#fff; --line:#e3e3e8; --muted:#6b7280;
          --accent:#3b6ea5; --ok:#2e7d46; --warn:#b26a00; --err:#b3261e; }
  @media (prefers-color-scheme: dark){ :root{ --bg:#1b1d22; --card:#24272e; --line:#33373f;
          --muted:#9aa0aa; --accent:#5a8dc9; } body{color:#e6e8ec;} }
  * { box-sizing:border-box; } body { margin:0; font:15px/1.5 system-ui,sans-serif; background:var(--bg); }
  .wrap { max-width:900px; margin:0 auto; padding:18px 16px 40px; }
  h1 { font-size:19px; margin:4px 0 2px; } .sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:10px 0; }
  .row { display:flex; justify-content:space-between; gap:12px; align-items:baseline; flex-wrap:wrap; }
  .fn { font-weight:600; } .muted { color:var(--muted); font-size:13px; }
  .badge { font-size:12px; font-weight:600; padding:2px 9px; border-radius:20px; white-space:nowrap; }
  .b-interactive { background:rgba(59,110,165,.15); color:var(--accent); }
  .b-batch { background:rgba(46,125,70,.15); color:var(--ok); }
  .b-hold { background:rgba(178,106,0,.15); color:var(--warn); }
  .b-running { background:rgba(59,110,165,.15); color:var(--accent); }
  .b-queued { background:rgba(178,106,0,.15); color:var(--warn); }
  .b-error, .b-offline { background:rgba(179,38,30,.15); color:var(--err); }
  .bar { height:8px; background:var(--line); border-radius:6px; overflow:hidden; margin-top:8px; }
  .bar > i { display:block; height:100%; background:var(--accent); transition:width .4s; }
  table { width:100%; border-collapse:collapse; margin-top:8px; font-size:13px; }
  th, td { text-align:left; padding:3px 8px 3px 0; } th { color:var(--muted); font-weight:500; }
  .grouphdr { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:22px 2px 4px; }
  .empty { color:var(--muted); text-align:center; padding:18px 0; }
  .foot { color:var(--muted); font-size:12px; margin-top:22px; text-align:center; }
  .btn { padding:6px 14px; border:1px solid var(--line); border-radius:8px; background:transparent;
         color:var(--accent); font-weight:600; cursor:pointer; }
  .btn:hover { border-color:var(--accent); }
  code { font:12px ui-monospace,monospace; }
</style></head>
<body><div class="wrap">
  <h1>GPU Manager</h1>
  <div class="sub">Live GPU state and the merged job queue. Read-only view; updates automatically.</div>
  <div class="grouphdr">GPUs</div><div id="gpus"><div class="empty">Loading&hellip;</div></div>
  <div class="grouphdr">Flagged / stalled jobs</div><div id="flagged"><div class="empty">Loading&hellip;</div></div>
  <div class="grouphdr">Queue</div><div id="queue"><div class="empty">Loading&hellip;</div></div>
  <div class="grouphdr">Models</div><div id="models"><div class="empty">Loading&hellip;</div></div>
  <div class="grouphdr" style="cursor:pointer" onclick="toggleSettings()">Settings &#9662;</div>
  <div id="settings" style="display:none">
    <div class="card">
      <div class="muted" style="margin-bottom:8px">Edits the live <code>config.yaml</code> (GPU roles,
        allow-patterns, models, presence, queue sources). Validated before saving; the service
        restarts itself to apply (~3&nbsp;s). A timestamped backup is kept next to the file.
        The API token lives on the server in <code>/opt/gpu-manager/env</code>.</div>
      <input type="password" id="tok" placeholder="API token" style="width:100%;margin-bottom:8px;
        padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:transparent;color:inherit">
      <div style="display:flex;gap:8px;margin-bottom:8px">
        <button onclick="loadCfg()" class="btn">Load config</button>
        <button onclick="saveCfg()" class="btn">Validate &amp; save</button>
        <span id="cfgmsg" class="muted"></span>
      </div>
      <textarea id="cfg" spellcheck="false" style="width:100%;min-height:320px;font:12px/1.5 ui-monospace,monospace;
        border:1px solid var(--line);border-radius:8px;background:transparent;color:inherit;padding:8px"
        placeholder="Press 'Load config' to fetch the live configuration."></textarea>
    </div>
  </div>
  <div class="foot" id="foot"></div>
</div>
<script>
function esc(s){ return (s||"").toString().replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function fmtEta(s){ if(s==null) return "\\u2014";
  if(s<90) return "~"+Math.round(s)+"s"; return "~"+Math.round(s/60)+" min"; }
function gpuCard(g){
  var pct = (g.mem_total_mib ? Math.round(100*g.mem_used_mib/g.mem_total_mib) : 0);
  var badges = '<span class="badge b-'+esc(g.role)+'">'+esc(g.role)+'</span>';
  if (g.hold && g.hold.active) badges += ' <span class="badge b-hold">hold</span>';
  if (g.mps && g.mps.server_up) badges += ' <span class="badge b-running">MPS'
    + (g.mps.concurrent_batch ? '' : ' (serial)') + '</span>';
  (g.locks||[]).forEach(function(l){ if(l.held) badges += ' <span class="badge b-hold">'+esc(l.label)+' held</span>'; });
  if (!g.online) badges += ' <span class="badge b-offline">offline</span>';
  var leases = (g.leases||[]).map(function(l){
    return '<div class="muted">lease: '+esc(l.initiator)+' \\u00b7 '+esc(l.label)
      +(l.owner?(' \\u00b7 owner '+esc(l.owner)):'')
      +(l.capability?(' \\u00b7 '+esc(l.capability)):'')
      +(l.job_type&&l.job_type!=='oneshot'?(' \\u00b7 '+esc(l.job_type)):'')
      +(l.vram_mib?(' \\u00b7 '+l.vram_mib+' MiB reserved'):'')+' \\u00b7 '+l.age_s+'s</div>'; }).join("");
  var procs = (g.processes||[]).filter(function(p){ return p.used_mib==null || p.used_mib>=10; });
  var rows = procs.map(function(p){
    return '<tr><td>'+p.pid+'</td><td>'+esc(p.name)+'</td><td>'+esc(p.kind)+'</td><td>'
      +(p.used_mib==null?'\\u2014':p.used_mib+' MiB')+'</td></tr>'; }).join("");
  var table = procs.length ? '<table><tr><th>pid</th><th>process</th><th>kind</th><th>VRAM</th></tr>'+rows+'</table>'
                           : '<div class="muted">No significant GPU processes.</div>';
  var mem = g.online ? (g.mem_used_mib+' / '+g.mem_total_mib+' MiB') : 'n/a';
  return '<div class="card"><div class="row"><span class="fn">'+esc(g.name)+'</span><span>'+badges+'</span></div>'
    +'<div class="muted">'+esc(g.uuid)+' \\u00b7 '+esc(g.host)+' \\u00b7 VRAM '+mem+'</div>'
    +'<div class="bar"><i style="width:'+pct+'%"></i></div>'+leases+table+'</div>';
}
function flaggedCard(f){
  var b = (f.action && f.action.indexOf('evicted')===0) ? 'b-error' : 'b-hold';
  return '<div class="card"><div class="row"><span class="fn">'+esc(f.label)+'</span>'
    +'<span class="badge '+b+'">stalled</span></div>'
    +'<div class="muted">owner '+esc(f.owner)+' \\u00b7 pid '+f.pid+' \\u00b7 idle '+f.idle_s+'s'
    +' \\u00b7 vouch '+esc(f.vouch)+' \\u00b7 '+esc(f.action)+'</div></div>';
}
function queueCard(e){
  return '<div class="card"><div class="row"><span class="fn">'+esc(e.label)+'</span>'
    +'<span class="badge b-'+esc(e.state)+'">'+esc(e.state)+'</span></div>'
    +'<div class="muted">'+esc(e.initiator)+' \\u00b7 '+esc(e.id)
    +(e.created_at?(' \\u00b7 created '+esc(e.created_at)):'')
    +(e.state==='queued' ? (' \\u00b7 ETA '+fmtEta(e.eta_seconds)) : '')
    +(e.stale ? ' \\u00b7 (cached view)' : '')+'</div></div>';
}
async function tick(){
  try{
    var rs = await fetch("v1/gpu/status", {cache:"no-store"}); var ds = await rs.json();
    document.getElementById("gpus").innerHTML =
      (ds.gpus||[]).map(gpuCard).join("") || '<div class="empty">No GPUs configured.</div>';
    document.getElementById("flagged").innerHTML =
      (ds.flagged||[]).map(flaggedCard).join("") || '<div class="empty">No flagged jobs.</div>';
    var rq = await fetch("v1/gpu/queue", {cache:"no-store"}); var dq = await rq.json();
    document.getElementById("queue").innerHTML =
      (dq.entries||[]).map(queueCard).join("") || '<div class="empty">Queue is empty.</div>';
    document.getElementById("foot").textContent = "Updated "+new Date().toLocaleTimeString();
  }catch(e){ /* keep last view on transient errors */ }
}
async function modelsTick(){
  try{
    var r = await fetch("v1/models", {cache:"no-store"}); var d = await r.json();
    document.getElementById("models").innerHTML = (d.models||[]).map(function(m){
      var st = m.unit_state === "active" ? "running" : (m.unit_state === "activating" ? "queued" : "offline");
      var lbl = m.unit_state === "active" ? "resident" : m.unit_state;
      return '<div class="card"><div class="row"><span class="fn">'+esc(m.name)+'</span>'
        +'<span class="badge b-'+st+'">'+esc(lbl)+'</span></div>'
        +'<div class="muted">'+esc(m.unit)+' \\u00b7 '+m.vram_mib+' MiB floor'
        +(m.port?(' \\u00b7 :'+m.port):'')
        +(m.idle_s!=null?(' \\u00b7 last ensure '+m.idle_s+'s ago'):'')+'</div></div>';
    }).join("") || '<div class="empty">No models declared.</div>';
  }catch(e){}
}
function toggleSettings(){
  var el = document.getElementById("settings");
  el.style.display = el.style.display === "none" ? "block" : "none";
  var t = document.getElementById("tok");
  if (!t.value && localStorage.gmTok) t.value = localStorage.gmTok;
}
function hdrs(){ localStorage.gmTok = document.getElementById("tok").value;
  return {"Authorization": "Bearer "+document.getElementById("tok").value}; }
async function loadCfg(){
  var m = document.getElementById("cfgmsg");
  try{
    var r = await fetch("v1/config", {headers: hdrs(), cache:"no-store"});
    if(!r.ok){ m.textContent = "load failed: HTTP "+r.status; return; }
    document.getElementById("cfg").value = await r.text(); m.textContent = "loaded";
  }catch(e){ m.textContent = "load failed"; }
}
async function saveCfg(){
  var m = document.getElementById("cfgmsg"); m.textContent = "validating\\u2026";
  try{
    var h = hdrs(); h["Content-Type"] = "application/json";
    var r = await fetch("v1/config", {method:"PUT", headers:h,
      body: JSON.stringify({yaml: document.getElementById("cfg").value})});
    var d = await r.json();
    m.textContent = r.ok ? ("saved ("+d.backup+"); restarting\\u2026") : ("rejected: "+(d.detail||r.status));
  }catch(e){ m.textContent = "save failed"; }
}
tick(); modelsTick(); setInterval(tick, 2500); setInterval(modelsTick, 5000);
</script></body></html>"""
