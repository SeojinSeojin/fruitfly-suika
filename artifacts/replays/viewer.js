'use strict';
const by = id => document.getElementById(id);
const labels = {...Object.fromEntries(Object.entries(window.siteCaptures ?? {}).map(([key,value])=>[key,value.label])),drop5:'5 drops', drop1000:'1,000 drops', early:'Early · 5%', middle:'Middle · 50%', final:'Final · 100%', website:'Website transfer', websiteFiltered:'Website · ads blocked', websiteGameOver:'Website · game-over mode', websiteSettling:'Website · first uncapped run', websiteAimStop:'Website · 1,855-point run'};
const regionNames = {IPS_L:'Left inferior posterior slope', IPS_R:'Right inferior posterior slope',
  LOP_R:'Right lobula plate', LO_R:'Right lobula', ME_R:'Right medulla',
  PVLP_R:'Right posterior ventrolateral protocerebrum', SPS_R:'Right superior posterior slope', WED_R:'Right wedge'};
const siteFolders = {...Object.fromEntries(Object.entries(window.siteCaptures ?? {}).map(([key,value])=>[key,value.folder])),website:'../demo/', websiteFiltered:'../demo-ad-filtered/', websiteGameOver:'../demo-game-over/', websiteSettling:'../demo-settling-stop/', websiteAimStop:'../demo-aim-stop/'};
const video = by('video'), image = by('capture'), slider = by('drop'), canvas = by('circuit');
const ctx = canvas.getContext('2d');
let graph, replay, entry, current = 0, projected = [], selected = null, timer = null, openToken = 0, regionFocus = null;

async function unpack(encoded) {
  const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).json();
}

function positions(graph) {
  const known = graph.positions.filter(Boolean);
  if (!known.length) return graph.ids.map((_, i) => {
    const t = i * 2.39996323, r = Math.sqrt((i + .5) / graph.ids.length);
    return [r * Math.cos(t), r * Math.sin(t), 0];
  });
  const minimum = [0,1,2].map(j => Math.min(...known.map(p => p[j])));
  const maximum = [0,1,2].map(j => Math.max(...known.map(p => p[j])));
  const span = Math.max(...maximum.map((v,j) => v - minimum[j]));
  return graph.positions.map(p => p && p.map((v,j) => 2 * (v - (minimum[j] + maximum[j])/2) / span));
}

function tone(value) {
  const target = value < 0 ? [87,172,245] : [248,170,105];
  return 'rgb(' + target.map((v,i) => Math.round([38,52,56][i] * (1-Math.abs(value)) + v * Math.abs(value))).join(',') + ')';
}

function draw() {
  if (!replay?.decisions.length) return;
  const decision = replay.decisions[current], step = Number(by('step').value);
  const state = decision.steps[step - 1];
  const theta = Number(by('rotate').value) * Math.PI / 180;
  by('stepValue').value = step;
  slider.value = decision.drop;
  const score = entry.website ? `${decision.score_before ?? '?'} → ${decision.score ?? '?'}` : decision.score;
  by('decision').value = `#${decision.drop} · aim ${decision.action + 1}/15 · score ${score}`;
  by('regionTitle').textContent = `Neuropil activity · drop ${decision.drop}, recurrent step ${step}/${decision.steps.length}`;
  const focusedNodes = new Set(graph.edges.filter(edge => edge[3][regionFocus]).map(edge => edge[0]));
  if (entry.website && image.getAttribute('src') !== entry.folder + decision.image) image.src = entry.folder + decision.image;
  projected = graph.points.map((p,i) => p && ({
    x: 292 + 243 * (Math.cos(theta)*p[0] + Math.sin(theta)*p[2]),
    y: 256 + 220*p[1],
    z: -Math.sin(theta)*p[0] + Math.cos(theta)*p[2], i
  }));
  ctx.clearRect(0,0,584,512);
  ctx.strokeStyle = graph.positions.some(Boolean) ? '#31484650' : '#5e858080';
  ctx.lineWidth = .6;
  ctx.beginPath();
  for (const [a,b] of graph.edges) {
    if (!projected[a] || !projected[b]) continue;
    ctx.moveTo(projected[a].x, projected[a].y);
    ctx.lineTo(projected[b].x, projected[b].y);
  }
  ctx.stroke();
  for (const {x,y,i} of projected.filter(Boolean).sort((a,b) => a.z-b.z)) {
    ctx.globalAlpha = regionFocus && !focusedNodes.has(i) ? .25 : 1;
    ctx.fillStyle = tone(state.delta[i]);
    ctx.beginPath(); ctx.arc(x,y,2.8,0,2*Math.PI); ctx.fill();
    if (Math.abs(decision.direct[i]) > .00001 || i === selected) {
      ctx.strokeStyle = i === selected ? '#ffffff' : '#9cbdb3';
      ctx.beginPath(); ctx.moveTo(x-4,y); ctx.lineTo(x,y-4); ctx.lineTo(x+4,y); ctx.lineTo(x,y+4); ctx.closePath(); ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  const f = x => Number(x).toFixed(3);
  by('node').textContent = selected === null ? 'Select a neuron to inspect its inputs and activity.' :
    `Neuron ${graph.ids[selected]} · direct ${f(decision.direct[selected])} · recurrent ${f(state.recurrent[selected])} · activation ${f(state.activation[selected])} · zero-input ${f(state.reference[selected])}`;
  const regions = {};
  for (const [source,,, annotations] of graph.edges) {
    for (const [region,count] of Object.entries(annotations)) {
      if (region === 'UNASGD') continue;
      const sums = regions[region] ??= [0,0];
      sums[0] += count * Math.abs(state.delta[source]); sums[1] += count;
    }
  }
  by('regionFocus').textContent = regionFocus
    ? `${regionFocus} highlighted · ${focusedNodes.size} retained presynaptic neurons`
    : 'Select a neuropil to highlight its retained presynaptic neurons.';
  by('regions').replaceChildren(...Object.entries(regions).sort((a,b) => b[1][0]/b[1][1]-a[1][0]/a[1][1]).map(([name,[sum,n]]) => {
    const li = document.createElement('li');
    const button = document.createElement('button');
    const magnitude = sum/n;
    button.className = 'region';
    button.setAttribute('aria-pressed', regionFocus === name);
    button.style.setProperty('--magnitude', `${100*magnitude}%`);
    button.textContent = `${regionNames[name] ?? name} (${name}) · ${f(magnitude)} · ${n.toLocaleString()} synapses`;
    button.onclick = () => { regionFocus = regionFocus === name ? null : name; draw(); };
    li.append(button);
    return li;
  }));
}

function jump(index) {
  current = Math.min(Math.max(index,0),replay.decisions.length-1);
  if (!entry.website) video.currentTime = replay.decisions[current].timestamp + .001;
  draw();
}

function pause() {
  video.pause(); clearInterval(timer); timer = null; by('play').textContent = 'Play';
}

async function open(label) {
  pause();
  const token = ++openToken;
  const requested = label in siteFolders ? {website:true, trace:label, folder:siteFolders[label]} : window.localManifest[label];
  const data = await unpack(window.replays[requested.trace]);
  if (token !== openToken) return;
  entry = requested;
  replay = data;
  for (const button of by('milestones').children) button.setAttribute('aria-pressed', button.dataset.label === label);
  video.hidden = Boolean(entry.website); image.hidden = !entry.website;
  by('terminalCapture').hidden = !entry.website || !replay.meta.final_capture;
  if (entry.website && replay.meta.final_capture) by('terminalCapture').href = entry.folder + replay.meta.final_capture;
  by('gameTitle').textContent = entry.website ? 'Website transfer' : 'Gameplay';
  current = 0;
  by('status').textContent = entry.website
    ? `${({hybrid:'FlyWire + local planning · ',robust:'FlyWire + perturbation-aware planning · '})[replay.meta.strategy] ?? ''}${replay.decisions.length} drop${replay.decisions.length === 1 ? '' : 's'} · site ${replay.meta.outcome} · final observed score ${replay.meta.final_score ?? 'unknown'}`
    : `Requested ${entry.requested.toLocaleString()} drops · checkpoint ${entry.drops.toLocaleString()} drops · ${entry.updates} PPO updates${entry.pre_update ? ' · pre-update policy' : ''} · fruit seed ${entry.fruit_seed}`;
  if (!entry.website) video.src = '../videos/' + entry.video;
  slider.max = replay.decisions.length;
  draw();
}

canvas.onclick = event => {
  const bounds = canvas.getBoundingClientRect();
  const x = (event.clientX-bounds.left)*584/bounds.width, y = (event.clientY-bounds.top)*512/bounds.height;
  const nearest = projected.filter(Boolean).map(p => [Math.hypot(x-p.x,y-p.y),p.i]).sort((a,b)=>a[0]-b[0])[0];
  selected = nearest && nearest[0] < 15 ? nearest[1] : null; draw();
};
by('play').onclick = async () => {
  if (!replay) return;
  if (entry.website) {
    if (timer) { pause(); return; }
    if (current === replay.decisions.length-1) jump(0);
    by('play').textContent = 'Pause';
    timer = setInterval(() => current === replay.decisions.length-1 ? pause() : jump(current+1), 1000);
  } else if (video.paused) {
    if (video.ended) jump(0);
    await video.play(); by('play').textContent = 'Pause';
  } else pause();
};
by('back').onclick = () => { pause(); jump(current-1); };
by('next').onclick = () => { pause(); jump(current+1); };
slider.oninput = () => { pause(); jump(Number(slider.value)-1); };
by('step').oninput = by('rotate').oninput = draw;
video.onended = pause;
video.ontimeupdate = () => {
  if (!entry || entry.website) return;
  const i = Math.min(replay.decisions.length-1, Math.floor(video.currentTime * entry.fps / entry.frames_per_drop));
  if (i !== current) { current = i; draw(); }
};

(async () => {
  graph = await unpack(window.replays.graph);
  graph.points = positions(graph);
  by('coverage').textContent = `${(100 * (graph.annotation_coverage - graph.edges.reduce((n, edge) => n + (edge[3].UNASGD ?? 0), 0) /
    graph.edges.reduce((n, edge) => n + edge[2], 0))).toFixed(2)}% of retained synapses assigned to a neuropil. The proxy averages absolute presynaptic activation changes, weighted by retained outgoing synapses in each neuropil. ${graph.positions.some(Boolean) ? '3D coordinate projection.' : 'Connectivity layout.'}`;
  for (const [key,label] of Object.entries(labels)) {
    if (!(key in window.localManifest) && !(key in window.replays)) continue;
    const button = document.createElement('button'); button.textContent = label; button.dataset.label = key;
    button.onclick = () => open(key); by('milestones').append(button);
  }
  const requested = Array.from(by('milestones').children).find(button => '#' + button.dataset.label === location.hash);
  const initial = requested ?? by('milestones').firstChild;
  if (initial) await open(initial.dataset.label);
})().catch(error => { by('status').textContent = error.message; });
