// Shared Chart.js defaults
Chart.defaults.color = '#5a7394';
Chart.defaults.borderColor = '#1a2d44';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 11;

const C = { a1:'#00e5ff', a2:'#ff6b35', a3:'#7b61ff', a4:'#00ff9d', muted:'#5a7394', border:'#1a2d44' };
const charts = {};

function mkChart(id, type, data, opts={}) {
  const el = document.getElementById(id);
  if (!el) return;
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(el, { type, data, options:{ responsive:true, maintainAspectRatio:false, ...opts }});
  return charts[id];
}

function lineOpts(extra={}) {
  return {
    responsive:true, maintainAspectRatio:false,
    interaction:{ mode:'index', intersect:false },
    plugins:{
      legend:{ labels:{ color:C.muted, boxWidth:12 }},
      tooltip:{ backgroundColor:'#0d1520', borderColor:C.border, borderWidth:1 }
    },
    scales:{
      x:{ grid:{ color:C.border }, ticks:{ color:C.muted }},
      y:{ grid:{ color:C.border }, ticks:{ color:C.muted }}
    },
    ...extra
  };
}

function barOpts(extra={}) {
  return {
    responsive:true, maintainAspectRatio:false,
    plugins:{
      legend:{ labels:{ color:C.muted, boxWidth:12 }},
      tooltip:{ backgroundColor:'#0d1520', borderColor:C.border, borderWidth:1 }
    },
    scales:{
      x:{ grid:{ display:false }, ticks:{ color:C.muted }},
      y:{ grid:{ color:C.border }, ticks:{ color:C.muted }}
    },
    ...extra
  };
}