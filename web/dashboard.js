const classes=[{id:0,label:'Normal',color:'#668b72'},{id:1,label:'Desgaste ferramenta',color:'#bd8747'},{id:2,label:'Falha refrigeração',color:'#c76d56'},{id:3,label:'Desalinhamento',color:'#647d88'}];
const $=id=>document.getElementById(id),fmt=(n,d=0)=>Number(n).toLocaleString('pt-BR',{minimumFractionDigits:d,maximumFractionDigits:d}),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let rows=[],missing=0;
function filters(){[['cellFilter',[...new Set(rows.map(r=>r.id_celula))].sort()],['alloyFilter',[...new Set(rows.map(r=>r.liga_metalica))].sort()]].forEach(([id,values])=>values.forEach(v=>$(id).insertAdjacentHTML('beforeend',`<option value="${esc(v)}">${esc(v.replaceAll('_',' '))}</option>`)));}
function selected(){let data=rows.filter(r=>($('cellFilter').value==='all'||r.id_celula===$('cellFilter').value)&&($('alloyFilter').value==='all'||r.liga_metalica===$('alloyFilter').value)),p=$('periodFilter').value;if(p!=='all'&&data.length){let end=Math.max(...data.map(r=>Date.parse(r.timestamp))),span=(p==='day'?1:7)*86400000;data=data.filter(r=>Date.parse(r.timestamp)>=end-span);}return data;}
function bars(id,data,percent=false){let W=620,H=220,L=44,R=8,T=12,B=48,h=H-T-B,w=W-L-R,max=Math.max(1,...data.map(x=>x.value));if(!data.length){$(id).innerHTML='<div class="empty-state">Sem dados para os filtros.</div>';return;}let s=`<svg viewBox="0 0 ${W} ${H}">`;for(let j=0;j<=3;j++){let y=T+h*j/3;s+=`<line class="chart-gridline" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text class="chart-label" text-anchor="end" x="${L-6}" y="${y+4}">${fmt(max*(1-j/3))}${percent?'%':''}</text>`;}let step=w/data.length,bw=Math.min(68,step*.58);data.forEach((x,i)=>{let bh=Math.max(2,x.value/max*h),bx=L+i*step+(step-bw)/2,by=T+h-bh;s+=`<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" fill="${x.color}"/><text class="chart-value" text-anchor="middle" x="${bx+bw/2}" y="${Math.max(T+10,by-5)}">${fmt(x.value,percent?1:0)}${percent?'%':''}</text><text class="chart-label" text-anchor="middle" x="${bx+bw/2}" y="${H-10}">${esc(x.label)}</text>`;});$(id).innerHTML=s+'</svg>';}
function render(){let d=selected(),fault=d.filter(r=>+r.tipo_falha!==0);$('sampleCount').textContent=fmt(d.length);$('activeCells').textContent=fmt(new Set(d.map(r=>r.id_celula)).size);$('failureRate').textContent=`${fmt(d.length?100*fault.length/d.length:0,1)}%`;$('failureCount').textContent=`${fmt(fault.length)} registros não normais`;$('missingRate').textContent=`${fmt(missing,2)}%`;bars('faultChart',classes.map(c=>({label:c.label,value:d.filter(r=>+r.tipo_falha===c.id).length,color:c.color})));let cells=[...new Set(d.map(r=>r.id_celula))].sort().map((c,i)=>{let a=d.filter(r=>r.id_celula===c);return{label:c,value:a.length?100*a.filter(r=>+r.tipo_falha!==0).length/a.length:0,color:['#668b72','#647d88','#bd8747','#967c62'][i%4]};});bars('cellChart',cells,true);let alloys=[...new Set(d.map(r=>r.liga_metalica))].sort();$('alloyTable').innerHTML=alloys.map(a=>{let x=d.filter(r=>r.liga_metalica===a),avg=k=>x.reduce((s,r)=>s+(+r[k]||0),0)/Math.max(1,x.length);return`<tr><td>${esc(a.replaceAll('_',' '))}</td><td>${fmt(x.length)}</td><td>${fmt(avg('temperatura_fuso_c'),1)} °C</td><td>${fmt(avg('vibracao_triaxial_rms_mms'),2)} mm/s</td><td>${fmt(avg('iem'),2)}</td><td>${fmt(x.filter(r=>+r.tipo_falha!==0).length)}</td></tr>`;}).join('');}
async function report(){try{let p=await(await fetch('/api/model-report')).json();if(!p.disponivel){$('modelReport').innerHTML=`<p class="subtle">${esc(p.mensagem)}</p>`;return;}let names=['Normal','Desgaste ferramenta','Falha refrigeração','Desalinhamento / trinca'];$('modelStatus').textContent=p.modelo.toUpperCase();if(p.oee_estimado!=null){$('oeeSub').textContent='Proxy demonstrativo pelo modelo';$('oeeMark').textContent='≈';$('oeeTitle').textContent=`OEE estimado: ${(p.oee_estimado*100).toLocaleString('pt-BR',{maximumFractionDigits:1})}%`;$('oeeText').textContent=`Proxy a partir das classes previstas no teste temporal (disponibilidade estimada ${(p.disponibilidade_estimada*100).toLocaleString('pt-BR',{maximumFractionDigits:1})}% × qualidade estimada ${(p.qualidade_estimada*100).toLocaleString('pt-BR',{maximumFractionDigits:1})}%). Não é o OEE real da fábrica: o CSV não contém tempo planejado, paradas, desempenho e peças produzidas.`;}let rec=names.map((n,i)=>{let v=+p.recall_por_classe[i]||0;return`<div class="report-row"><span>${n}</span><div class="progress-track"><i style="width:${Math.min(100,v*100)}%"></i></div><b>${fmt(v*100,1)}%</b></div>`;}).join(''),imp=p.atributos_importantes.map(x=>`<div class="importance-row"><span>${esc(x.nome)}</span><b>${fmt(x.importancia,3)}</b></div>`).join(''),matrix=p.matriz_confusao.map((r,i)=>`<tr><th>${names[i]}</th>${r.map(v=>`<td>${fmt(v)}</td>`).join('')}</tr>`).join(''),warn=p.recall_medio_falhas<.9?'<div class="model-warning">O recall médio ainda está abaixo da meta de 90% do enunciado.</div>':'<div class="model-success">A meta de recall médio de 90% foi atingida.</div>';$('modelReport').innerHTML=`<div class="model-summary">Melhor modelo: <b>${esc(p.modelo)}</b> · recall médio das falhas: <b>${fmt(p.recall_medio_falhas*100,1)}%</b></div>${warn}<div class="model-columns"><div><div class="panel-tag">RECALL POR CLASSE</div><div class="report-list">${rec}</div></div><div><div class="panel-tag">ATRIBUTOS IMPORTANTES</div>${imp}</div></div><div class="confusion-wrap"><table class="confusion-table"><thead><tr><th>REAL ↓ / PREVISTO →</th>${names.map(n=>`<th>${n}</th>`).join('')}</tr></thead><tbody>${matrix}</tbody></table></div>`;}catch(_){$('modelStatus').textContent='RELATÓRIO INDISPONÍVEL';}}
async function load(){ $('refreshButton').disabled=true;try{let p=await(await fetch('/api/dashboard-data')).json();rows=p.rows;missing=p.missing_rate;if(!$('cellFilter').dataset.loaded){filters();$('cellFilter').dataset.loaded='1';}render();await report();}catch(_){$('faultChart').innerHTML='<div class="empty-state">Falha ao carregar dados. Confira se a API está ativa.</div>'; }finally{$('refreshButton').disabled=false;}}
['cellFilter','alloyFilter','periodFilter'].forEach(id=>$(id).addEventListener('change',render));$('refreshButton').addEventListener('click',load);load();

// --- PRESETS DE CENÁRIOS DE TESTE ---
const presets = {
  presetNormal: { celula: 'CEL-201', liga: 'Aco_Carbono_316L', temp: 45.0, vib: 1.2, torque: 90, vazao: 30.0, corrente: 25.0, pressao: 160, ruido: 55 },
  presetDesgaste: { celula: 'CEL-202', liga: 'Aco_Carbono_316L', temp: 62.0, vib: 3.5, torque: 195, vazao: 22.0, corrente: 42.0, pressao: 165, ruido: 82 },
  presetRefrigeracao: { celula: 'CEL-203', liga: 'Aluminio_7075', temp: 74.0, vib: 2.0, torque: 110, vazao: 15.2, corrente: 30.0, pressao: 150, ruido: 68 },
  presetDesalinhamento: { celula: 'CEL-204', liga: 'Liga_Titanio_Ti6Al4V', temp: 58.0, vib: 3.6, torque: 160, vazao: 28.0, corrente: 52.0, pressao: 120, ruido: 79 }
};

Object.entries(presets).forEach(([btnId, p]) => {
  const btn = document.getElementById(btnId);
  if (btn) {
    btn.addEventListener('click', () => {
      document.getElementById('pred_celula').value = p.celula;
      document.getElementById('pred_liga').value = p.liga;
      document.getElementById('pred_temp').value = p.temp;
      document.getElementById('pred_vib').value = p.vib;
      document.getElementById('pred_torque').value = p.torque;
      document.getElementById('pred_vazao').value = p.vazao;
      document.getElementById('pred_corrente').value = p.corrente;
      document.getElementById('pred_pressao').value = p.pressao;
      document.getElementById('pred_ruido').value = p.ruido;
    });
  }
});

// --- EXECUÇÃO E EXIBIÇÃO DE DIAGNÓSTICO ---
const predictForm = document.getElementById('predictForm');
if (predictForm) {
  predictForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('predictResult');
    resultDiv.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--muted);">Analisando leitura no modelo de Machine Learning...</div>';

    const torque = parseFloat(document.getElementById('pred_torque').value);
    const vib = parseFloat(document.getElementById('pred_vib').value);
    const vazao = parseFloat(document.getElementById('pred_vazao').value);
    const iemCalculado = (torque * vib) / (vazao + 1);

    const payload = {
      id_celula: document.getElementById('pred_celula').value,
      liga_metalica: document.getElementById('pred_liga').value,
      temperatura_fuso_c: parseFloat(document.getElementById('pred_temp').value),
      vibracao_triaxial_rms_mms: vib,
      torque_fuso_nm: torque,
      vazao_refrigerante_lmin: vazao,
      corrente_estator_amp: parseFloat(document.getElementById('pred_corrente').value),
      pressao_hidraulica_bar: parseFloat(document.getElementById('pred_pressao').value),
      nivel_ruido_db: parseFloat(document.getElementById('pred_ruido').value)
    };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Erro na predição');

      const isNormal = data.tipo_falha === 0;
      const statusClass = isNormal ? 'model-success' : 'model-warning';
      const badgeColor = isNormal ? 'var(--green)' : 'var(--danger)';
      const confPct = (data.confianca * 100).toFixed(1);

      const orientacoes = {
        0: 'Equipamento operando dentro dos parâmetros nominais. Nenhuma ação corretiva necessária.',
        1: 'Alto estresse mecânico detectado (Vibração e Torque elevados). Recomenda-se substituição/inspeção da ferramenta de corte.',
        2: 'Temperatura elevada acompanhada de baixa vazão de refrigerante. Verifique obstruções no fluido de corte.',
        3: 'Nível crítico de vibração e alta corrente no estator. Verifique o alinhamento do eixo e integridade da estrutura.'
      };

      resultDiv.innerHTML = `
        <div class="${statusClass}" style="flex-direction: column; align-items: flex-start; gap: 12px; padding: 18px; border-radius: var(--radius-md);">
          <div style="display: flex; justify-content: space-between; width: 100%; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
              <span style="font-size: 20px;">${isNormal ? '✅' : '⚠️'}</span>
              <strong style="font-size: 16px;">Diagnóstico: ${data.descricao}</strong>
            </div>
            <span style="font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 12px; background: rgba(0,0,0,0.05); color: ${badgeColor}; border: 1px solid ${badgeColor};">
              Confiança da IA: ${confPct}%
            </span>
          </div>
          
          <p style="margin: 0; font-size: 12px; opacity: 0.95;">
            <b>Ação Recomendada:</b> ${orientacoes[data.tipo_falha] || 'Verifique o estado geral do equipamento.'}
          </p>

          <div style="display: flex; gap: 20px; font-size: 11px; margin-top: 6px; padding-top: 10px; border-top: 1px solid rgba(0,0,0,0.08); width: 100%;">
            <span>Índice de Estresse (IEM): <b>${iemCalculado.toFixed(2)}</b></span>
            <span>Célula: <b>${payload.id_celula}</b></span>
            <span>Liga: <b>${payload.liga_metalica.replaceAll('_',' ')}</b></span>
          </div>
        </div>
      `;
    } catch (err) {
      resultDiv.innerHTML = `<div class="model-warning" style="color: var(--danger);"><b>Erro ao simular:</b> ${err.message}. Certifique-se de executar <code>python treinar_modelo.py</code> antes.</div>`;
    }
  });
}