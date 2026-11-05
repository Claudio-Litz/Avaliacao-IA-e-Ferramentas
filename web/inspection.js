const input=document.getElementById('weldFile'),canvas=document.getElementById('weldCanvas'),ctx=canvas.getContext('2d'),placeholder=document.getElementById('canvasPlaceholder'),button=document.getElementById('analyzeButton'),result=document.getElementById('inspectionResult'),selectionLabel=document.getElementById('selectionLabel');
const modeButtons=document.querySelectorAll('.mode-button');
let mode='solda',photo=new Image(),file=null,dragStart=null,selection=null,scale=1;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const buttonLabels={solda:'Analisar solda',termica:'Analisar imagem térmica'};
function point(event){const r=canvas.getBoundingClientRect();return{x:(event.clientX-r.left)*canvas.width/r.width,y:(event.clientY-r.top)*canvas.height/r.height};}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(photo,0,0,canvas.width,canvas.height);if(selection){const{x,y,w,h}=selection;ctx.fillStyle='rgba(207,104,64,.12)';ctx.fillRect(x,y,w,h);ctx.strokeStyle='#df7147';ctx.lineWidth=Math.max(2,canvas.width/500);ctx.setLineDash([8,5]);ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);}}
modeButtons.forEach(btn=>btn.addEventListener('click',()=>{mode=btn.dataset.mode;modeButtons.forEach(b=>b.classList.toggle('selected',b===btn));button.textContent=buttonLabels[mode];result.replaceChildren();}));
input.addEventListener('change',()=>{file=input.files[0];if(!file)return;result.replaceChildren();selection=null;selectionLabel.textContent='Nenhuma região marcada; a foto inteira será analisada.';photo=new Image();photo.onload=()=>{scale=Math.min(1,1100/photo.naturalWidth,600/photo.naturalHeight);canvas.width=Math.round(photo.naturalWidth*scale);canvas.height=Math.round(photo.naturalHeight*scale);canvas.hidden=false;placeholder.hidden=true;draw();button.disabled=false;};photo.src=URL.createObjectURL(file);});
canvas.addEventListener('pointerdown',event=>{if(!photo.src)return;canvas.setPointerCapture(event.pointerId);dragStart=point(event);selection={x:dragStart.x,y:dragStart.y,w:0,h:0};});
canvas.addEventListener('pointermove',event=>{if(!dragStart)return;const p=point(event);selection={x:Math.min(dragStart.x,p.x),y:Math.min(dragStart.y,p.y),w:Math.abs(p.x-dragStart.x),h:Math.abs(p.y-dragStart.y)};draw();});
canvas.addEventListener('pointerup',()=>{if(!dragStart)return;dragStart=null;if(selection.w<8||selection.h<8){selection=null;selectionLabel.textContent='Nenhuma região marcada; a foto inteira será analisada.';}else selectionLabel.textContent=`Região selecionada: ${Math.round(selection.w/scale)} × ${Math.round(selection.h/scale)} pixels.`;draw();result.replaceChildren();});
function renderTorchResult(data){
  const isThermal=data.tipo==='termica';
  const cls=data.avaliacao.startsWith('Revisar')?'verdict-review':data.avaliacao==='Imagem inconclusiva'?'verdict-unknown':'verdict-clear';
  const reasons=data.motivos.map(x=>`<li>${esc(x)}</li>`).join('');
  const details=isThermal
    ?`<dl><div><dt>Região quente (HSV)</dt><dd>${data.regiao_quente_percentual}%</dd></div><div><dt>Zona crítica</dt><dd>${data.zona_critica_percentual}%</dd></div><div><dt>Zonas críticas</dt><dd>${data.zonas_criticas}</dd></div><div><dt>Maior zona</dt><dd>${data.maior_zona_percentual}%</dd></div></dl>`
    :`<dl><div><dt>Possíveis poros</dt><dd>${data.possiveis_poros}</dd></div><div><dt>Área estimada</dt><dd>${data.area_porosa_estimada_percentual}%</dd></div><div><dt>Bordas na região</dt><dd>${data.densidade_bordas_percentual}%</dd></div><div><dt>Nitidez</dt><dd>${data.nitidez}</dd></div></dl>`;
  const caption=isThermal
    ?'Zonas quentes destacadas (HSV) — regiões avermelhadas indicam o possível sobreaquecimento.'
    :'Mapa visual sobre a foto — áreas quentes indicam contraste/textura.';
  const heading=isThermal?'O que o filtro HSV encontrou':'O que o filtro encontrou';
  return `<div class="verdict ${cls}">${esc(data.avaliacao)}</div><p class="assessment-note">${esc(data.observacao)}</p><div class="heat-result"><figure><img src="data:image/png;base64,${data.mapa_visual_base64}" alt="Imagem analisada com regiões destacadas"><figcaption>${caption}</figcaption></figure><div class="analysis-details"><h3>${heading}</h3><ul>${reasons}</ul>${details}</div></div>`;
}
button.addEventListener('click',async()=>{if(!file)return;const endpoint=mode==='termica'?'/inspect/thermal':'/inspect/weld';button.disabled=true;button.textContent='Analisando...';result.innerHTML='<p>Analisando a região selecionada...</p>';const form=new FormData();form.append('imagem',file);if(selection){form.append('roi_x',Math.round(selection.x/scale));form.append('roi_y',Math.round(selection.y/scale));form.append('roi_w',Math.round(selection.w/scale));form.append('roi_h',Math.round(selection.h/scale));}
  try{const response=await fetch(endpoint,{method:'POST',body:form}),data=await response.json();if(!response.ok)throw Error(data.detail||'Falha na análise');result.innerHTML=renderTorchResult(data);
  }catch(error){result.innerHTML=`<p class="error-text">${esc(error.message)}</p>`;}finally{button.disabled=false;button.textContent=buttonLabels[mode];}});
