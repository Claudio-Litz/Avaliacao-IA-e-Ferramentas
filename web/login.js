const formulario = document.getElementById('demoLoginForm');
const mensagemLogin = document.getElementById('loginMessage');

formulario.addEventListener('submit', async (evento) => {
  evento.preventDefault();
  const botao = formulario.querySelector('button[type="submit"]');
  botao.disabled = true;
  mensagemLogin.textContent = 'Preparando o acesso demonstrativo…';

  try {
    const resposta = await fetch('/demo-login', {
      method: 'POST',
      body: new FormData(formulario)
    });
    if (!resposta.ok) throw new Error('Não foi possível iniciar a sessão. Tente novamente.');
    window.location.replace('/dashboard');
  } catch (erro) {
    mensagemLogin.textContent = erro.message;
    botao.disabled = false;
  }
});
