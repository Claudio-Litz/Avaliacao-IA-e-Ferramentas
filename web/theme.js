(() => {
  const chave = 'vulcantech-tema';
  const html = document.documentElement;
  const botoes = document.querySelectorAll('.theme-toggle');

  function aplicarTema(tema) {
    html.dataset.theme = tema;
    localStorage.setItem(chave, tema);
    botoes.forEach((botao) => {
      const escuro = tema === 'dark';
      botao.innerHTML = escuro ? '☀ <span>Tema</span>' : '☾ <span>Tema</span>';
      botao.setAttribute('aria-label', escuro ? 'Ativar tema claro' : 'Ativar tema escuro');
      botao.title = escuro ? 'Mudar para tema claro' : 'Mudar para tema escuro';
    });
  }

  aplicarTema(localStorage.getItem(chave) || 'light');
  botoes.forEach((botao) => {
    botao.addEventListener('click', () => {
      aplicarTema(html.dataset.theme === 'dark' ? 'light' : 'dark');
    });
  });
})();
