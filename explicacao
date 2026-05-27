import gradio as gr # Gradio: Cria a interface gráfica (a tela bonitinha) sem precisarmos programar em HTML/CSS.
import joblib       # Joblib: Serve para salvar e carregar a IA depois de pronta (como se fosse o "Save Game").
import pandas as pd # Pandas: É o "Excel do Python". Serve para ler e organizar nosso arquivo CSV com os e-mails.
import numpy as np  # Numpy: Faz cálculos matemáticos avançados (a IA só entende números, não texto).
import re           # re (Regex): Um buscador de texto avançado. Usamos para caçar links (http, www) na mensagem.
from sklearn.feature_extraction.text import TfidfVectorizer # Tradutor que transforma as palavras do e-mail em números.
from sklearn.ensemble import RandomForestClassifier         # O nosso Algoritmo de IA (O cérebro do projeto).
from scipy.sparse import hstack                             # Cola listas de números umas nas outras.
from nltk.corpus import stopwords                           # Dicionário de palavras inúteis (o, a, de, para).
import nltk         # NLTK: Biblioteca especialista em entender a linguagem humana (Processamento de Linguagem Natural).

# =====================================================================
# 1. PREPARAÇÃO DO AMBIENTE
# =====================================================================
# Baixa as palavras vazias (stopwords) para o sistema não perder tempo com "de", "para", "com".
nltk.download('stopwords', quiet=True)
print("✅ Ambiente 'CyberShield Recomeço' pronto!")

# =====================================================================
# 2. TREINAMENTO DO MODELO (A FASE ONDE A IA ESTUDA E APRENDE)
# =====================================================================
# Link de onde os dados (e-mails reais) estão vindo.
url_seu_git = "https://raw.githubusercontent.com/IlkerSobrinho/iAxSpam/main/dataset_cyber.csv"

def extract_features_pt(text):
    """
    FUNÇÃO DE HEURÍSTICA (AS REGRAS MANUAIS)
    Aqui a gente não usa IA ainda. É um filtro manual para ajudar a IA depois.
    Se perguntarem na apresentação: "Isso é a nossa Defesa em Profundidade".
    """
    text = str(text).lower() # Transforma tudo em minúsculo para facilitar a busca.
    
    # 1. Tem link no e-mail? Hackers adoram links. Se achar http ou www, marca 1 (Sim). Se não, 0 (Não).
    has_link = 1 if re.search(r'http|https|www', text) else 0
    
    # 2. Palavras de Engenharia Social (O cara quer te colocar medo ou urgência)
    p_terms = ['bloqueio', 'suspens', 'verificar', 'acesso', 'senha', 'perder', 'urgente', 'clique aqui']
    # 3. Palavras de Spam (O cara quer te vender algo a todo custo)
    s_terms = ['ganhou', 'prêmio', 'desconto', 'oferta', 'promoção', 'pix', 'sorteio']

    # Conta quantas palavras suspeitas apareceram no texto.
    p_score = sum(1 for w in p_terms if w in text)
    s_score = sum(1 for w in s_terms if w in text)

    # Devolve uma lista de 3 números: [TemLink?, PontosDePhishing, PontosDeSpam] e as palavras que ele achou.
    return [has_link, p_score, s_score], (p_terms, s_terms)

# BLOCO DE TREINO DA IA (Só roda uma vez para criar o modelo)
try:
    # Lê o CSV da internet e joga as linhas vazias no lixo (.dropna)
    df_final = pd.read_csv(url_seu_git).dropna()

    # TF-IDF: O TRADUTOR DE TEXTO PARA MATEMÁTICA
    # max_features=1500: Pega só as 1500 palavras mais importantes, ignora o resto pra ficar rápido.
    # ngram_range=(1,2): Lê palavras sozinhas e também pares de palavras (ex: "conta" e "conta bloqueada").
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
    
    # Aqui ele lê todos os textos do CSV e transforma num planilhão de números.
    X_tfidf = tfidf.fit_transform(df_final['text'])

    # Passa todos os e-mails na nossa função manual lá de cima pra pegar os pontos de cada um.
    extra_feats = [extract_features_pt(t)[0] for t in df_final['text']]
    
    # Junta os números do TF-IDF com os números da nossa função manual.
    X_final = hstack([X_tfidf, np.array(extra_feats)])

    # O CÉREBRO: RANDOM FOREST (Floresta Aleatória)
    # n_estimators=150: Ele não cria 1 regra, ele cria 150 árvores de decisão. Elas vão "votar" qual a resposta certa.
    # class_weight='balanced': MUITO IMPORTANTE. Se tiver muito e-mail Legítimo e pouco Phishing no CSV, 
    # ele equilibra os pesos para a IA não ficar preguiçosa e chutar tudo como Legítimo.
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    
    # .fit é o comando mágico. É aqui que a máquina efetivamente APRENDE olhando os dados.
    model.fit(X_final, df_final['label'])

    # Aqui a gente "salva o jogo". Salva o cérebro treinado em arquivos .pkl para não ter que treinar tudo de novo amanhã.
    joblib.dump(model, 'modelo_shields_PT.pkl')
    joblib.dump(tfidf, 'vetorizador_shields_PT.pkl')
    print(f"✅ IA Calibrada e pronta! Classes: {df_final['label'].unique()}")

except Exception as e:
    print(f"❌ Erro no treino: {e}")


# =====================================================================
# 3. APLICAÇÃO (A FASE ONDE O USUÁRIO USA O SISTEMA PRONTO)
# =====================================================================
# Carrega os cérebros que salvamos ali em cima.
model_gr = joblib.load('modelo_shields_PT.pkl')
tfidf_gr = joblib.load('vetorizador_shields_PT.pkl')

def auditoria_avancada(texto):
    """
    ESSA FUNÇÃO É O QUE ACONTECE QUANDO O USUÁRIO CLICA NO BOTÃO "ANALISAR" NA TELA.
    """
    if not texto.strip(): return "Insira um texto.", 0, "Aguardando entrada..."

    # 1. Passa o texto do usuário no filtro manual (Heurística)
    vetor, (p_terms, s_terms) = extract_features_pt(texto)
    
    # 2. Transforma o texto do usuário em matemática usando o tradutor salvo
    X_text = tfidf_gr.transform([texto])
    
    # 3. Junta tudo num formato só
    X_input = hstack([X_text, np.array([vetor])])

    # 4. Faz a previsão! .predict_proba pede a PORCENTAGEM de certeza que a IA tem (ex: 90% de chance de ser Spam).
    probs = model_gr.predict_proba(X_input)[0]
    classes = model_gr.classes_ # Pega os nomes: Phishing, Spam, Legitimo
    
    # Pega qual categoria recebeu mais votos das 150 árvores.
    idx = np.argmax(probs)
    classe = classes[idx]
    
    # Transforma a probabilidade (0.90) em porcentagem bonita (90%).
    confianca = probs[idx] * 100

    # Define as cores bonitinhas que vão aparecer na tela dependendo do perigo.
    colors = {"Phishing": "#ff4b4b", "Spam": "#ffa500", "Legitimo": "#00ff7f"}
    color = colors.get(classe, "#ffffff")

    # REGRA DE SEGURANÇA: Se a IA estiver com menos de 35% de certeza, ela avisa que tá confusa. 
    # Isso é bom pra mostrar pros professores que o sistema é transparente e não mente.
    aviso_incerteza = ""
    if confianca < 35:
        aviso_incerteza = "<p style='color:yellow;'>⚠️ <b>Análise de Baixa Confiança:</b> O modelo está em dúvida.</p>"

    # Monta a caixinha colorida que aparece na tela (HTML puro)
    veredito_html = f"""
    <div style='text-align:center; padding:20px; border-radius:10px; background-color:{color}22; border:2px solid {color}'>
        <h1 style='color:{color}; margin:0;'>{classe.upper()}</h1>
        {aviso_incerteza}
        <p style='color:white;'>Precisão Estatística: {confianca:.1f}%</p>
    </div>
    """

    # Monta a explicação de por que a IA tomou essa decisão (para o usuário não achar que é mágica)
    explica = f"### 🔍 Análise de Vetores de Ataque\n"
    detect_p = [w for w in p_terms if w in texto.lower()]
    detect_s = [w for w in s_terms if w in texto.lower()]

    if detect_p: explica += f"- 🚨 **Engenharia Social:** Encontramos termos de pressão: `{', '.join(detect_p)}`.\n"
    if detect_s: explica += f"- 💰 **Marketing Agressivo:** Gatilhos de spam: `{', '.join(detect_s)}`.\n"
    if vetor[0]: explica += "- 🔗 **Vetor de Redirecionamento:** Contém links externos.\n"

    explica += f"\n---\n**Metodologia:** Auditoria híbrida via Random Forest (NLP + Heurística)."

    # Devolve 3 coisas para a interface gráfica: O HTML colorido, a barrinha de porcentagem e o texto da explicação.
    return veredito_html, confianca/100, explica

# =====================================================================
# 4. A INTERFACE GRÁFICA (O FRONT-END)
# =====================================================================
# Isso aqui desenha a tela web. Dividimos em 2 colunas.
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="slate")) as demo:
    gr.Markdown("# 🛡️ CyberShield Auditor: Segurança Inteligente")
    gr.Markdown("Auditoria de ameaças em tempo real para comunicações digitais.")

    with gr.Row(): # Cria uma linha na tela
        with gr.Column(scale=1): # Coluna da Esquerda (Onde o usuário digita)
            input_text = gr.Textbox(
                label="Mensagem para Auditoria",
                lines=10,
                placeholder="Cole o e-mail suspeito aqui..."
            )
            with gr.Row():
                btn_clear = gr.ClearButton([input_text], value="🗑️ Limpar")
                btn_run = gr.Button("⚡ Analisar Segurança", variant="primary")

            # ATENÇÃO GRUPO: Na hora de apresentar para a banca, é só clicar nesses textos aqui. 
            # Não precisa digitar nada ao vivo pra não passar vergonha se der erro de digitação.
            gr.Examples(
                examples=[
                    ["ASSUNTO: Sua casa protegida por menos de R$ 2 por dia | CORPO: Monitoramento 24h, câmeras HD e alarme inteligente com instalação grátis. Proteja sua família e seu patrimônio com a Verisure. Peça seu orçamento online agora."],
                    ["ASSUNTO: GitHub: [Ação Necessária] Habilite o 2FA na sua conta | CORPO: Para manter a segurança da comunidade, o GitHub passará a exigir autenticação de dois fatores em todos os perfis de desenvolvedores. Configure o seu agora."],
                    ["ASSUNTO: Alerta Crítico: Bypass de Autenticação Multifator (MFA) detectado | CORPO: Ilker, nosso SOC identificou uma tentativa de login bem-sucedida que contornou o token SMS. Localização: Kiev, Ucrânia. O acesso foi bloqueado preventivamente. Para validar sua identidade e recuperar o controle total, realize o procedimento de auditoria forçada no servidor seguro: http://mfa-recovery-portal.com"]
                ],
                inputs=input_text,
                label="Clique para Testar Casos Reais"
            )

        with gr.Column(scale=1): # Coluna da Direita (Onde sai o resultado colorido)
            output_html = gr.HTML()
            output_prob = gr.Label(label="Probabilidade")
            output_md = gr.Markdown()

    # Painel de informações para dar moral no projeto.
    with gr.Accordion("📊 Estatísticas do Projeto (A3 - 2026)", open=False):
        gr.Markdown(f"""
        - **Dataset:** ~400+ amostras (Phishing, Spam, Legitimo)
        - **Modelo:** Random Forest Classifier (150 estimadores)
        - **Tecnologia:** Scikit-Learn + NLP (TF-IDF)
        - **Desenvolvedor:** Ilker Sobrinho
        """)

    # Liga o botão "Analisar" à função "auditoria_avancada" que criamos lá em cima.
    btn_run.click(
        fn=auditoria_avancada,
        inputs=input_text,
        outputs=[output_html, output_prob, output_md]
    )

# Comando obrigatório para manter o site no ar (Rodar o servidor web localmente)
if __name__ == "__main__":
    demo.launch()
