import gradio as gr
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from nltk.corpus import stopwords
import nltk

# 1. Configuração Inicial e Download de dependências do NLTK
nltk.download('stopwords', quiet=True)
print("✅ Ambiente 'CyberShield Recomeço' pronto!")

# =====================================================================
# TREINAMENTO DO MODELO 
# =====================================================================
# Link RAW do GitHub para os datasets
url_seu_git = "https://raw.githubusercontent.com/IlkerSobrinho/iAxSpam/main/dataset_cyber.csv"

def extract_features_pt(text):
    text = str(text).lower()
    has_link = 1 if re.search(r'http|https|www', text) else 0
    # Gatilhos separados por "intenção"
    p_terms = ['bloqueio', 'suspens', 'verificar', 'acesso', 'senha', 'perder', 'urgente', 'clique aqui']
    s_terms = ['ganhou', 'prêmio', 'desconto', 'oferta', 'promoção', 'pix', 'sorteio']

    p_score = sum(1 for w in p_terms if w in text)
    s_score = sum(1 for w in s_terms if w in text)

    return [has_link, p_score, s_score], (p_terms, s_terms)

try:
    df_final = pd.read_csv(url_seu_git).dropna()

    # N-grams ajudam a pegar contextos como "clique aqui" ou "conta suspensa"
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df_final['text'])

    extra_feats = [extract_features_pt(t)[0] for t in df_final['text']]
    X_final = hstack([X_tfidf, np.array(extra_feats)])

    # O segredo: class_weight='balanced' ajuda a IA a não viciar em uma só categoria
    model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
    model.fit(X_final, df_final['label'])

    joblib.dump(model, 'modelo_shields_PT.pkl')
    joblib.dump(tfidf, 'vetorizador_shields_PT.pkl')
    print(f"✅ IA Calibrada e pronta! Classes: {df_final['label'].unique()}")

except Exception as e:
    print(f"❌ Erro no treino: {e}")


# =====================================================================
# CARREGAMENTO DO MODELO E INTERFACE GRADIO
# =====================================================================
model_gr = joblib.load('modelo_shields_PT.pkl')
tfidf_gr = joblib.load('vetorizador_shields_PT.pkl')

def auditoria_avancada(texto):
    if not texto.strip(): return "Insira um texto.", 0, "Aguardando entrada..."

    vetor, (p_terms, s_terms) = extract_features_pt(texto)
    X_text = tfidf_gr.transform([texto])
    X_input = hstack([X_text, np.array([vetor])])

    probs = model_gr.predict_proba(X_input)[0]
    classes = model_gr.classes_
    idx = np.argmax(probs)
    classe = classes[idx]
    confianca = probs[idx] * 100

    colors = {"Phishing": "#ff4b4b", "Spam": "#ffa500", "Legitimo": "#00ff7f"}
    color = colors.get(classe, "#ffffff")

    aviso_incerteza = ""
    if confianca < 35:
        aviso_incerteza = "<p style='color:yellow;'>⚠️ <b>Análise de Baixa Confiança:</b> O modelo está em dúvida.</p>"

    veredito_html = f"""
    <div style='text-align:center; padding:20px; border-radius:10px; background-color:{color}22; border:2px solid {color}'>
        <h1 style='color:{color}; margin:0;'>{classe.upper()}</h1>
        {aviso_incerteza}
        <p style='color:white;'>Precisão Estatística: {confianca:.1f}%</p>
    </div>
    """

    explica = f"### 🔍 Análise de Vetores de Ataque\n"
    detect_p = [w for w in p_terms if w in texto.lower()]
    detect_s = [w for w in s_terms if w in texto.lower()]

    if detect_p: explica += f"- 🚨 **Engenharia Social:** Encontramos termos de pressão: `{', '.join(detect_p)}`.\n"
    if detect_s: explica += f"- 💰 **Marketing Agressivo:** Gatilhos de spam: `{', '.join(detect_s)}`.\n"
    if vetor[0]: explica += "- 🔗 **Vetor de Redirecionamento:** Contém links externos.\n"

    explica += f"\n---\n**Metodologia:** Auditoria híbrida via Random Forest (NLP + Heurística)."

    return veredito_html, confianca/100, explica

# Interface Customizada
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="slate")) as demo:
    gr.Markdown("# 🛡️ CyberShield Auditor: Segurança Inteligente")
    gr.Markdown("Auditoria de ameaças em tempo real para comunicações digitais.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Mensagem para Auditoria",
                lines=10,
                placeholder="Cole o e-mail suspeito aqui..."
            )
            with gr.Row():
                btn_clear = gr.ClearButton([input_text], value="🗑️ Limpar")
                btn_run = gr.Button("⚡ Analisar Segurança", variant="primary")

            # EXEMPLOS PARA A APRESENTAÇÃO (para facilitar na apresentacao)
            gr.Examples(
                examples=[
                    ["ASSUNTO: Sua casa protegida por menos de R$ 2 por dia | CORPO: Monitoramento 24h, câmeras HD e alarme inteligente com instalação grátis. Proteja sua família e seu patrimônio com a Verisure. Peça seu orçamento online agora."],
                    ["ASSUNTO: GitHub: [Ação Necessária] Habilite o 2FA na sua conta | CORPO: Para manter a segurança da comunidade, o GitHub passará a exigir autenticação de dois fatores em todos os perfis de desenvolvedores. Configure o seu agora."],
                    ["ASSUNTO: Alerta Crítico: Bypass de Autenticação Multifator (MFA) detectado | CORPO: Ilker, nosso SOC identificou uma tentativa de login bem-sucedida que contornou o token SMS. Localização: Kiev, Ucrânia. O acesso foi bloqueado preventivamente. Para validar sua identidade e recuperar o controle total, realize o procedimento de auditoria forçada no servidor seguro: http://mfa-recovery-portal.com"]
                ],
                inputs=input_text,
                label="Clique para Testar Casos Reais"
            )

        with gr.Column(scale=1):
            output_html = gr.HTML()
            output_prob = gr.Label(label="Probabilidade")
            output_md = gr.Markdown()

    # DASHBOARD DE MÉTRICAS
    with gr.Accordion("📊 Estatísticas do Projeto (A3 - 2026)", open=False):
        gr.Markdown(f"""
        - **Dataset:** ~400+ amostras (Phishing, Spam, Legitimo)
        - **Modelo:** Random Forest Classifier (150 estimadores)
        - **Tecnologia:** Scikit-Learn + NLP (TF-IDF)
        - **Desenvolvedor:** Ilker Sobrinho
        """)

    btn_run.click(
        fn=auditoria_avancada,
        inputs=input_text,
        outputs=[output_html, output_prob, output_md]
    )

# Bloqueio if __name__ necessário para servidores Web
if __name__ == "__main__":
    demo.launch()
