# 🛡️ CyberShield Auditor: Segurança Inteligente

![Status](https://img.shields.io/badge/Status-Online-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Model-Random_Forest-purple)

Auditor de Cibersegurança desenvolvido para detectar ameaças em comunicações digitais (Phishing, Spam e e-mails Legítimos) em tempo real, utilizando Processamento de Linguagem Natural (NLP) e Heurística. 

Projeto desenvolvido para a avaliação **A3** do curso de Ciência da Computação.

## 🚀 Acesse a Aplicação ao Vivo
O modelo está treinado e hospedado no Hugging Face Spaces. Você pode testar a ferramenta agora mesmo sem precisar instalar nada:

👉 **[Testar o CyberShield Auditor](https://huggingface.co/spaces/Nizakix/CyberShield-Auditor)**

## 🧠 Metodologia
- **Dataset:** Base de dados própria curada com mais de 350 amostras (Phishing profissional, Spams reais de e-commerce e notificações sistêmicas de serviços e bancos).
- **Vetorização:** TF-IDF (`TfidfVectorizer`) com N-grams.
- **Algoritmo:** `RandomForestClassifier` com 150 estimadores e pesos balanceados.
- **Engenharia de Features:** Script heurístico que analisa vetores de ataque como urgência, pressão psicológica e links externos.
- **Deploy:** MLOps contínuo via Hugging Face e interface Gradio.

## 💻 Desenvolvedor
- **Ilker Sobrinho (Nizakix)**
