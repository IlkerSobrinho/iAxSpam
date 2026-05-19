Resumo do Projeto

O CyberShield Auditor é uma ferramenta de segurança inteligente projetada para auditar comunicações digitais (e-mails) e classificar ameaças em tempo real. Diferente de filtros comuns, o sistema utiliza uma abordagem híbrida que combina Processamento de Linguagem Natural (NLP) com heurísticas de segurança para identificar Phishing, Spam e e-mails Legítimos, fornecendo uma explicação clara dos vetores de ataque detectados.

Problema e Motivação

Com o aumento de 40% nos ataques de engenharia social em 2025, os usuários enfrentam dificuldades em diferenciar e-mails transacionais reais de tentativas de roubo de credenciais. O projeto visa reduzir a carga cognitiva do usuário, automatizando a auditoria de mensagens suspeitas através de modelos de Machine Learning.

Metodologia e Tecnologias

Para garantir a robustez e transparência da ferramenta, foram utilizadas as seguintes tecnologias:

    Linguagem: Python 3.10+

    Algoritmo de Aprendizado: Random Forest Classifier (150 Estimadores). Escolhido por sua capacidade de evitar overfitting através do consenso entre múltiplas árvores de decisão.

    NLP (Processamento de Linguagem Natural): Vetorização TF-IDF para análise da relevância de termos em vez de apenas contagem simples.

    Heurística de Segurança: Script customizado para detecção de vetores de redirecionamento (links) e análise semântica de pressão/urgência.

    Interface: Framework Gradio para uma experiência de usuário (UX) focada em auditoria rápida.

Desenvolvimento Técnico
O Dataset

O modelo foi treinado com um banco de dados curado de aproximadamente 350 amostras, composto por:

    Dados Reais: E-mails coletados e anonimizados de serviços como gov.br, bancos, e-commerces e sistemas acadêmicos.

    Dados Sintéticos: Expansão da base para incluir variações de ataques de Spear Phishing e marketing agressivo internacional.

Explicabilidade (XAI)

Um diferencial do CyberShield é a Auditoria de Vetores de Ataque. O sistema não fornece apenas um veredito, mas isola os termos que causaram a classificação, como:

    Engenharia Social: Identificação de gatilhos mentais (ex: "urgente", "bloqueio", "contas").

    Marketing Agressivo: Identificação de padrões de venda (ex: "oferta", "desconto", "grátis").

    Vetor de Redirecionamento: Detecção de URLs externas que podem ocultar destinos maliciosos.

Resultados Preliminares

Atualmente, o modelo apresenta estabilidade estatística em e-mails complexos, como notificações de segundo fator de autenticação (2FA) e alertas de novos acessos (ex: Netflix, Instagram), reduzindo a taxa de falsos positivos através do balanceamento de pesos entre as classes.

Conclusão e Próximos Passos

O rascunho valida a viabilidade técnica da solução. A etapa final consistirá na ampliação do dataset para 500+ amostras e no ajuste fino dos limites de confiança (thresholds) para garantir que comunicações críticas não sejam erroneamente bloqueadas.
