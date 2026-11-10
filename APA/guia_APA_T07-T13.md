# Guía de estudio · Aprendizaje Automático (APA) · Temas T07–T13

Basada en las transparencias y boletines de actividades de APA 2026/27 (ETSINF, DSIC, UPV).

## Índice

1. [Cómo usar esta guía](#uso)
2. [Glosario para preguntas tipo test](#glosario)
   - [T07 Decisión](#g07)
   - [T08 Análisis discriminante](#g08)
   - [T09 Regresión logística](#g09)
   - [T10 Regresión lineal](#g10)
   - [T11 Métodos basados en ejemplos](#g11)
   - [T12 Árboles, bagging, bosques y boosting](#g12)
   - [T13 Aprendizaje no supervisado](#g13)
3. [Teoría con la técnica Feynman](#teoria)
   - [T07 Decisión](#t07)
   - [T08 Análisis discriminante](#t08)
   - [T09 Regresión logística](#t09)
   - [T10 Regresión lineal](#t10)
   - [T11 Métodos basados en ejemplos](#t11)
   - [T12 Árboles, bagging, bosques y boosting](#t12)
   - [T13 Aprendizaje no supervisado](#t13)
4. [Recetas paso a paso para los ejercicios](#recetas)
   - [Estrategia para preguntas tipo test](#r-test)
   - [T07](#r07) · [T08](#r08) · [T09](#r09) · [T10](#r10) · [T11](#r11) · [T12](#r12) · [T13](#r13)
5. [Chuleta para los últimos 10 minutos](#chuleta)

---

<a id="uso"></a>
## 1. Cómo usar esta guía

Cada tema de la sección 3 sigue el mismo esquema:

- **Idea clave**: el tema en una o dos frases.
- **Explicación para un chaval de 12 años**: analogía cotidiana, sin tecnicismos.
- **Fórmulas imprescindibles**.
- **Dudas típicas**: vacíos de quien empieza, aclarados con la misma técnica.

Método recomendado:

1. Lee la explicación sencilla y cuéntala en voz alta sin mirar.
2. Comprueba en la tabla de dudas qué te has dejado.
3. Repasa el glosario del tema (sección 2): es lo que más cae en el test.
4. Practica con las recetas de la sección 4.

| Tema | Idea en una frase | Ejercicio típico |
|---|---|---|
| T07 Decisión | Elegir la acción con menos pérdida esperada usando probabilidades | Bayes, riesgo, matriz de confusión, ROC/PR, F1 |
| T08 Discriminante | Modelar cada clase con una gaussiana y comparar | Estimar medias/covarianzas, frontera QDA/LDA |
| T09 Logística | Clasificador lineal que da probabilidades (sigmoide/softmax) | Probabilidades, regiones, una iteración de gradiente |
| T10 Regresión lineal | Ajustar una recta/plano minimizando el error cuadrático | Mínimos cuadrados a mano, densidad, $R^2$ |
| T11 Ejemplos | Decidir mirando a los datos más parecidos | KNN con empates, Parzen, Nadaraya-Watson |
| T12 Árboles | Preguntas sí/no que separen bien las clases; ensambles | Gini/entropía, mejor corte, boosting, AdaBoost |
| T13 No supervisado | Encontrar grupos en datos sin etiquetas | Densidad de una mezcla, iteración de EM, C-medias |

---

<a id="glosario"></a>
## 2. Glosario para preguntas tipo test

La columna **Ojo** recoge la trampa habitual o el detalle que suelen cambiar en las opciones incorrectas.

<a id="g07"></a>
### T07 · Decisión

**Conceptos básicos de decisión bayesiana**

| Término | Definición (versión examen) | Ojo |
|---|---|---|
| Estados de la naturaleza $H$ | Conjunto de hipótesis de interés | En clasificación $H = A = Y = \{1,\dots,C\}$ |
| Prior $p(h)$ | Probabilidad estimada antes de ver los datos | — |
| Evidencia $x$ | Datos observados tras estimar el prior | — |
| Verosimilitud $p(x\mid h)$ | Probabilidad de observar $x$ si $h$ es cierta | No es el posterior $p(h\mid x)$ |
| Verosimilitud marginal $p(x)$ | Probabilidad (no nula) de observar $x$: $\sum_h p(h)\,p(x\mid h)$ | No hace falta para el argmax |
| Posterior $p(h\mid x)$ | Probabilidad de $h$ tras ver $x$: $p(h)\,p(x\mid h)/p(x)$ | Es la regla de Bayes |
| Función de pérdida $\ell(h,a)$ | Coste de elegir $a$ cuando el estado real es $h$ | "Riesgo" = pérdida: mejor cuanto menor |
| Riesgo a posteriori $R(a\mid x)$ | Pérdida esperada: $\sum_h \ell(h,a)\,p(h\mid x)$ | — |
| Estimador de Bayes / política óptima | Acción de **mínimo** riesgo a posteriori | No "poco riesgo": el mínimo |
| Pérdida 0-1 | 0 si aciertas, 1 si fallas; $R(\hat y\mid x) = 1 - p(\hat y\mid x)$ | — |
| Clasificador MAP | Bayes con pérdida 0-1: $\arg\max_{\hat y} p(\hat y\mid x) = \arg\max_{\hat y} p(\hat y)\,p(x\mid \hat y)$ | Las tres formas ($\arg\min R$, argmax del posterior, argmax de prior × verosimilitud) son válidas |
| Clasificación sensible al coste | Bayes con costes de error distintos | Tiene menor riesgo que MAP si los costes difieren |
| Opción de rechazo | Acción extra "0" con coste $\lambda_r < \lambda_e$ | $R(0\mid x) = \lambda_r$; $R(\hat y\mid x) = \lambda_e\,(1 - p(\hat y\mid x))$ |
| Regresor de Bayes L2 (MMSE) | Predice la media a posteriori $E[y\mid x]$ | Es la regla estándar en regresión |

**Matrices de confusión y tasas**

| Término | Definición | Ojo |
|---|---|---|
| Matriz de confusión | Recuento de pares (real, reconocida) | Filas = real, columnas = reconocida; aciertos en la diagonal |
| Normalización por filas | $p(\hat y\mid y)$: "de los que eran $c$, ¿qué dije?" | Da TPR, FPR, FNR, TNR |
| Normalización por columnas | $p(y\mid \hat y)$: "de los que llamé $c$, ¿qué eran?" | Da PPV, NPV, FDR, FOR |
| Normalización global | $p(y,\hat y)$: divide por el total $M$ | — |
| Clasificador por umbral | $\hat y_\tau(x) = I\big(p(y=1\mid x) \ge 1 - \tau\big)$; $\tau$ = margen de positividad | $\tau = 0$: solo positivos seguros; $\tau = 1$: todo positivo |
| TPR (recall, sensibilidad, hit) | $TP/P$ | Es la $R$ de la curva PR |
| FPR (falsa alarma, error tipo I, fallout) | $FP/N$ | — |
| FNR (miss, error tipo II) | $FN/P = 1 - TPR$ | — |
| TNR (especificidad) | $TN/N$ | — |
| PPV (precisión) | $TP/\hat P$, con $\hat P$ = predichos positivos | Es la $P$ de la curva PR |
| NPV · FOR · FDR | $TN/\hat N$ · $FN/\hat N$ · $FP/\hat P$ | — |

**Curvas y medidas resumen**

| Término | Definición | Ojo |
|---|---|---|
| Curva ROC | $TPR$ en función de $FPR$, para $\tau$ de 0 a 1 | No al revés |
| ROC perfecta / azar / realista | Perfecta: $(0,1)$ y $(1,1)$. Azar: diagonal. Realista: creciente sobre la diagonal | Mejor cuanto más arriba |
| AUC | Área bajo la ROC | **Mayor** = mejor |
| EER (equal error rate, cross-over rate) | $FPR$ tal que $FPR = FNR$; abscisa del corte con la diagonal inversa $(1,0)$–$(0,1)$ | **Menor** = mejor |
| Curva PR | Precisión en función de recall | Útil si "negativo" es difuso o los positivos son raros |
| PR perfecta / azar / realista | Perfecta: $(1,1)$ y $(1, P/M)$. Azar: horizontal a $P/M$. Realista: aprox. decreciente sobre $P/M$ | El recall crece con $\tau$; la precisión no siempre baja |
| PR interpolada | Sustituye $\mathcal P_\tau$ por $\max_{\tilde\tau \ge \tau} \mathcal P_{\tilde\tau}$ | Se asume interpolada salvo aviso |
| AP | Área bajo la PR (precisión media) | — |
| mAP | Media de varias AP (p. ej. una por clase) | — |
| Medidas @K | Calculadas con los $K$ datos más probables (P@K, AP@K, mAP@K) | — |
| $F_\beta$ | $(1+\beta^2)\,\dfrac{PR}{\beta^2 P + R}$; $\beta$ = importancia de $R$ frente a $P$ | $F_0 = P$; $F_\infty = R$ |
| $F_1$ | Media armónica de $P$ y $R$ | Exige que ambas sean altas |
| Media aritmética de $P$ y $R$ | Llega a 0,5 aunque $P = 0$ | Por eso se usa $F$ |
| Macro $F_1$ | Media simple de los $F_1$ por clase | — |
| Weighted $F_1$ | Media de $F_1$ por clase ponderada por el prior de clase | — |
| Micro $F_1$ | $F_1$ global $= TP/M$ (traza / total) | $FN = FP$ = suma fuera de la diagonal |

<a id="g08"></a>
### T08 · Análisis discriminante

| Término | Definición | Ojo |
|---|---|---|
| Clasificador generativo | Usa Bayes con prior $p(y)$ y condicionales de clase $p(x\mid y)$ | — |
| Interpretación generativa | $(x,y)$ se genera en dos pasos: $y \sim p(y)$, luego $x \sim p(x\mid y)$ | — |
| Condicional de clase | $p(x\mid y)$ | — |
| Incondicional | $p(x) = \sum_y p(y)\,p(x\mid y)$ | No hace falta para clasificar |
| NLL | $-\sum_n \log p(x_n,y_n\mid\theta) = -\sum_c N_c \log \pi_c - \sum_n \log p(x_n\mid y_n,\theta')$ | Las tres formas del boletín son válidas |
| MLE del prior | $\hat\pi_c = N_c/N$ (frecuencia relativa) | No depende de $\theta'$ |
| MLE de las condicionales | Con parámetros independientes, $\theta'_c$ solo depende de los datos de la clase $c$ | Depende de la forma de las condicionales |
| Funciones discriminantes | $y(x) = \arg\max_c g_c(x)$ | — |
| Discriminantes equivalentes | Mismo argmax. Valen: $\log g_c$ (si $g_c \ge 0$), $a\,g_c + b$ con $a > 0$, $e^{g_c}$, $\arg\min_c -g_c$ | Se pueden quitar términos comunes a todas las clases |
| QDA | MAP con condicionales gaussianas $\mathcal N(\mu_c, \Sigma_c)$ independientes | Frontera cuadrática |
| $A_c$ · $\beta_c$ · $\gamma_c$ (QDA) | $-\tfrac12\Sigma_c^{-1}$ · $\Sigma_c^{-1}\mu_c$ · $\log\pi_c - \tfrac12\log\det\Sigma_c - \tfrac12\mu_c^t\Sigma_c^{-1}\mu_c$ | $g_c(x) = x^tA_cx + \beta_c^tx + \gamma_c$ |
| Región / frontera | $R_c = \{x : g_c(x) \ge \max_{c'\ne c} g_{c'}(x)\}$; $F_c$: igualdad; entre varias clases: intersección | — |
| MLE en QDA | $\hat\mu_c$ = media de la clase; $\hat\Sigma_c$ dividiendo por $N_c$ | Como una sola gaussiana por clase |
| Regularización | $\tilde\Sigma_c = (1-\lambda)\hat\Sigma_c + \lambda\hat\Sigma$, con $0 \le \lambda \le 1$ | $\lambda = 1$ da LDA |
| LDA | QDA máximamente regularizado: $\Sigma$ común | Frontera lineal |
| $\beta_c$ · $\gamma_c$ (LDA) | $\Sigma^{-1}\mu_c$ · $\log\pi_c - \tfrac12\mu_c^t\Sigma^{-1}\mu_c$ | Clases equiprobables: $\gamma_c = -\tfrac12\mu_c^t\Sigma^{-1}\mu_c$ |
| $\hat\Sigma$ común | $\sum_c \hat\pi_c\hat\Sigma_c = \frac1N\sum_c\sum_{n:y_n=c}(x_n-\hat\mu_c)(x_n-\hat\mu_c)^t$ | Las dos expresiones son iguales |
| NCM | LDA con priors idénticos = media de clase más cercana | $\gamma_c$ lleva signo **menos** |
| Distancia de Mahalanobis | $\lVert z\rVert_W^2 = z^tW^tWz$ con $W^tW = \Sigma^{-1}$ | Con $\Sigma = I$ es la euclídea |
| Uso de NCM | Muchas clases y muy pocos datos por clase (biometría) | — |
| Naive Bayes | MAP con características independientes dada la clase: $p(x\mid c) = \prod_d p(x_d\mid c)$ | "Ingenuo" pero sencillo y a veces muy preciso |
| NB gaussiano | NB con gaussianas 1d = QDA diagonal | — |

<a id="g09"></a>
### T09 · Regresión logística

| Término | Definición | Ojo |
|---|---|---|
| Regresión logística binaria | Bernoulli condicional: $p(y=1\mid x) = \sigma(w^tx + b)$ | — |
| Sigmoide $\sigma(a)$ | $\dfrac{1}{1+e^{-a}}$; $\sigma(-a) = 1 - \sigma(a)$; $\sigma(0) = 0{,}5$ | Con etiquetas $\pm1$: $p(\tilde y\mid x) = \sigma(\tilde y\,a)$ |
| Logit / log-odds | $a = \log\dfrac{p(y=1\mid x)}{p(y=0\mid x)} = w^tx + b$ | MAP: clase 1 si $a > 0$ |
| Notación homogénea | Añadir un 1 a $x$ e incluir $b$ en $w$ | — |
| Frontera MAP (binaria) | $\{x : w^tx + b = 0\}$: recta, plano o hiperplano | $w$ es normal a ella y apunta a la clase 1 |
| Entropía cruzada binaria | $L = -\frac1N\sum_n\big[y_n\log\hat y_n + (1-y_n)\log(1-\hat y_n)\big]$ | Es la pérdida que se minimiza |
| Gradiente binario | $\frac1N\sum_n(\hat y_n - y_n)\,x_n = \frac1N X^t(\hat y - y)$ | "Error × entrada" |
| Descenso por gradiente | $w \leftarrow w - \eta\,\nabla L$ | Batch con $\frac1N$; unitario sin $\frac1N$ |
| Regresión logística (multiclase) | Categórica condicional: $\hat y = S(Wx + b)$ | — |
| Softmax $S(a)$ | $S(a)_c = \dfrac{e^{a_c}}{\sum_k e^{a_k}}$ | Conserva el orden: MAP $= \arg\max_c a_c$ |
| Gradiente multiclase | $\frac1N(\hat Y - Y)^tX$, con $Y$ en one-hot | Matriz $C \times D$ |
| Región multiclase | Intersección de $C-1$ semiespacios: poliedro $D$-dimensional (acotado o no) | — |
| Frontera entre varias clases | Intersección de las fronteras de cada una | — |
| Ingeniería de características | Construir $\phi(x)$ que linealice datos no linealmente separables | Ej.: XOR, $\phi(x) = x^2$ |
| Características cuadráticas | $\phi(x) = (1, x, \mathrm{triu}(xx^t))$ | Inviable con $D$ muy grande |
| LR frente a LDA | Ambos lineales con distinto ajuste; $(\gamma_c, \beta_c) \leftrightarrow$ fila $c$ de $W$ | — |
| QLR frente a QDA | Ambos cuadráticos con distinto ajuste | — |
| Ventajas | LR: más precisa y flexible en la entrada. DA: más fácil de ajustar y de añadir clases | — |

<a id="g10"></a>
### T10 · Regresión lineal

| Término | Definición | Ojo |
|---|---|---|
| Regresión lineal | Gaussiana 1d condicional: $p(y\mid x) = \mathcal N(y\mid w^tx + w_0, \sigma^2)$ | — |
| Simple / múltiple | Entrada unidimensional / multidimensional | — |
| Pérdida L2 | $\ell_2(y,\hat y) = (y - \hat y)^2$ | — |
| Regresor MMSE | $\hat y = \mu = E[y\mid x]$ | Su riesgo es $\sigma^2$ (la varianza) |
| Ajuste por pérdida cuadrática | $\min_w L_2(w) = \frac1N\sum_n(y_n - \hat y_n)^2$; gradiente $\frac2N X^t(\hat y - y)$ | Lleva un 2 |
| OLS | $\hat w = X^\dagger y$, con $X^\dagger = (X^tX)^{-1}X^t$ (pseudoinversa izquierda) | — |
| Ajuste por log-pérdida | En dos pasos: primero $\hat w$ (igual que OLS), luego $\hat\sigma^2 = L_2(\hat w)$ | "Básicamente lo mismo" |
| Regresión polinómica | Regresión lineal con $\phi(x) = (1, x, x^2, \dots, x^D)$ | Lineal en $w$ |
| Gráfica de residuos | $r_n = y_n - \hat y_n$ frente a $x_n$ | Solo regresión simple |
| Gráfica $\hat y$ frente a $y$ | Mejor cuanto más cerca de la diagonal | Simple y múltiple |
| RSS | $\sum_n(y_n - \hat y_n)^2$; menor = mejor | — |
| RMSE | $\sqrt{RSS/N}$ | Transformación monótona de la RSS |
| TSS | $\sum_n(y_n - \bar y)^2$: RSS del regresor que predice la media | — |
| $R^2$ | $1 - RSS/TSS$; $R^2 \le 1$; mejor cuanto más cerca de 1 | Mide la calidad frente a predecir la media |

<a id="g11"></a>
### T11 · Métodos basados en ejemplos

**KNN**

| Término | Definición | Ojo |
|---|---|---|
| Modelo no paramétrico | Mantiene los datos en inferencia | Muy costoso con muchos datos |
| KNN | Etiqueta más votada entre los $K$ prototipos más cercanos | Es de **clasificación**, no de regresión |
| Posterior KNN | $p(c\mid x) = \frac1K\sum_{n\in N_K(x,D)} I(y_n = c)$ | $I(y_n = c)$, no $\ne$ |
| NN | KNN con $K = 1$ | 2NN = NN |
| Empate a votos | Probabilidad significativa; se desempata con el 1NN entre las clases empatadas | — |
| Empate a distancias | Probabilidad insignificante; se desempata al azar | — |
| Distancia | Euclídea al cuadrado por defecto; en general, Mahalanobis | — |
| NN asintótico | Error $\le 2\times$ error de Bayes | — |
| KNN asintótico | Converge al error de Bayes si $K \to \infty$ y $K/N \to 0$ (p. ej. $K = \sqrt N$) | — |
| Maldición de la dimensionalidad | KNN empeora mucho con entradas de alta dimensión | — |
| Coste | Espacial y temporal muy alto; se reduce eliminando prototipos y con búsqueda eficiente (FAISS) | — |
| $K$ como regularizador | $K\downarrow$ sobreajuste; $K\uparrow$ subajuste (posteriors $\to$ priors) | — |
| Open set recognition | Fácil añadir clases nuevas (few-shot, biometría) | — |

**Kernels y KDE**

| Término | Definición | Ojo |
|---|---|---|
| Kernel | Función integrable, real y no negativa | — |
| Densidad kernel | Kernel normalizado ($\int K = 1$) y simétrico (par) | Por eso $\int x\,K(x)\,dx = 0$ |
| Kernel gaussiano | $K(x) = (2\pi)^{-1/2}\exp(-x^2/2)$; $K(0) = 0{,}3989$ | Signo menos en el exponente |
| Tophat · Epanechnikov · Linear · Cosine | $\tfrac12$ · $\tfrac34(1 - x^2)$ · $1 - \lvert x\rvert$ · $\tfrac\pi4\cos\tfrac{\pi x}{2}$ | Soporte $\lvert x\rvert \le 1$ |
| Gaussian · Exponential | Gaussiana · $\tfrac12 e^{-\lvert x\rvert}$ | Sin condición de soporte |
| Valores en 0 | 0,3989 · 0,5 · 0,75 · 0,5 · 1 · 0,7854 (Gauss, Tophat, Epan., Exp., Lin., Cos.) | — |
| Kernel centrado | $K(x - \mu)$; su media es $\mu$ | — |
| Ancho de banda $h$ | $K_h(x) = \frac1h K\!\left(\frac xh\right)$ | **$h\downarrow$ sobreajuste** (pico); $h\uparrow$ subajuste (plano) |
| RBF gaussiana | $K_h(x) = \prod_d K_h(x_d) = \dfrac{1}{h^D(2\pi)^{D/2}}\exp\!\left(-\dfrac{\lVert x\rVert^2}{2h^2}\right)$ | $K_h(x - \mu) = \mathcal N(\mu, h^2 I)$ |
| KDE / ventanas de Parzen-Rosenblatt | $p(x\mid h,D) = \frac1N\sum_n K_h(x - x_n)$: mezcla uniforme de un kernel por dato | Suma con $\frac1N$, no producto |
| KDE para clasificación | Una Parzen por clase + priors $N_c/N$ | — |
| KDE variable (balloon) | Crece un volumen $V(x)$ hasta tener $K$ datos: $p(x\mid c) = \dfrac{N_c(x)/N_c}{V(x)}$ | Su posterior es KNN: $N_c(x)/K$ |
| Regresión kernel / Nadaraya-Watson | KDE de la conjunta $\Rightarrow \hat y = \sum_n w_n(x)\,y_n$ | — |
| Pesos NW | $w_n(x) = \dfrac{K_h(x - x_n)}{\sum_{n'} K_h(x - x_{n'})}$ | Sin $\frac1N$ |
| Pesos NW gaussianos | Softmax de logits $a_n = -\dfrac{1}{2h^2}\lVert x - x_n\rVert^2$ | Signo menos y $h^2$ |

<a id="g12"></a>
### T12 · Árboles, bagging, bosques y boosting

**Árboles**

| Término | Definición | Ojo |
|---|---|---|
| CART | Árbol que particiona el espacio recursivamente hasta las hojas | Clasificación y regresión |
| Nodo interno | Split paralelo a un eje: si $x_{d_i} \le t_i$, rama izquierda | — |
| Hoja | Región + predicción (valor $w_j$ o distribución de clases) | — |
| Árbol de regresión | $f(x) = \sum_j w_j\,I(x \in R_j)$ | — |
| Aprendizaje voraz | CART, C4.5, ID3: añaden un nodo por iteración | Solución aproximada |
| Splits candidatos | Real: un $t$ por valor observado ($\le$ / $>$). Categórica: $= t$ / $\ne t$ | — |
| Coste / impureza | Regresión: MSE respecto a la media del nodo. Clasificación: Gini o entropía | Independiente del tamaño del nodo |
| Índice Gini | $G_i = 1 - \sum_c \hat\pi_{ic}^2$: error esperado si etiquetas al azar según las proporciones | Máx. 0,5 con 2 clases |
| Entropía | $H_i = -\sum_c \hat\pi_{ic}\log\hat\pi_{ic}$ | Máx. 1 bit con 2 clases ($\log_2$) |
| Mejor split | Minimiza la suma de costes de los hijos ponderados por su tamaño | — |
| Regularización | Parar con pocos datos en el nodo o profundidad máxima | — |
| Inestabilidad | Pequeños cambios en los datos $\to$ árboles muy distintos | Alta varianza |

**Ensambles**

| Término | Definición | Ojo |
|---|---|---|
| Ensamble | Promedio de $M$ modelos base: $f = \frac1M\sum_m f_m$ | Reduce la varianza |
| Método comité | Voto mayoritario en clasificación | — |
| Bagging | Bootstrap (muestreo con reemplazamiento) + ensamble | Sesgo similar, menos varianza |
| Fracción vista | Cada modelo ve $\approx$ **63 %** de los datos | No 50 % |
| Out-of-bag | $\approx$ 37 % ($e^{-1}$) no seleccionado | Sirve de test |
| Random forest | Bagging + subconjunto aleatorio de variables en cada split | Más preciso que bagging; paralelizable |
| Modelo aditivo | $f(x) = \sum_m \beta_m F_m(x;\theta_m)$ | — |
| Boosting (FSAM) | $f_m = f_{m-1} + \beta_m F_m$, cada aprendiz ajustado según el anterior | Reduce el **sesgo** |
| Weak / strong learner | Modelo base añadido / ensamble acumulado | — |
| Boosting mínimos cuadrados | $\beta = 1$ y $F_m$ ajustado a los residuos $y - f_{m-1}(x)$ | — |
| AdaBoost | Pérdida exponencial, etiquetas $\pm1$, datos ponderados | $f(x) = \mathrm{sgn}\big(\sum_m \beta_m F_m(x)\big)$ |
| Gradient boosting | Boosting como descenso por gradiente en un espacio de funciones | Visión actual del boosting |
| XGBoost | Implementación popular de gradient boosting | — |
| Importancia de una variable | Suma de ganancias (reducción de coste) en los nodos que la usan, promediada entre árboles | — |
| Dependencia parcial | Predicción media en función de una o dos variables | — |

<a id="g13"></a>
### T13 · Aprendizaje no supervisado

| Término | Definición | Ojo |
|---|---|---|
| Aprendizaje no supervisado | Dataset de entradas sin salidas $D = \{x_n\}$ | — |
| GMM | $p(x) = \sum_c \pi_c\,\mathcal N(x\mid\mu_c, \Sigma_c)$ | — |
| Evidencia (log-verosimilitud marginal) | $LL = \sum_n \log\sum_c \pi_c\,\mathcal N(x_n\mid\mu_c,\Sigma_c)$ | Se **maximiza** |
| Algoritmo EM | Algoritmo MM para modelos con datos perdidos o variables ocultas | Técnica estándar de MLE para mezclas |
| Responsabilidad $r_{nc}$ | $p(c\mid x_n,\theta^{(i)}) = \dfrac{\pi_c\,\mathcal N(x_n\mid\mu_c,\Sigma_c)}{\sum_k \pi_k\,\mathcal N(x_n\mid\mu_k,\Sigma_k)} = \dfrac{\pi_c\,\mathcal N(x_n\mid\mu_c,\Sigma_c)}{p(x_n\mid\theta^{(i)})}$ | Las tres expresiones son válidas |
| Paso E | Calcula las responsabilidades (función sustituta $Q$) | — |
| Paso M | $\pi_c = r_c/N$; $\mu_c = \sum_n r_{nc}x_n/r_c$; $\Sigma_c = \sum_n r_{nc}(x_n - \mu_c)(x_n - \mu_c)^t/r_c$ | $\Sigma_c$ con la **nueva** $\mu_c$ |
| Inicialización | Barajar datos; $\pi_c = 1/C$; $\mu_c = x_c$; $\Sigma_c = \hat\Sigma/C$ | — |
| Caso $\Sigma$ común | $\Sigma = \sum_c \pi_c\Sigma_c$ | — |
| C-medias | EM con $\Sigma_c = \sigma^2 I$ y paso E reducido a un clasificador de distancia mínima | Caso particular de EM |
| $\sigma^2 \to 0^+$ | Responsabilidades duras: todo al centro más cercano (empates según priors) | — |
| $\sigma^2 \to \infty$ | Responsabilidades $\to$ priors | — |
| MM (majorize-minimize) | Optimizar una sustituta $Q(\theta,\theta^{(i)}) \le LL(\theta)$ que toca $LL$ en $\theta^{(i)}$ | Garantiza que $LL$ no baja |
| ELBO | Cota inferior de $LL$ (por Jensen); igual a $LL$ si $q$ = posterior | El hueco es una KL |

---

<a id="teoria"></a>
## 3. Teoría con la técnica Feynman

<a id="t07"></a>
### T07 · Teoría de la decisión

**Idea clave:** decidir bien es elegir la acción con menor pérdida esperada. Se usa lo que sabíamos antes (prior) corregido con lo que hemos visto (evidencia).

#### Explicación para un chaval de 12 años

Eres detective y alguien se ha comido la tarta.

- **Prior**: antes de investigar ya sospechas de tu hermano pequeño, porque lo hace a menudo.
- **Evidencia**: encuentras migas en su mesa.
- **Verosimilitud**: "si hubiera sido él, ¿qué probable es ver estas migas?".
- **Posterior**: tu sospecha actualizada al mezclar lo anterior. Eso es la regla de Bayes.

El detective no solo adivina: tiene que **actuar**. Acusar a un inocente cuesta caro y dejar libre al culpable también (**función de pérdida**). La decisión sabia es la que, de media, hace perder menos (**riesgo mínimo**).

Si todos los errores cuestan igual, basta con acusar al más probable: eso es el clasificador **MAP**.

Para evaluarte apuntas aciertos y confusiones en una tabla (**matriz de confusión**). Si tienes un "medidor de sospecha" de 0 a 1, puedes mover la raya a partir de la cual acusas:

- Raya baja: pillas a casi todos los culpables, pero acusas a muchos inocentes.
- Raya alta: al revés.
- Las **curvas ROC y PR** dibujan lo que pasa en cada posición de la raya.

#### Fórmulas imprescindibles

$$p(h\mid x)=\frac{p(h)\,p(x\mid h)}{p(x)},\qquad p(x)=\sum_h p(h)\,p(x\mid h)$$

$$R(a\mid x)=\sum_h \ell(h,a)\,p(h\mid x),\qquad \pi^*(x)=\arg\min_a R(a\mid x)$$

$$\text{Pérdida 0-1: } R(\hat y\mid x)=1-p(\hat y\mid x)\ \Rightarrow\ \pi(x)=\arg\max_{\hat y}p(\hat y\mid x)$$

$$F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R},\qquad F_1=\frac{2PR}{P+R}$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| ¿$p(x\mid h)$ y $p(h\mid x)$ son lo mismo? | No. "Si es un perro, ¿tendrá cuatro patas?" (casi seguro) no es "si tiene cuatro patas, ¿es un perro?" (hay gatos, vacas…). Bayes convierte una en otra. |
| ¿Para qué sirve $p(x)$? | Normaliza para que los posteriores sumen 1. Es igual para todas las hipótesis, así que no cambia el ganador del argmax. |
| ¿Por qué minimizar riesgo y no elegir lo más probable? | Porque los errores no siempre cuestan igual. Un detector de incendios prefiere falsas alarmas a no sonar. Si FN cuesta $c$ veces FP: positivo si $p_1 \ge \frac{1}{1+c}$. |
| ¿Qué es la opción de rechazo? | Decir "no lo sé" a un precio fijo $\lambda_r$. Como el concursante que se planta: solo responde si $p^* > 1 - \lambda_r/\lambda_e$. |
| ¿Filas o columnas? | Filas = verdad; columnas = lo que dijo el clasificador. Por filas sale recall/TPR/FPR; por columnas, precisión/PPV. |
| ¿Cómo entiendo el umbral $\tau$? | Como "generosidad": positivo si $p(y=1\mid x) \ge 1 - \tau$. $\tau = 0$ solo acusa a los seguros; $\tau = 1$ acusa a todos. En la práctica se ordena por puntuación y se baja la raya dato a dato. |
| ¿ROC o PR? | ROC compara $TPR$ con $FPR$ y no le importa cuántos negativos hay. PR se centra en los positivos: es la buena cuando son raros (aguja en un pajar). |
| ¿Qué resumen AUC, EER y AP? | AUC: área bajo ROC ($\uparrow$ mejor; 0,5 = azar). EER: punto con $FPR = FNR$ ($\downarrow$ mejor). AP: área bajo la PR interpolada ($\uparrow$ mejor). |
| ¿Por qué $F_1$ y no la media de $P$ y $R$? | La media aritmética da 0,5 con $P = 0$. $F_1$ es como una cadena: tan fuerte como su eslabón más débil. $\beta > 1$ premia el recall; $\beta < 1$, la precisión. |
| Macro, weighted, micro | Macro: todas las clases cuentan igual. Weighted: pesa cada clase por su frecuencia. Micro: junta todos los aciertos; sale $TP/M$ (exactitud). |
| ¿Qué pinta la regresión aquí? | Si la respuesta es un número y la pérdida es el error al cuadrado, la mejor apuesta es la **media** del posterior. Como adivinar una nota diciendo la media esperada. |

<a id="t08"></a>
### T08 · Análisis discriminante

**Idea clave:** un clasificador generativo aprende cómo es cada clase, $p(x\mid c)$, y cuánto abunda, $p(c)$. Luego elige la clase con mayor $p(c)\,p(x\mid c)$.

#### Explicación para un chaval de 12 años

Tienes que distinguir manzanas de naranjas por peso y diámetro. El clasificador hace un **retrato robot** de cada fruta:

- El retrato es una campana (**gaussiana**) con un centro (**media**) y una anchura y forma (**covarianza**).
- Además apunta cuántas frutas de cada tipo hay en la caja (**prior**).

Cuando llega una fruta nueva, pregunta a cada retrato "¿te pega esta fruta?" y multiplica por lo abundante que es esa clase. Gana la que saca más puntos.

Versiones según lo detallado del retrato:

| Versión | Qué supone | Frontera |
|---|---|---|
| QDA | Cada clase tiene su propia campana con su forma | Curva (cuadrática) |
| LDA | Todas comparten la forma; solo cambia el centro | Recta |
| NCM | LDA con clases igual de abundantes: "¿a qué centro estás más cerca?" | Recta (mediatriz si $\Sigma = I$) |
| Naive Bayes | Un retrato por rasgo, como si los rasgos no tuvieran relación | Depende de las distribuciones |

#### Fórmulas imprescindibles

$$\hat\pi_c=\frac{N_c}{N},\qquad \hat\mu_c=\frac{1}{N_c}\sum_{n:y_n=c}x_n,\qquad \hat\Sigma_c=\frac{1}{N_c}\sum_{n:y_n=c}(x_n-\hat\mu_c)(x_n-\hat\mu_c)^t$$

$$\text{QDA: } g_c(x)=\log\pi_c-\tfrac12\log|\Sigma_c|-\tfrac12(x-\mu_c)^t\Sigma_c^{-1}(x-\mu_c)$$

$$\text{LDA: } g_c(x)=\beta_c^tx+\gamma_c,\quad \beta_c=\Sigma^{-1}\mu_c,\quad \gamma_c=\log\pi_c-\tfrac12\mu_c^t\Sigma^{-1}\mu_c$$

$$\tilde\Sigma_c=(1-\lambda)\hat\Sigma_c+\lambda\hat\Sigma,\qquad \hat\Sigma=\sum_c\hat\pi_c\hat\Sigma_c$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| ¿Generativo frente a discriminativo? | El generativo sabe *dibujar* cada clase (primero elige $y$, luego genera $x$). El discriminativo (T09) solo aprende a *trazar la raya* entre clases. |
| ¿Por qué puedo quitar $p(x)$? | Es el mismo número para todas las clases: como restar los mismos puntos a todos los concursantes. Basta $g_c = \log p(c) + \log p(x\mid c)$. |
| ¿Por qué logaritmos? | El log no cambia quién gana y convierte productos en sumas. También puedes quitar constantes comunes, como $\frac D2\log 2\pi$. |
| ¿Qué es la máxima verosimilitud? | Elegir los parámetros que hacen más probables los datos vistos. Sale lo intuitivo: frecuencias, medias y dispersiones de cada clase (dividiendo por $N_c$). |
| ¿Cada clase se aprende por separado? | Sí: el retrato de las manzanas se hace mirando solo manzanas. |
| ¿Por qué la frontera de LDA es recta? | Con la misma $\Sigma$, el término $x^t\Sigma^{-1}x$ es igual en todas las clases y se cancela al comparar. En QDA no se cancela. |
| ¿Para qué regularizar? | Con pocos datos, la covarianza de una clase es poco fiable. Mezclarla con la global es pedir opinión al grupo; $\lambda$ dice cuánto caso le haces. |
| ¿Cómo saco la frontera? | Igualando $g_1(x) = g_2(x)$. En LDA sale $(\beta_1 - \beta_2)^tx + (\gamma_1 - \gamma_2) = 0$. |
| ¿Naive Bayes gaussiano y QDA? | NB gaussiano es QDA con covarianzas diagonales: ignora la correlación entre rasgos. Con rasgos discretos, el MLE es contar frecuencias. |

<a id="t09"></a>
### T09 · Regresión logística

**Idea clave:** calcula una puntuación lineal $a = w^tx + b$ y la convierte en probabilidad con la sigmoide (2 clases) o la softmax (C clases). Se entrena por descenso por gradiente minimizando la entropía cruzada.

#### Explicación para un chaval de 12 años

Imagina el jurado de un concurso de talentos:

- Cada rasgo (voz, baile, simpatía) tiene un **peso**. Sumas puntos × pesos y sale una nota que puede ser cualquier número: la **puntuación lineal** $a$.
- Un **aplastador** (la sigmoide) convierte esa nota en "probabilidad de pasar": muy negativa → casi 0 %; muy positiva → casi 100 %; 0 → 50 %.
- Si hay varios premios, la **softmax** reparte el 100 % entre ellos: quien tiene más nota se lleva el trozo mayor.

¿Cómo aprende el jurado? Probando y corrigiendo. Si dijo "80 % de que pasa" y no pasó, ajusta un poquito los pesos en la dirección que reduce el error (**descenso por gradiente**). Ese "poquito" es la **tasa de aprendizaje** $\eta$.

#### Fórmulas imprescindibles

$$\hat y=\sigma(a)=\frac{1}{1+e^{-a}},\quad a=w^tx+b,\qquad \text{MAP: clase 1 si } a>0$$

$$L(w)=-\frac1N\sum_n\big[y_n\log\hat y_n+(1-y_n)\log(1-\hat y_n)\big],\qquad w\leftarrow w-\eta\,\frac1N X^t(\hat y-y)$$

$$\hat y=S(a),\quad S(a)_c=\frac{e^{a_c}}{\sum_k e^{a_k}},\quad a=Wx+b,\qquad W\leftarrow W-\eta\,\frac1N(\hat Y-Y)^tX$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| Si es "regresión", ¿por qué clasifica? | Predice un número continuo (la probabilidad) y luego se usa para clasificar. Nombre histórico. |
| ¿Clase 1 si $a > 0$ o si $\hat y > 0{,}5$? | Es lo mismo: $\sigma(0) = 0{,}5$ y la sigmoide siempre sube. $a$ es el log-odds. |
| ¿Qué relación tiene $w$ con la frontera? | La frontera es $w^tx + b = 0$; $w$ es perpendicular y apunta a la clase 1, como una flecha que señala "por aquí están los sí". |
| ¿Qué es la notación homogénea? | Añadir un 1 delante de $x$ y meter $b$ como primer peso. Mira en cada ejercicio si $b$ ya está dentro de $w$. |
| ¿Por qué entropía cruzada? | Castiga mucho estar muy seguro y equivocado, como un profesor que penaliza más la barbaridad dicha con aplomo. Y da un gradiente limpio. |
| ¿Cómo leo $(\hat y - y)\,x$? | "Error × entrada": si predijiste de más, resta; si de menos, suma; si acertaste, apenas cambia. |
| ¿Batch o unitario? | Batch: promedias el gradiente de todos (con $\frac1N$) y das un paso. Unitario: un paso por dato, sin $\frac1N$. |
| ¿Hacen falta exponenciales para clasificar? | No: la softmax conserva el orden, gana el mayor $a_c$. Solo hacen falta para probabilidades y gradientes. |
| ¿Y si una recta no separa (XOR)? | Transforma $x$ con $\phi(x)$ ($x^2$, $x_1x_2$…) para que en el nuevo espacio sí se separen. Como desarrugar un papel para cortarlo recto. |
| ¿Logística o discriminante? | Se pueden traducir una en otra. LR suele ser más precisa y admite cualquier $\phi$; DA se ajusta sin iterar y añade clases sin reentrenar. |

<a id="t10"></a>
### T10 · Regresión lineal

**Idea clave:** predice un número con $\hat y = w^tx + w_0$ eligiendo los pesos que minimizan el error cuadrático medio. Tiene solución cerrada: $w = (X^tX)^{-1}X^ty$.

#### Explicación para un chaval de 12 años

Apuntas horas de estudio y notas de tus compañeros: salen puntos desperdigados. Quieres **la recta que mejor pasa entre ellos** para decir "si estudias 5 horas, sacarás un 7".

- Une cada punto a la recta con una gomita vertical: mide el fallo en ese punto (**residuo**).
- Cuanto más estirada, más tira (cuenta al cuadrado).
- La mejor recta es la que hace que las gomitas tiren lo menos posible en total: **mínimos cuadrados**.

La recta no promete la nota exacta: alrededor hay una nube de notas posibles (la **varianza** $\sigma^2$). Si los puntos dibujan una curva, usa $x, x^2, x^3\dots$ como datos nuevos: **regresión polinómica**.

#### Fórmulas imprescindibles

$$\hat y=w^tx\ \ \big(x=(1,x_1,\dots)\big),\qquad L_2(w)=\frac1N\sum_n(y_n-\hat y_n)^2$$

$$\hat w=(X^tX)^{-1}X^ty,\qquad \hat\sigma^2=L_2(\hat w),\qquad w\leftarrow w-\eta\,\frac2N X^t(\hat y-y)$$

$$RSS=\sum_n(y_n-\hat y_n)^2,\quad RMSE=\sqrt{RSS/N},\quad TSS=\sum_n(y_n-\bar y)^2,\quad R^2=1-\frac{RSS}{TSS}$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| ¿Qué tiene que ver con el T07? | $y$ es una campana centrada en $w^tx$. Con pérdida cuadrática, lo óptimo es predecir la media de la campana. El riesgo que queda, $\sigma^2$, es inevitable. |
| ¿Por qué al cuadrado? | Da una fórmula cerrada y equivale a suponer ruido gaussiano: mínimos cuadrados y máxima verosimilitud dan el mismo $w$. |
| ¿Para qué la columna de unos? | Multiplica a $w_0$ (la ordenada en el origen). Sin ella la recta pasaría obligatoriamente por $(0,0)$. Error típico: olvidarla. |
| ¿Cómo invierto $X^tX$ a mano? | $2\times2$: $\frac{1}{ad-bc}\,[d,\ -b;\ -c,\ a]$. $3\times3$: adjunta traspuesta / determinante. $X^tX$ es simétrica: úsalo para comprobar. |
| ¿Y si $X^tX$ no es invertible? | Pasa con columnas repetidas o pocos datos. Se usa descenso por gradiente o la pseudoinversa general. |
| ¿Por qué el gradiente lleva un 2? | Por derivar el cuadrado: $\frac{d}{de}e^2 = 2e$. En logística el 2 no aparece. |
| ¿La polinómica es no lineal? | No lineal en $x$, pero lineal en $w$. Solo cambian las columnas de $X$; el cálculo es el mismo. |
| ¿Cómo sé si es bueno? | $R^2$ compara con el "regresor tonto" que dice siempre la media: 1 perfecto, 0 igual, negativo peor. Residuos sin patrón y puntos $(y, \hat y)$ sobre la diagonal. |

<a id="t11"></a>
### T11 · Métodos basados en ejemplos (KNN, KDE, Nadaraya-Watson)

**Idea clave:** no resumen los datos en parámetros; los guardan todos y miran los parecidos a cada consulta.

| Método | Qué hace |
|---|---|
| KNN | Vota entre los $K$ vecinos más cercanos |
| KDE (Parzen) | Estima una densidad poniendo una campanita en cada dato |
| Nadaraya-Watson | Promedia las salidas pesándolas por cercanía |

#### Explicación para un chaval de 12 años

Llegas a un colegio nuevo y quieres saber si un chico juega al fútbol o al baloncesto. Miras a sus **5 amigos más cercanos**: si 3 juegan al baloncesto, apuestas por baloncesto. Eso es **KNN**: "dime con quién andas y te diré quién eres".

Ahora cada alumno deja caer en el patio un montón de arena con forma de campana:

- Donde hay muchos alumnos juntos sale una montaña; donde hay pocos, llanura. La altura es la **densidad estimada** (KDE).
- El **ancho de banda** $h$ es lo ancho de cada montón: estrechos → picos sueltos; anchos → todo plano.

Para predecir un número (la altura de alguien según su edad), **Nadaraya-Watson** promedia las alturas de los demás, dando más voto a los de edad parecida.

#### Fórmulas imprescindibles

$$p(y=c\mid x,D)=\frac1K\sum_{n\in N_K(x,D)}I(y_n=c)$$

$$K(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2},\qquad K_h(x)=\frac1h K\!\left(\frac xh\right),\qquad p(x\mid h,D)=\frac1N\sum_n K_h(x-x_n)$$

$$\hat y(x)=\sum_n w_n(x)\,y_n,\qquad w_n(x)=\frac{K_h(x-x_n)}{\sum_{n'}K_h(x-x_{n'})}=S\Big(\Big\{-\tfrac{1}{2h^2}\lVert x-x_{n'}\rVert^2\Big\}_{n'}\Big)_n$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| ¿Qué es "no paramétrico"? | El modelo *son* los datos. La logística guarda pesos y tira la libreta; KNN se queda con la libreta entera. Coste: memoria y tiempo. |
| ¿Distancia con o sin cuadrado? | Para ordenar vecinos da igual y el cuadrado ahorra la raíz. Por defecto, euclídea al cuadrado. |
| ¿Y si hay empate de votos? | Gana el vecino más cercano entre las clases empatadas. Por eso 2NN = NN. Empate de distancias: al azar. |
| ¿Qué $K$ elijo? | $K$ pequeño: te fías de un amigo que puede ser rarito (sobreajuste). $K$ enorme: gana siempre la clase más numerosa (subajuste). $K = \sqrt N$ es una referencia teórica. |
| ¿Maldición de la dimensionalidad? | En muchas dimensiones todos los puntos están casi igual de lejos: "el más cercano" deja de ser "el más parecido". |
| ¿Por qué $\frac1h K(x/h)$? | Al ensanchar la campana $h$ veces hay que bajarla $h$ veces para que el área siga siendo 1. $K_{0,5}(0) = 0{,}7979$; $K_2(0) = 0{,}1995$. |
| ¿Qué tiene que ver KDE con KNN? | Si fijas $K$ y haces crecer una burbuja hasta tocar $K$ datos (KDE variable), el posterior que sale es la votación de KNN. |
| ¿NW y softmax? | Con kernel gaussiano, los pesos son una softmax de $-\lVert x - x_n\rVert^2/(2h^2)$. Las constantes se cancelan al normalizar. |

<a id="t12"></a>
### T12 · Árboles, bagging, bosques y boosting

**Idea clave:** un árbol pregunta "¿$x_d \le t$?" y elige en cada nodo la pregunta que más reduce la impureza. Los ensambles corrigen su inestabilidad.

| Técnica | Cómo combina árboles | Qué reduce |
|---|---|---|
| Bagging | Árboles independientes con datos remuestreados; votan | Varianza |
| Random forest | Bagging + variables al azar en cada pregunta | Varianza (más) |
| Boosting | Árboles en fila; cada uno corrige al anterior | Sesgo |

#### Explicación para un chaval de 12 años

Es el **"¿Quién es quién?"**. Haces preguntas de sí o no: "¿lleva gafas?", "¿mide más de 1,80?".

- Un buen jugador elige primero la pregunta que **mejor parte en dos** a los candidatos, dejando grupos lo más "puros" posible.
- Cuando un grupo ya es puro, o quedan pocos, paras: es una **hoja**, y su respuesta es la clase mayoritaria.

Problema: si cambias un poco las cartas, la estrategia puede salir totalmente distinta (**inestable**). Soluciones:

- **Bagging**: 100 amigos juegan, cada uno con un mazo sacado al azar con repetición, y votan.
- **Bosque aleatorio**: además, en cada pregunta cada amigo solo puede usar unos pocos rasgos al azar, para que no copien todos la misma.
- **Boosting**: juegan en fila; cada uno se concentra en las cartas que falló el anterior. Al final se suman sus opiniones con pesos.

#### Fórmulas imprescindibles

$$G_i=1-\sum_c\hat\pi_{ic}^2,\qquad H_i=-\sum_c\hat\pi_{ic}\log\hat\pi_{ic},\qquad c_{\text{reg}}(D_i)=\frac{1}{|D_i|}\sum_{n\in D_i}(y_n-\bar y)^2$$

$$(j_i,t_i)=\arg\min_{j,t}\ \frac{|D_i^L|}{|D_i|}c(D_i^L)+\frac{|D_i^R|}{|D_i|}c(D_i^R)$$

$$\lim_{N\to\infty}(1-1/N)^N=e^{-1}\approx0.37\quad(\text{out-of-bag})$$

$$\text{err}_m=\frac{\sum_i w_{im}I(\tilde y_i\ne F_m(x_i))}{\sum_i w_{im}},\quad \beta_m=\tfrac12\log\frac{1-\text{err}_m}{\text{err}_m},\quad f(x)=\text{sgn}\Big(\sum_m\beta_mF_m(x)\Big)$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| ¿Qué es la impureza? | Lo mezclado que está un grupo. Saco solo de canicas rojas: 0. Mitad rojas, mitad azules: máxima (Gini 0,5; entropía 1 bit). |
| ¿Gini o entropía? | Miden lo mismo con distinta regla y casi siempre eligen el mismo corte. Usa la que pida el enunciado y fíjate en la base del log. |
| ¿Por qué ponderar por tamaño? | Dejar 1 dato puro y 99 mezclados no es buen corte: la impureza de los hijos se promedia según cuántos datos se lleva cada uno. |
| ¿Qué umbrales pruebo? | Cada valor observado de la característica en el nodo ($\le$ a la izquierda). Con categóricas, "$=$ categoría" frente a "$\ne$". Es voraz: no mira el futuro. |
| ¿Cuándo paro? | Hojas puras, pocos datos o profundidad máxima. Si no, el árbol memoriza (sobreajuste). Fronteras siempre en escalones paralelos a los ejes. |
| ¿Qué es bootstrap? | Sacar $N$ datos con reemplazamiento de $N$. Unos se repiten y $\approx$ 37 % no sale nunca: sirven de test gratis (out-of-bag). |
| ¿Bagging o boosting? | Bagging/bosques: árboles independientes, $\downarrow$ varianza. Boosting: dependientes, $\downarrow$ sesgo. Con L2, cada árbol se ajusta a los residuos. |
| ¿Cómo funciona $\beta$ en AdaBoost? | Si un aprendiz falla poco, $\beta$ es grande y su voto pesa mucho; si acierta como una moneda ($\text{err} = 0{,}5$), $\beta = 0$. Los fallados multiplican su peso por $e^{2\beta}$. |
| ¿Y la interpretación? | Importancia: cuánto redujo la impureza una variable donde se usó. Dependencia parcial: cómo cambia la predicción al mover 1 o 2 variables. |

<a id="t13"></a>
### T13 · Aprendizaje no supervisado (mezclas gaussianas, EM y C-medias)

**Idea clave:** sin etiquetas, se supone que los datos vienen de una mezcla de $C$ gaussianas y se ajusta con EM. C-medias es el caso límite en que cada dato va entero a su centro más cercano.

#### Explicación para un chaval de 12 años

En el recreo hay niños por todo el patio, pero nadie te dice de qué clase es cada uno. Sospechas que hay **dos corrillos**:

1. Pones dos banderitas al azar (**centros** de partida).
2. **Paso E ("¿de quién eres?")**: cada niño mira qué bandera tiene cerca.
   - Versión suave (EM): "soy 90 % de A y 10 % de B" → **responsabilidad**.
   - Versión dura (C-medias): "soy 100 % de A".
3. **Paso M ("mueve la bandera")**: cada bandera se coloca en el centro de sus niños, contando a cada uno según su porcentaje. Se recalculan también el tamaño ($\pi$) y lo esparcido (covarianza) de cada corrillo.
4. Repites E y M hasta que las banderas casi no se muevan.

Cada vuelta explica los datos igual o mejor que la anterior: nunca empeora.

#### Fórmulas imprescindibles

$$p(x\mid\theta)=\sum_c\pi_c\,\mathcal N(x\mid\mu_c,\Sigma_c),\qquad LL(\theta)=\sum_n\log\sum_c\pi_c\,\mathcal N(x_n\mid\mu_c,\Sigma_c)$$

$$\text{E: } r_{nc}=\frac{\pi_c\,\mathcal N(x_n\mid\mu_c,\Sigma_c)}{\sum_{k}\pi_{k}\,\mathcal N(x_n\mid\mu_{k},\Sigma_{k})}$$

$$\text{M: } r_c=\sum_n r_{nc},\quad \pi_c=\frac{r_c}{N},\quad \mu_c=\frac{1}{r_c}\sum_n r_{nc}x_n,\quad \Sigma_c=\frac{1}{r_c}\sum_n r_{nc}(x_n-\mu_c)(x_n-\mu_c)^t$$

$$\mathcal N(x\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)$$

#### Dudas típicas

| Duda | Aclaración |
|---|---|
| ¿En qué se diferencia del T08? | Mismo modelo, pero en el T08 sabíamos la clase de cada dato. Aquí está oculta: EM usa probabilidades $r_{nc}$ en vez de etiquetas, y el paso M es el MLE del T08 con pesos. |
| ¿La responsabilidad es un posterior? | Sí: $r_{nc} = p(c\mid x_n)$ con los parámetros actuales. Es Bayes del T07 dato a dato; cada fila suma 1. |
| ¿Qué media uso para la covarianza? | La **nueva**, recién calculada en ese paso M. Error típico: usar la antigua. |
| ¿Por qué EM siempre mejora? | Sube una sustituta (ELBO) que toca la log-verosimilitud en el punto actual y queda por debajo en el resto. Puede atascarse en óptimos locales. |
| ¿C-medias es otro algoritmo? | Es EM con covarianzas $\sigma^2I$ y $\sigma^2 \to 0$: cada dato va al 100 % a su centro más cercano y cada centro pasa a ser la media de su grupo. Con $\sigma^2 \to \infty$, $r \to$ priors. |
| ¿Cómo evalúo una gaussiana a mano? | $\mathcal N(0,1)$ en 0, 1, 2, 3 vale 0,3989; 0,2420; 0,0540; 0,0044. Si todas las $\sigma^2$ son iguales, en el paso E basta comparar $\pi_c\exp(-d^2/2\sigma^2)$. |

---

<a id="recetas"></a>
## 4. Recetas paso a paso para los ejercicios

Cada receta sirve para cualquier problema del mismo tipo. Los ejemplos son los de los boletines, resueltos y comprobados con Python. Los del T08 al T13 no tienen solución oficial en el material: son cálculos propios.

<a id="r-test"></a>
### Estrategia para preguntas tipo test

Formato habitual: "indica la opción incorrecta o la última si las tres primeras son correctas".

1. Busca **el detalle cambiado** en cada opción: signo, $=$ por $\ne$, mayor por menor, un porcentaje, $\lvert x\rvert \le 1$ por $\lvert x\rvert \le 2$.
2. Si las tres te parecen bien, marca la cuarta: en estos boletines es correcta con mucha frecuencia.
3. Si hay números, calcula siempre.

Trampas que aparecen en los boletines:

| Afirmación trampa | Lo correcto |
|---|---|
| "AUC y EER son mejores cuanto mayores" | EER es mejor cuanto **menor** |
| "Cada modelo de bagging ve un 50 % de los datos" | $\approx$ **63 %** |
| "KNN es una técnica de regresión" | Es de **clasificación** |
| Posterior KNN con $I(y_n \ne c)$ | $I(y_n = c)$ |
| Kernel gaussiano con $\exp(x^2/2)$ | $\exp(-x^2/2)$ |
| Epanechnikov y Cosine con $\lvert x\rvert \le 2$ | $\lvert x\rvert \le 1$ |
| "$h\downarrow$ subajuste; $h\uparrow$ sobreajuste" | Al revés: $h\downarrow$ **sobreajuste** |
| NCM con $\gamma_c = +\tfrac12\mu_c^t\Sigma^{-1}\mu_c$ | $\gamma_c = -\tfrac12\mu_c^t\Sigma^{-1}\mu_c$ |
| Parzen con producto o sin $\frac1N$ | $\frac1N\sum_n K_h(x - x_n)$ |
| Pesos NW con $\frac1N$ delante | Solo $K_h(x - x_n)/\sum_{n'} K_h(x - x_{n'})$ |
| Logits NW positivos o sin cuadrado | $-\lVert x - x_n\rVert^2/(2h^2)$ |
| ROC = $FPR$ en función de $TPR$ | $TPR$ en función de $FPR$ |
| PR interpolada con min o con $\tilde\tau < \tau$ | $\max$ con $\tilde\tau \ge \tau$ |

<a id="r07"></a>
### T07 · Decisión

**A) Posterior con Bayes (cajas, frutas, pruebas)**

1. Lista las hipótesis $h$ y sus priors ("caja al azar" = $1/n$ cada una).
2. Escribe las verosimilitudes $p(x\mid h)$ = favorables / total de cada caja.
3. Evidencia: $p(x) = \sum_h p(h)\,p(x\mid h)$.
4. Posterior: $p(h\mid x) = p(h)\,p(x\mid h)/p(x)$. Comprueba que suman 1.

- Ejemplo: 2 cajas de 40; caja 1 con 28 Gala, caja 2 con 20 Gala.
  - $p(G) = \frac12\cdot\frac{28}{40} + \frac12\cdot\frac{20}{40} = \frac{24}{40}$.
  - $p(1\mid G) = \dfrac{\frac12\cdot 28/40}{24/40} = \frac{7}{12} =$ **0,5833**.

**B) Decisión de mínimo riesgo**

1. Calcula los posteriores (receta A).
2. Para cada acción: $R(a\mid x) = \sum_h \ell(h,a)\,p(h\mid x)$. Con pérdida 0-1, $R(a\mid x) = 1 - p(a\mid x)$.
3. Elige la acción de menor riesgo.

- Ejemplo (Gala): $R(1\mid G) = 0{,}4167$ y $R(2\mid G) = 0{,}5833$ → **caja 1**.

**C) Umbral sensible al coste y rechazo**

1. Coste: con acierto gratis y $\ell_{10} = c\,\ell_{01}$ (FN $c$ veces más caro), positivo si $p_1 \ge \frac{1}{1+c}$.
   - Ejemplo: $c = 2$ → $p_1 \ge$ **1/3**.
2. Rechazo: $R(0\mid x) = \lambda_r$ y $R(\hat y\mid x) = \lambda_e\,(1 - p(\hat y\mid x))$.
   - Clasifica en $y^*$ si $p^* > 1 - \lambda_r/\lambda_e$; si no, rechaza.

**D) Matriz de confusión y tasas de error**

1. Tabla $C\times C$: filas = real, columnas = predicha. Suma 1 por cada par.
2. Error global = (total − diagonal) / total.
3. Error de la clase real $c$ = (fila $c$ − diagonal) / fila $c$.
4. Error de lo reconocido como $c$ = (columna $c$ − diagonal) / columna $c$.

- Ejemplo: $D = \{(3,1),(1,1),(2,2),(3,3),(1,1),(2,3),(3,2),(2,2),(3,3),(2,3)\}$.
  - Filas: $[2,0,0]$, $[0,2,2]$, $[1,1,2]$.
  - Error global $4/10$ = 40 %; reconocidas como 2: $1/3$ = 33 %; clase real 3: $2/4$ = 50 %.

**E) Curva ROC, AUC y EER**

1. Cuenta $P$ positivos y $N$ negativos.
2. Ordena por puntuación de **mayor a menor**; empieza en $(FPR, TPR) = (0, 0)$.
3. Baja la raya dato a dato: positivo → $TP \mathrel{+}= 1$; negativo → $FP \mathrel{+}= 1$.
4. Apunta $TPR = TP/P$ y $FPR = FP/N$; dibuja $TPR$ (eje $y$) frente a $FPR$ (eje $x$) hasta $(1,1)$.
5. AUC = suma de cada tramo horizontal: $\Delta FPR \times TPR$.
6. EER = $FPR$ del punto donde $FPR = 1 - TPR$.

- Ejemplo: 10 muestras alternas 1, 0, 1, 0… con puntuaciones 0,81 … 0,05.
  - La escalera sube 0,2 y avanza 0,2 alternativamente.
  - $AUC = 0{,}2\cdot(0{,}2 + 0{,}4 + 0{,}6 + 0{,}8 + 1{,}0) =$ **0,60**.
  - EER = **0,4** (en $FPR = 0{,}4$, $TPR = 0{,}6$).

**F) Curva PR y AP**

1. Ordena por puntuación de mayor a menor.
2. En cada corte: $P = TP/\hat P$ (predichos positivos) y $R = TP/P_{\text{total}}$.
3. Para cada nivel de recall $R_k$: $P_{\text{interp}}(R_k)$ = **máxima** precisión con recall $\ge R_k$.
4. $AP = \sum_k (R_k - R_{k-1})\,P_{\text{interp}}(R_k)$, con $R_0 = 0$.

- Ejemplo anterior:

| Recall | 0,2 | 0,4 | 0,6 | 0,8 | 1,0 |
|---|---|---|---|---|---|
| $P_{\text{interp}}$ | 1,000 | 0,667 | 0,600 | 0,571 | 0,556 |

  - $AP = 0{,}2\cdot(1 + 0{,}6667 + 0{,}6 + 0{,}5714 + 0{,}5556) =$ **0,6787**.

**G) $F_1$ macro, weighted y micro**

1. Para cada clase: $TP$ = diagonal; $FP$ = columna − $TP$; $FN$ = fila − $TP$.
2. $P_c = \frac{TP}{TP+FP}$; $R_c = \frac{TP}{TP+FN}$; $F_{1,c} = \frac{2P_cR_c}{P_c+R_c}$.
3. Macro = media de $F_{1,c}$; weighted = $\sum_c \frac{\text{fila}_c}{M}F_{1,c}$; micro = traza / $M$.

- Ejemplo: matriz $[[2,0,1],[0,2,0],[2,1,2]]$.

| Clase | $TP$ | $FP$ | $FN$ | $P$ | $R$ | $F_1$ | Prior |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 1 | 2/4 | 2/3 | 4/7 | 3/10 |
| 2 | 2 | 1 | 0 | 2/3 | 1 | 4/5 | 2/10 |
| 3 | 2 | 1 | 3 | 2/3 | 2/5 | 1/2 | 5/10 |

  - Macro = **0,59**; weighted = **0,58**; micro = $6/10$ = **0,60**.

<a id="r08"></a>
### T08 · Análisis discriminante

**A) Estimación por máxima verosimilitud**

1. $\pi_c = N_c/N$.
2. $\mu_c$ = media de los $x$ de la clase $c$.
3. $\Sigma_c = \frac{1}{N_c}\sum_{n:y_n=c}(x_n - \mu_c)(x_n - \mu_c)^t$ (divide por $N_c$). LDA: $\Sigma = \sum_c \pi_c\Sigma_c$.
4. Datos binarios (Bernoulli): $\mu_c$ = número de unos en la clase $c$ / $N_c$.

**B) Clasificar un punto o hallar probabilidades generativas**

1. Calcula $p(c)\,p(x\mid c)$ para cada clase.
2. Clase MAP = mayor producto.
3. $p(x)$ = suma de los productos; $p(c\mid x)$ = producto / $p(x)$.

**C) Frontera QDA entre dos gaussianas (elipse o hipérbola)**

1. Escribe $g_c(x) = \log\pi_c - \frac12\log\det\Sigma_c - \frac12(x - \mu_c)^t\Sigma_c^{-1}(x - \mu_c)$. Con priors iguales, quita $\log\pi_c$.
2. Con $\Sigma = \mathrm{diag}(s_1, s_2)$: $(x-\mu)^t\Sigma^{-1}(x-\mu) = \frac{(x_1-\mu_1)^2}{s_1} + \frac{(x_2-\mu_2)^2}{s_2}$.
3. Iguala $g_1 = g_2$, multiplica por $-2$ y pasa todo a un lado.
4. Completa cuadrados y divide para dejar "$= 1$".
   - Mismo signo en los dos cuadrados → elipse.
   - Signos opuestos → hipérbola.

- Ejemplo 1: $\mu_1 = (0,0)$, $\Sigma_1 = I$; $\mu_2 = (2,0)$, $\Sigma_2 = \mathrm{diag}(1/5, 1/4)$.
  - Queda $4(x_1 - 2{,}5)^2 + 3x_2^2 = 5 + \log 20$.
  - Centro **(2,5; 0)**; $a = \sqrt{(5 + \log 20)/4} \approx$ **1,414**; $b = \sqrt{(5 + \log 20)/3} \approx$ **1,633**.
- Ejemplo 2: $\Sigma_1 = \mathrm{diag}(1, 1/9)$, $\Sigma_2 = \mathrm{diag}(1/9, 1)$, $\mu_2 = (2,0)$.
  - Los log-determinantes son iguales y se cancelan.
  - Queda $\frac{(x_1 - 2{,}25)^2}{0{,}5625} - \frac{x_2^2}{0{,}5625} = 1$ → centro **(2,25; 0)**, **$a = b = 0{,}75$**.

**D) Regiones LDA con $\Sigma = I$ y clases equiprobables**

1. Equivale a la media más cercana: las fronteras son las mediatrices entre medias.
2. Región de una clase = intersección de los semiplanos "más cerca de mi media".

- Ejemplo: $\mu_1 = (0,0)$, $\mu_2 = (0,4)$, $\mu_3 = (4,4)$.
  - Clase 2 frente a 1: $x_2 > 2$. Frente a 3: $x_1 < 2$.
  - $R_2 = \{x : x_1 < 2,\ x_2 > 2\}$.

<a id="r09"></a>
### T09 · Regresión logística

**A) Probabilidad de una entrada**

1. Binaria: $a = w^tx + b$; $p(y=1\mid x) = \frac{1}{1 + e^{-a}}$.
   - Ejemplo: $w = (0{,}5;\ -0{,}5)$, $x = (-0{,}6;\ -1)$ → $a = 0{,}2$ → $P \approx$ **0,55**.
2. Multiclase: $a = Wx + b$ (un logit por fila); $p_c = e^{a_c}/\sum_k e^{a_k}$.
   - Ejemplo: $W = [3, -1;\ 0, 2;\ -3, -1]$, $x = (1;\ 0{,}5)$ → $a = (2{,}5;\ 1;\ -3{,}5)$ → $p = (0{,}816;\ 0{,}182;\ 0{,}002)$ → $P =$ **0,182**.

**B) Frontera y regiones de decisión**

1. Binaria: frontera $w^tx + b = 0$; clase 1 donde $w^tx + b \ge 0$.
   - Ejemplo: $w = (3,3)$, $b = 0$ → frontera $x_2 = -x_1$; $R_1 = \{x_1 + x_2 \ge 0\}$; $R_0 = \{x_1 + x_2 \le 0\}$.
2. Multiclase: escribe cada logit $a_c(x)$; $R_c = \{x : a_c \ge a_k\ \forall k \ne c\}$; simplifica.
3. Fronteras: $F_{ij} = R_i \cap R_j$; con $b = 0$, la frontera común suele ser el origen.

- Ejemplo: $W = [1, 0;\ 0, 1;\ 0, -1]$ → $a = (x_1,\ x_2,\ -x_2)$.

| Región / frontera | Resultado |
|---|---|
| $R_1$ | $x_1 \ge \lvert x_2\rvert$ |
| $R_2$ | $x_2 \ge \max(x_1, 0)$ |
| $R_3$ | $x_2 \le \min(0, -x_1)$ |
| $F_{12}$ | $x_2 = x_1$, con $x_1 \ge 0$ |
| $F_{13}$ | $x_2 = -x_1$, con $x_1 \ge 0$ |
| $F_{23}$ | $x_2 = 0$, con $x_1 \le 0$ |
| $F_{123}$ | $\{0\}$ |

**C) Una iteración de descenso por gradiente**

1. Predicciones actuales: $\hat y = \sigma(Xw)$ o $\hat Y$ = softmax por filas de $XW^t$. Con pesos a 0: $\hat y = 0{,}5$ y $\hat Y = 1/C$.
2. Error: $\hat y - y$, o $\hat Y - Y$ con $Y$ en one-hot.
3. Gradiente: $\frac1N X^t(\hat y - y)$, o $\frac1N(\hat Y - Y)^tX$.
4. Actualiza: $w \leftarrow w - \eta\,\nabla L$.

- Ejemplo binario: $X = [1, 0;\ 0, 1;\ -1, 0;\ 0, -1]$, $y = (1, 0, 0, 1)$, $\eta = 0{,}1$.
  - $\hat y - y = (-0{,}5;\ 0{,}5;\ 0{,}5;\ -0{,}5)$; $X^t(\hat y - y) = (-1;\ 1)$.
  - $w_1 = (0{,}025;\ -0{,}025)$.
- Ejemplo multiclase: $X = [1, 0;\ 0, 1;\ 0, -1]$, $y = (1, 2, 3)$, $\eta = 0{,}1$.
  - $W_1 = [0{,}0222,\ 0;\ -0{,}0111,\ 0{,}0333;\ -0{,}0111,\ -0{,}0333]$.

**D) Con ingeniería de características**

1. Transforma cada fila con $\phi(x)$, p. ej. $(1, x_1, x_2, x_1x_2)$, y usa $\Phi$ en vez de $X$.
2. Aplica la receta C. Para un solo peso: $\frac{\partial L}{\partial W_{cd}} = \frac1N\sum_n(\hat y_{nc} - y_{nc})\,\phi_d(x_n)$.

- Ejemplo del boletín ($W$ dado, $\eta = 1$): gradiente de $W_{11} \approx -0{,}0017$ → $W_{11}$ pasa de 1,3793 a **≈ 1,3810**.

<a id="r10"></a>
### T10 · Regresión lineal

**A) Densidad de una observación**

1. $\mu = w^tx + w_0$.
2. $\mathcal N(y\mid\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)$. Una densidad puede superar 1.

- Ejemplo: $w = 0{,}5$, $x = 0{,}1$, $\sigma^2 = 1/6$, $y = 0$ → $\mu = 0{,}05$ → $p \approx$ **0,97**.

**B) Mínimos cuadrados a mano**

1. Monta $X$ con columna de unos (o columnas $1, x, x^2$ si es polinómica).
2. Calcula $X^tX$ y $X^ty$.
3. Invierte $X^tX$: en $2\times2$, $\frac{1}{ad-bc}[d, -b;\ -c, a]$; en $3\times3$, adjunta / determinante.
4. $w = (X^tX)^{-1}X^ty$. Comprueba $\hat y = Xw$; si piden $\sigma^2$: $\sigma^2 = \frac1N\sum_n(y_n - \hat y_n)^2$.

- Ejemplo simple: $X = [1, -1;\ 1, 1;\ 1, 1]$, $y = (1/2,\ 0,\ -1)$.
  - $X^tX = [3, 1;\ 1, 3]$; $X^ty = (-1/2;\ -3/2)$.
  - $w = (0;\ -0{,}5)$; $\hat y = (0{,}5;\ -0{,}5;\ -0{,}5)$; $\sigma^2 = 1/6$.
- Ejemplo polinómico: $x = (-1, -1, 0, 1)$, $y = (0, 2, 0, 1)$, $\phi(x) = (1, x, x^2)$.
  - $X^tX = [4, -1, 3;\ -1, 3, -1;\ 3, -1, 3]$, $\det = 8$.
  - $w = (0, 0, 1)$ → $\hat y = x^2$; $\sigma^2 = 0{,}5$.

**C) Evaluación con $R^2$**

1. $\hat y = Xw$; $r = y - \hat y$; $RSS = \sum_n r_n^2$.
2. $\bar y$ y $TSS = \sum_n(y_n - \bar y)^2$.
3. $R^2 = 1 - RSS/TSS$; $RMSE = \sqrt{RSS/N}$.

- Ejemplo: $X = [1, 0;\ 1, 3;\ 1, 3;\ 1, 6]$, $y = (1, 1, 3, 3)$, $w = (1;\ 1/3)$.
  - $\hat y = (1, 2, 2, 3)$; $RSS = 2$; $TSS = 4$ → $R^2 =$ **0,5**.

<a id="r11"></a>
### T11 · Métodos basados en ejemplos

**A) Clasificar con KNN**

1. Distancia euclídea al cuadrado de $x$ a cada prototipo.
2. Ordénalas y quédate con las $K$ menores.
3. Cuenta votos: $p(c\mid x)$ = votos$_c$ / $K$.
4. Empate de votos → gana la clase del 1NN entre las empatadas.

- Ejemplo: $x = (4,3)$, $K = 4$, con los 12 prototipos del boletín.

| Prototipo | Clase | $d^2$ |
|---|---|---|
| $(4,1)$ | 0 | 4 |
| $(5,1)$ | 1 | 5 |
| $(6,4)$ | 1 | 5 |
| $(2,4)$ | 0 | 5 |

  - Votos 2-2 → empate → el más cercano, $(4,1)$, es clase 0 → **clase 0**.

**B) Densidades kernel y Parzen**

1. $K(u) = 0{,}3989\,\exp(-u^2/2)$; con ancho $h$: $K_h(u) = \frac1h K(u/h)$.
2. Parzen: $p(x) = \frac1N\sum_n K_h(x - x_n)$.
3. Para clasificar: Parzen por clase × prior $N_c/N$; gana la mayor.

- Ejemplo: $D = \{0, 4\}$, $h = 1$, $x = 2$ → $p = \frac12\big(K(2) + K(-2)\big) = K(2) =$ **0,0540**.

**C) Predecir con Nadaraya-Watson**

1. Calcula $K_h(x - x_n)$ para cada dato (o los logits $-\lVert x - x_n\rVert^2/(2h^2)$).
2. Normaliza para obtener los pesos $w_n$ (o aplica softmax a los logits).
3. $\hat y = \sum_n w_n\,y_n$.

- Ejemplo: $D = \{(0,1), (4,1)\}$, $x = 2$ → pesos $\frac12$ y $\frac12$ → $\hat y =$ **1**.

<a id="r12"></a>
### T12 · Árboles y ensambles

**A) Impureza y mejor corte (clasificación)**

1. Proporciones del nodo: $\pi_c = n_c/n$.
2. $G = 1 - \sum_c\pi_c^2$; $H = -\sum_c\pi_c\log_2\pi_c$.
3. Para cada característica $j$ y umbral $t$: izquierda $x_j \le t$, derecha el resto.
4. Coste $= \frac{n_L}{n}c(L) + \frac{n_R}{n}c(R)$. Gana el menor.

- Ejemplo 2d: clase 1 en $x_1 \in \{1, 3\}$ (4 datos); clase 2 en $x_1 = 2$ (2 datos).
  - $H = 0{,}918$ ($\le 1$); $G = 0{,}444$ ($> 0{,}4$).
  - $x_1 \le 1{,}5$: izquierda pura, derecha mitad y mitad → 0,667 (entropía) o 0,333 (Gini).
  - Cortar por $x_2$ no reduce nada → **$(j, t) = (1;\ 1{,}5)$**.
- Ejemplo 3d: $\{(1,1,2), (1,2,1)\}$ clase 1 y $\{(2,1,2), (2,2,1)\}$ clase 2.
  - $G = 0{,}5$ ($\le 0{,}5$); $x_1 \le 1{,}5$ separa perfectamente → **$(1;\ 1{,}5)$**.

**B) Mejor corte en regresión**

1. Coste del nodo = media de $(y - \bar y)^2$.
2. Prueba cada corte ponderando el MSE de los hijos por su tamaño.

- Ejemplo: $y = (1, 2, 1, 1)$ en $x_1 = (1, 2, 5, 5)$.
  - $c(D) = 0{,}1875$ ($> 0{,}1$).
  - $x_1 \le 3{,}5$ → hijos $\{1,2\}$ (MSE 0,25) y $\{1,1\}$ (0) → 0,125.
  - $x_1 \le 1{,}5$ → 0,167 → mejor **$(1;\ 3{,}5)$**.

**C) Predecir con boosting de mínimos cuadrados**

1. Evalúa cada tocón $F_m$ en $x$ (¿a qué lado del umbral cae?).
2. Suma: $f(x) = \sum_m F_m(x)$.

- Ejemplo 1d: $x = 2{,}5$ → $F_1 = 4$, $F_2 = -1/3$ → $f \approx$ **3,67**.
- Ejemplo 2d: $x = (4,6)$ → $\frac13 + 0{,}583 - 0{,}264$ → $f \approx$ **0,65**.

**D) AdaBoost**

1. Pesos iniciales $w_i = 1/N$.
2. $\text{err}_m$ = pesos de los mal clasificados / suma de pesos.
3. $\beta_m = \frac12\log\frac{1 - \text{err}_m}{\text{err}_m}$.
4. Multiplica por $e^{2\beta_m}$ el peso de cada mal clasificado.
5. Predicción: $f(x) = \mathrm{sgn}\big(\sum_m \beta_m F_m(x)\big)$.

- Ejemplo: $x = (4,6)$ → $F_1 = +1$, $F_2 = -1$, $F_3 = -1$.
  - $f_2 = \mathrm{sgn}(0{,}693 - 0{,}973) =$ **−1**.
  - $f_3 = \mathrm{sgn}(-0{,}280 + 0{,}896) =$ **+1**.

<a id="r13"></a>
### T13 · No supervisado

**A) Densidad de una mezcla**

1. Evalúa cada gaussiana en $x$. En 1d con $\sigma^2 = 1$: $\mathcal N(0) = 0{,}3989$, $\mathcal N(\pm1) = 0{,}2420$, $\mathcal N(\pm2) = 0{,}0540$, $\mathcal N(\pm3) = 0{,}0044$.
2. En 2d con $\Sigma$ diagonal: $\mathcal N = \dfrac{\exp\!\big(-\frac12\sum_d (x_d - \mu_d)^2/s_d\big)}{2\pi\sqrt{s_1s_2}}$.
3. $p(x) = \sum_c \pi_c\,\mathcal N(x\mid\mu_c,\Sigma_c)$.

- Ejemplo 1d: $x = 3$, $\mu = (0, 2)$, $\sigma^2 = 1$ → $0{,}5\cdot0{,}0044 + 0{,}5\cdot0{,}2420 =$ **0,123**.
- Ejemplo 2d del boletín: $p(0) \approx$ **0,056**.

**B) Una iteración de EM**

1. Paso E: para cada dato y componente, $\pi_c\,\mathcal N(x_n\mid\mu_c,\sigma_c^2)$; normaliza por filas → $r_{nc}$.
2. Paso M: $r_c = \sum_n r_{nc}$; $\pi_c = r_c/N$; $\mu_c = \sum_n r_{nc}x_n / r_c$.
3. Con las **nuevas** medias: $\sigma_c^2 = \sum_n r_{nc}(x_n - \mu_c)^2 / r_c$.

- Ejemplo: $X = (-1, 3)$, $\pi = (0{,}5;\ 0{,}5)$, $\mu = (0, 2)$, $\sigma^2 = 1$.
  - Paso E: $r = [0{,}982,\ 0{,}018;\ 0{,}018,\ 0{,}982]$.
  - Paso M: $\pi = (0{,}5;\ 0{,}5)$, $\mu = (-0{,}928;\ 2{,}928)$, $\sigma^2 = (0{,}283;\ 0{,}283)$.

**C) Una iteración de C-medias**

1. Paso E duro: cada dato va al 100 % a la media más cercana (empate exacto → según priors).
2. Paso M: $\pi_c$ = datos asignados / $N$; $\mu_c$ = media de los asignados.

- Ejemplo: $X = (-1, 3)$, $\mu = (0, 2)$.
  - $-1$ va con la media 0; $3$ va con la media 2.
  - $\pi = (0{,}5;\ 0{,}5)$, $\mu = (-1, 3)$.

---

<a id="chuleta"></a>
## 5. Chuleta para los últimos 10 minutos

**Test:** busca el detalle cambiado (signo, $=$/$\ne$, mayor/menor, %). Si las tres primeras están bien → la 4.

### T07 Decisión
- **Bayes**: $p(h\mid x) = p(h)\,p(x\mid h) / \sum_h p(h)\,p(x\mid h)$.
- **Riesgo**: $R(a\mid x) = \sum_h \ell(h,a)\,p(h\mid x)$ → mínimo. 0-1 → **MAP** $= \arg\max p(y\mid x) = \arg\max p(y)\,p(x\mid y)$.
- **Coste** FN $= c\cdot$FP → positivo si $p_1 \ge \frac{1}{1+c}$.
- **Rechazo**: $R(0) = \lambda_r$; $R(\hat y) = \lambda_e(1 - p)$; clasifica si $p^* > 1 - \lambda_r/\lambda_e$.
- **Filas** (real): $TPR$ = recall $= TP/P$; $FPR = FP/N$ (tipo I); $FNR$ (tipo II); $TNR$ = especificidad.
- **Columnas** (predicha): $PPV$ = precisión $= TP/\hat P$.
- **ROC**: $TPR$ vs $FPR$; azar = diagonal; AUC $\uparrow$; **EER $\downarrow$**.
- **PR**: $P$ vs $R$; azar $= P/M$; interpolada = max con $\tilde\tau \ge \tau$; $AP = \sum \Delta R\cdot P_{\text{interp}}$.
- **$F_\beta$** $= (1+\beta^2)\frac{PR}{\beta^2P+R}$; $F_0 = P$; $F_\infty = R$; $F_1$ = armónica.
- Macro = media · weighted $= \sum$ prior $\cdot F_1$ · **micro = traza / $M$**.
- Regresión L2 → **media** posterior.

### T08 Discriminante
- MLE: $\pi_c = N_c/N$; $\mu_c$ = media; $\Sigma_c$ dividiendo por $N_c$; Bernoulli = n.º de unos / $N_c$.
- **QDA**: $g_c = \log\pi_c - \frac12\log\det\Sigma_c - \frac12(x-\mu_c)^t\Sigma_c^{-1}(x-\mu_c)$ → cuadrática.
  - $A_c = -\frac12\Sigma_c^{-1}$; $\beta_c = \Sigma_c^{-1}\mu_c$; $\gamma_c = \log\pi_c - \frac12\log\det\Sigma_c - \frac12\mu_c^t\Sigma_c^{-1}\mu_c$.
- **LDA** ($\Sigma = \sum_c\pi_c\Sigma_c$; $\lambda = 1$): $\beta_c = \Sigma^{-1}\mu_c$; $\gamma_c = \log\pi_c - \frac12\mu_c^t\Sigma^{-1}\mu_c$ → lineal.
- **NCM** = LDA con priors iguales = media más cercana.
- **NB**: $p(x\mid c) = \prod_d p(x_d\mid c)$; NB gaussiano = QDA diagonal.
- Frontera: $g_1 = g_2$ → $\times(-2)$ → completar cuadrados (mismos signos: elipse; opuestos: hipérbola).

### T09 Logística
- $\sigma(0) = 0{,}5$; clase 1 si **$a > 0$**; $w \perp$ frontera y apunta a la clase 1.
- Softmax → MAP $= \arg\max_c a_c$; región = intersección de $C-1$ semiespacios.
- $\nabla$: $\frac1N X^t(\hat y - y)$ · $\frac1N(\hat Y - Y)^tX$; **$w \leftarrow w - \eta\nabla$**; unitario sin $\frac1N$.
- $\phi(x)$ linealiza (XOR). LR más precisa; DA más fácil y añade clases.

### T10 Regresión lineal
- Columna de **unos**; **$w = (X^tX)^{-1}X^ty$**; $\sigma^2 = RSS/N$; $\nabla = \frac2N X^t(\hat y - y)$.
- Inversa $2\times2$: $\frac{1}{ad-bc}[d, -b;\ -c, a]$.
- $RMSE = \sqrt{RSS/N}$; **$R^2 = 1 - RSS/TSS$** ($\le 1$).
- Polinómica = lineal con $\phi = (1, x, x^2, \dots)$. Riesgo mínimo $= \sigma^2$.

### T11 Ejemplos
- **KNN**: $d^2$ → $K$ menores → votos / $K$; empate → 1NN; 2NN = NN.
- $K\downarrow$ sobreajuste; $K = \sqrt N$ → Bayes; NN $\le 2\times$ Bayes; maldición de la dimensionalidad.
- **Kernel**: $K(0) = 0{,}3989$; $K(2) = 0{,}0540$; $K_h(x) = \frac1h K(x/h)$; **$h\downarrow$ sobreajuste**.
- Soporte $\lvert x\rvert \le 1$: tophat, Epanechnikov, linear, cosine.
- **Parzen** $= \frac1N\sum_n K_h(x - x_n)$; KDE variable = KNN.
- **NW**: $\hat y = \sum_n w_n y_n$; $w = \mathrm{softmax}\big(-\lVert x - x_n\rVert^2/2h^2\big)$.

### T12 Árboles
- $G = 1 - \sum\pi^2$ (máx. 0,5); $H = -\sum\pi\log_2\pi$ (máx. 1); regresión = MSE.
- Corte: $\min \sum_{\text{hijos}} \frac{n_{\text{hijo}}}{n}\,c(\text{hijo})$; $x_j \le t$ a la izquierda.
- **Bagging**: bootstrap, **63 %** / 37 % OOB, voto, $\downarrow$ varianza.
- **RF**: + variables al azar; paralelizable.
- **Boosting** ($\downarrow$ sesgo): L2 → ajusta residuos; $f = \sum_m F_m$.
- **AdaBoost**: err ponderado; $\beta = \frac12\log\frac{1-\text{err}}{\text{err}}$; fallos $\times e^{2\beta}$; $f = \mathrm{sgn}(\sum\beta F)$.

### T13 No supervisado
- GMM: $p(x) = \sum_c\pi_c\,\mathcal N(x\mid\mu_c,\Sigma_c)$. $\mathcal N(0,1)$ en 0/1/2/3 = 0,3989 / 0,2420 / 0,0540 / 0,0044.
- **E**: $r_{nc} = \pi_c\mathcal N_c(x_n)/\sum_k\pi_k\mathcal N_k(x_n)$. **M**: $\pi_c = r_c/N$; $\mu_c = \sum_n r_{nc}x_n/r_c$; $\Sigma_c$ con la **nueva** $\mu_c$.
- Inicio: $\pi_c = 1/C$; $\mu_c$ = datos; $\Sigma_c = \hat\Sigma/C$. EM nunca baja $LL$ (MM/ELBO); óptimos locales.
- **C-medias** = EM con $\sigma^2I$, $\sigma^2 \to 0$: asignación dura; $\mu_c$ = media del grupo. $\sigma^2 \to \infty$: $r \to$ priors.
